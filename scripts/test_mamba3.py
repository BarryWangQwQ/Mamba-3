"""Numerical self-checks and benchmarks for mamba3.py.

Run directly: python scripts/test_mamba3.py

The self-checks cover:
  1. The chunked parallel scan agrees with the step-by-step recurrence in both
     forward values and gradients (training takes the chunked path, decoding
     takes the recurrent one)
  2. Segmented resumption through inference_params, and the official step()
     path, both reproduce a single full-sequence forward
  3. The decode step's dedicated single-token path is bit-identical to the
     general scan evaluated at length one
  4. Key conventions line up with the official mamba_ssm: state shapes, bias
     layouts, norm semantics, Block return values
  5. CPU and the accelerator agree, so no backend is a special case
  6. Graph decoding matches call-by-call forwards, and states can be reset
     per sample
  7. A torch.compile'd model still works through the graph cache, unchanged

Whichever accelerator the torch build exposes is used, CUDA or otherwise; checks
6 and 7 need graph capture, which only some backends implement, and are skipped
where it is missing rather than where the device is not CUDA.
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# python puts this file's directory on sys.path, not the one it was launched
# from, so point at the repo root to reach mamba3.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mamba3 import (
    Block,
    GatedMLP,
    InferenceParams,
    Mamba3,
    MixerModel,
    _chunk_scan,
    _mamba3_combined,
    _mamba3_step,
    _new_graph,
    _recurrent_scan,
    create_block,
    initialize_states,
    mamba3_siso_combined,
    rms_norm_ref,
    update_graph_cache,
)


def _pick_device() -> torch.device:
    """The accelerator this build actually exposes, else CPU.

    current_accelerator() names the compiled-in backend whether or not it can be
    reached (a torch built with MPS reports mps on a machine where Metal is
    unavailable), so is_available() is the part that decides.
    """
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def _graph_capture_supported(device) -> bool:
    """Whether this backend can capture an accelerator graph at all.

    Asks the same _new_graph that capture_graph uses, so the answer tracks
    whatever mamba3.py can actually reach rather than a list of device names.
    Only CUDA and XPU register an implementation; on MPS, for one, the neutral
    API reports "Graph is not supported on device type: mps" and the CUDA
    fallback cannot be instantiated, so graph decoding is unavailable there.
    """
    if device.type == "cpu":
        return False
    try:
        _new_graph()
    except Exception:
        return False
    return True


def _device_label(device) -> str:
    """A printable name for the benchmark header."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return "Apple Silicon, Metal"
    return str(device)


def _scan_inputs(device, seqlen=40, rank=1, requires_grad=False):
    """Build a valid set of scan inputs (ADT < 0, alpha/beta > 0)."""
    batch, nheads, headdim, d_state = 3, 4, 16, 32
    randn = lambda *s: torch.randn(*s, device=device)  # noqa: E731
    trap = torch.sigmoid(randn(batch, seqlen, nheads))
    dt = F.softplus(randn(batch, seqlen, nheads)) * 0.1 + 1e-4
    tensors = [
        randn(batch, seqlen, rank, nheads, headdim),        # V
        randn(batch, seqlen, rank, nheads, d_state),        # K
        randn(batch, seqlen, rank, nheads, d_state),        # Q
        -F.softplus(randn(batch, seqlen, nheads)) * 0.1 - 1e-4,  # ADT
        dt * (1.0 - 0.5 * trap),                            # alpha
        dt * (0.5 * trap),                                  # beta
    ]
    if requires_grad:
        tensors = [t.detach().requires_grad_(True) for t in tensors]
    state = (
        randn(batch, nheads, headdim, d_state),
        randn(batch, rank, nheads, headdim),
        randn(batch, rank, nheads, d_state),
    )
    return tensors, state


def test_scan(device) -> None:
    """Compare the two implementations directly at the scan level."""
    print("--- Selective scan: chunked vs step-by-step recurrence ---")

    for rank in (1, 2, 4):
        args, state = _scan_inputs(device, rank=rank)
        y_ref, h_ref = _recurrent_scan(*args, *state)

        for q in (1, 3, 8, 40, 128):
            y, h = _chunk_scan(*args, *state, chunk_size=q)
            ey = (y - y_ref).abs().max().item()
            eh = (h - h_ref).abs().max().item()
            assert ey < 1e-4 and eh < 1e-4, f"rank={rank}, chunk_size={q}: forward mismatch"
        print(f"  R={rank}  forward max|d| = {ey:.3e} (Y), {eh:.3e} (state); "
              f"agrees for chunk sizes 1..128")

        args_a, state = _scan_inputs(device, rank=rank, requires_grad=True)
        args_b = [t.detach().clone().requires_grad_(True) for t in args_a]
        # Random weights constrain Y and the final state at once, so errors in
        # some directions cannot cancel out under summation.
        wy, wh = torch.randn_like(args_a[0]), torch.randn_like(state[0])

        y_a, h_a = _recurrent_scan(*args_a, *state)
        g_a = torch.autograd.grad((y_a * wy).sum() + (h_a * wh).sum(), args_a)
        y_b, h_b = _chunk_scan(*args_b, *state, chunk_size=8)
        g_b = torch.autograd.grad((y_b * wy).sum() + (h_b * wh).sum(), args_b)

        err = max((a - b).abs().max().item() for a, b in zip(g_a, g_b))
        scale = max(a.abs().max().item() for a in g_a)
        print(f"        gradient max|d| = {err:.3e}  (|g|max = {scale:.3e})")
        assert err < 1e-4 * max(scale, 1.0), f"rank={rank}: gradient mismatch"


def test_mamba3(device) -> None:
    """Check inference_params resumption and single-step decoding on a full layer."""
    variants = [
        ("SISO", dict()),
        ("SISO + outproj_norm", dict(is_outproj_norm=True)),
        ("SISO + rope_fraction=1", dict(rope_fraction=1.0)),
        ("MIMO(R=2)", dict(is_mimo=True, mimo_rank=2, chunk_size=32)),
        ("MIMO(R=4) + norm", dict(is_mimo=True, mimo_rank=4, chunk_size=16,
                                  is_outproj_norm=True)),
        ("MIMO(R=4) unfused norm", dict(is_mimo=True, mimo_rank=4, chunk_size=16,
                                        is_outproj_norm=True,
                                        fuse_pregate_headwise_norm=False)),
    ]

    print("\n--- Full layer: inference_params resumption and single-step decoding ---")
    for tag, overrides in variants:
        mixer = Mamba3(d_model=128, d_state=32, headdim=32, layer_idx=0,
                       **overrides).to(device).float()
        u = torch.randn(3, 40, mixer.d_model, device=device)

        with torch.no_grad():
            y_full = mixer(u)

            # Segmented forward: seqlen_offset accumulates, and chaining the
            # states should reproduce the single-shot forward.
            params = InferenceParams(max_seqlen=40, max_batch_size=3)
            y_a = mixer(u[:, :17], inference_params=params)
            params.seqlen_offset += 17
            y_b = mixer(u[:, 17:], inference_params=params)
            e_seg = (torch.cat([y_a, y_b], dim=1) - y_full).abs().max().item()

            # Stepwise decoding: takes the official step path, and should
            # reproduce the whole sequence frame by frame.
            params = InferenceParams(max_seqlen=40, max_batch_size=3)
            states = mixer._get_states_from_cache(params, 3)
            outs = []
            for t in range(u.shape[1]):
                out_t, *_ = mixer.step(u[:, t], *states)
                outs.append(out_t)
            e_step = (torch.stack(outs, dim=1) - y_full).abs().max().item()

        print(f"  {tag:<24} segmented max|d| = {e_seg:.3e}   stepwise max|d| = {e_step:.3e}")
        assert e_seg < 2e-4 and e_step < 2e-4, f"{tag}: state path mismatch"


def test_step_dedicated_path(device) -> None:
    """Check the decode step's dedicated path against the general scan at L=1.

    _mamba3_step drops the length axis _mamba3_combined carries, which is only
    sound if it reorders no arithmetic: a cumsum over one element, a stack of one
    tensor and a rank-one einsum are replaced by their values, not by anything
    cheaper. So this asserts equality bit for bit, not within a tolerance.
    """
    variants = [
        ("SISO", dict()),
        ("SISO + outproj_norm", dict(is_outproj_norm=True)),
        ("SISO + rope_fraction=1", dict(rope_fraction=1.0)),
        ("MIMO(R=4)", dict(is_mimo=True, mimo_rank=4, chunk_size=16)),
        ("MIMO(R=4) + norm", dict(is_mimo=True, mimo_rank=4, chunk_size=16,
                                  is_outproj_norm=True)),
        ("MIMO(R=4) unfused norm", dict(is_mimo=True, mimo_rank=4, chunk_size=16,
                                        is_outproj_norm=True,
                                        fuse_pregate_headwise_norm=False)),
    ]

    print("\n--- Decode step: dedicated path vs the general scan at L=1 ---")
    for tag, overrides in variants:
        m = Mamba3(d_model=128, d_state=32, headdim=32, layer_idx=0,
                   **overrides).to(device).float()
        batch = 3

        with torch.no_grad():
            states = m.allocate_inference_cache(batch, 8)
            for s in states:  # non-zero, so the resume terms are exercised
                s.normal_()
            u = torch.randn(batch, m.d_model, device=device)

            z, x, B, C, dd_dt, dd_A, trap_logit, angles = m._split_in_proj(u)
            DT, B, C, x, z, trap, A, angles = m._preprocess(
                dd_A, dd_dt, B, C, x, z, trap_logit, angles
            )

            if m.is_mimo:
                gated = m.fuse_pregate_headwise_norm or not m.is_outproj_norm
                shared = dict(
                    rotate_pairwise=False, MIMO_V=m.mimo_x, MIMO_Z=m.mimo_z,
                    MIMO_Out=m.mimo_o if gated else None,
                    outproj_norm_weight=(
                        m.norm.weight if m.fuse_pregate_headwise_norm else None),
                    outproj_norm_eps=(
                        m.norm.eps if m.fuse_pregate_headwise_norm else 1e-5),
                )
                Z = z if gated else None
            else:
                shared = dict(rotate_pairwise=True, MIMO_V=None, MIMO_Z=None,
                              MIMO_Out=None, outproj_norm_weight=None,
                              outproj_norm_eps=1e-5)
                Z = z if not m.is_outproj_norm else None

            # The step path takes the gate, the batched one takes the logit and
            # applies the sigmoid itself; both end up on the same float32 value.
            got = _mamba3_step(
                Q=C, K=B, V=x, ADT=A * DT, DT=DT, Trap=trap,
                Q_bias=m.C_bias, K_bias=m.B_bias, Angles=angles, D=m.D, Z=Z,
                Input_States=states, **shared,
            )
            ref = _mamba3_combined(
                C.unsqueeze(1), B.unsqueeze(1), x.unsqueeze(1),
                (A * DT).unsqueeze(-1), DT.unsqueeze(-1),
                trap_logit.unsqueeze(-1), m.C_bias, m.B_bias,
                angles.unsqueeze(1), m.D, None if Z is None else Z.unsqueeze(1),
                chunk_size=m.chunk_size, Input_States=states, **shared,
            )
            ref = (ref[0].squeeze(1),) + tuple(ref[1:])

        exact = all(torch.equal(a, b) for a, b in zip(got, ref))
        worst = max((a - b).abs().max().item() for a, b in zip(got, ref))
        print(f"  {tag:<24} {'bit-identical' if exact else 'DIFFERS':<15} "
              f"max|d| = {worst:.3e}")
        assert exact, f"{tag}: dedicated decode path diverges from the general scan"


def test_official_conventions(device) -> None:
    """Verify the parts that must agree with the official implementation."""
    print("\n--- Conventions shared with the official implementation ---")
    siso = Mamba3(d_model=128, d_state=32, headdim=32, layer_idx=0).to(device)
    mimo = Mamba3(d_model=128, d_state=32, headdim=32, is_mimo=True, mimo_rank=4,
                  chunk_size=16, is_outproj_norm=True, layer_idx=0).to(device)

    angle_dt_state, ssm_state, k_state, v_state = mimo.allocate_inference_cache(2, 16)
    print(f"  allocate_inference_cache: angle_dt_state={tuple(angle_dt_state.shape)} "
          f"ssm_state={tuple(ssm_state.shape)} k_state={tuple(k_state.shape)} "
          f"v_state={tuple(v_state.shape)}")
    assert angle_dt_state.shape == (2, mimo.nheads, mimo.num_rope_angles)
    assert ssm_state.shape == (2, mimo.nheads, mimo.headdim, 32)
    assert k_state.shape == (2, 4, mimo.nheads, 32)
    assert v_state.shape == (2, mimo.nheads, mimo.headdim)
    assert siso.allocate_inference_cache(2, 16)[2].shape == (2, 1, siso.nheads, 32)

    assert siso.B_bias.shape == (siso.nheads, 1, 32), "unexpected B_bias shape"
    assert mimo.C_bias.shape == (mimo.nheads, 4, 32), "unexpected C_bias shape"
    assert mimo.norm.weight.numel() == mimo.nheads * mimo.headdim, (
        "unexpected outproj norm weight count"
    )
    assert mimo.norm.group_size == mimo.headdim and mimo.norm.norm_before_gate
    assert (siso.rotary_dim_divisor, siso.split_tensor_size, siso.num_rope_angles) == (4, 16, 8)
    assert siso.mimo_rank == 1 and mimo.mimo_rank == 4
    assert mimo.fuse_pregate_headwise_norm
    print(f"  in_proj output {siso.in_proj.out_features} = 2*d_inner + 2*N*G*R + 3*H + S")
    print(f"  rope: rotary_dim_divisor={siso.rotary_dim_divisor}, "
          f"split_tensor_size={siso.split_tensor_size}, "
          f"num_rope_angles={siso.num_rope_angles}")

    # The combined scan can be called standalone with the official signature.
    batch, seqlen, nheads, headdim, d_state = 2, 12, siso.nheads, 32, 32
    y = mamba3_siso_combined(
        Q=torch.randn(batch, seqlen, 1, d_state, device=device),
        K=torch.randn(batch, seqlen, 1, d_state, device=device),
        V=torch.randn(batch, seqlen, nheads, headdim, device=device),
        ADT=-F.softplus(torch.randn(batch, nheads, seqlen, device=device)) - 1e-4,
        DT=F.softplus(torch.randn(batch, nheads, seqlen, device=device)) + 1e-4,
        Trap=torch.randn(batch, nheads, seqlen, device=device),
        Q_bias=torch.ones(nheads, d_state, device=device),
        K_bias=torch.ones(nheads, d_state, device=device),
        Angles=torch.randn(batch, seqlen, nheads, 8, device=device),
        D=torch.ones(nheads, device=device),
        Z=torch.randn(batch, seqlen, nheads, headdim, device=device),
    )
    assert y.shape == (batch, seqlen, nheads, headdim)
    print(f"  standalone mamba3_siso_combined -> {tuple(y.shape)}")

    # Block returns (hidden_states, residual) per the official convention.
    block = create_block(128, d_intermediate=256, rms_norm=True,
                         ssm_cfg=dict(d_state=32, headdim=32), layer_idx=0).to(device).float()
    x = torch.randn(2, 33, 128, device=device)
    hidden_states, residual = block(x)
    assert isinstance(block, Block)
    assert hidden_states.shape == x.shape and residual.shape == x.shape
    assert isinstance(block.mlp, GatedMLP) and block.mlp.fc2.out_features == 128
    print(f"  create_block -> Block returns (hidden_states, residual) = "
          f"({tuple(hidden_states.shape)}, {tuple(residual.shape)})")

    # MixerModel provides the contract capture_graph relies on.
    model = MixerModel(128, n_layer=2, ssm_cfg=dict(d_state=32, headdim=32)).to(device).float()
    assert model(x).shape == x.shape
    cache = model.allocate_inference_cache(2, 16)
    assert set(cache) == {0, 1} and len(cache[0]) == 4
    print(f"  MixerModel output {tuple(model(x).shape)}, "
          f"inference cache laid out as {{layer_idx: 4-tuple}}")

    # Gated semantics agree with the official rms_norm_ref.
    w = torch.randn(64, device=device)
    a, b = torch.randn(5, 64, device=device), torch.randn(5, 64, device=device)
    ref = (a.float() / a.float().square().mean(-1, keepdim=True).add(1e-5).sqrt()) * w * F.silu(b.float())
    assert (rms_norm_ref(a, w, None, z=b, eps=1e-5) - ref).abs().max() < 1e-5
    print("  rms_norm_ref: norm_before_gate=True semantics match the official one")


def test_backend_parity(device) -> None:
    """The same weights must give the same answers on CPU and on the accelerator."""
    print("\n--- CPU vs accelerator: same file, same numbers ---")
    if device.type == "cpu":
        print("  skipped (no accelerator to compare against).")
        return

    torch.manual_seed(0)
    ref = Mamba3(d_model=128, d_state=32, headdim=32, is_mimo=True, mimo_rank=2,
                 layer_idx=0).float()
    x0 = torch.randn(2, 64, 128)
    token0 = torch.randn(2, 1, 128)

    out = {}
    for dev in (torch.device("cpu"), device):
        layer = copy.deepcopy(ref).to(dev)
        x = x0.clone().to(dev).requires_grad_(True)
        y = layer(x)
        y.sum().backward()
        with torch.inference_mode():
            params = InferenceParams(max_seqlen=128, max_batch_size=2)
            layer(x0.to(dev), inference_params=params)
            params.seqlen_offset = x0.shape[1]
            step = layer(token0.to(dev), inference_params=params)
        out[dev.type] = (y.detach().cpu(), x.grad.cpu(), step.cpu())

    for i, name in enumerate(("forward", "backward", "decode step")):
        a, b = out["cpu"][i], out[device.type][i]
        err = (a - b).abs().max().item()
        print(f"  {name:<12s} cpu vs {device.type} max|d| = {err:.3e}")
        assert err < 1e-4, f"{name} disagrees across backends"


def test_compiled_decode(device) -> None:
    """A compiled model must pass through the graph cache and still match eager.

    Nothing in mamba3.py mentions torch.compile: this guards that it stays that
    way, i.e. that update_graph_cache keeps working on the OptimizedModule
    wrapper rather than reaching past it.
    """
    print("\n--- torch.compile through the graph cache ---")
    if not _graph_capture_supported(device):
        print(f"  skipped ({device.type} has no graph capture backend).")
        return
    import torch._dynamo as dynamo
    if not dynamo.is_inductor_supported():
        print("  skipped (inductor unavailable here).")
        return

    def build():
        torch.manual_seed(0)
        return MixerModel(256, n_layer=3, rms_norm=True,
                          ssm_cfg=dict(d_state=32, headdim=64)).to(device).eval().float()

    batch, nsteps, max_seqlen = 4, 32, 64
    torch.manual_seed(1)
    frames = [torch.randn(batch, 1, 256, device=device) for _ in range(nsteps)]

    with torch.inference_mode():
        model = build()
        params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch)
        ref = []
        for x in frames:
            ref.append(model(x, inference_params=params).clone())
            params.seqlen_offset += 1
        ref_state = [s[1].clone() for s in params.key_value_memory_dict.values()]

        cache = update_graph_cache(torch.compile(build(), fullgraph=True), None,
                                   batch, 0, max_seqlen)
        got = [cache.run(x) for x in frames]
        got_state = [s[1].clone()
                     for s in cache.inference_params.key_value_memory_dict.values()]

    # Checked after a run of steps, not one, because a decode error small enough
    # to pass on frame one can still compound through the state.
    err = max((a - b).abs().max().item() for a, b in zip(ref, got))
    serr = max((a - b).abs().max().item() for a, b in zip(ref_state, got_state))
    print(f"  {nsteps} steps, compiled + graph vs eager: max|d| = {err:.3e}")
    print(f"  ssm state after {nsteps} steps:            max|d| = {serr:.3e}")
    assert err < 1e-5, "compiled graph decode diverges from eager"
    assert serr < 1e-5, "compiled graph decode leaves a different state"
    dynamo.reset()


@torch.inference_mode()
def test_graph_decoding(device) -> None:
    """Graph decoding must give the same results as calling forward directly."""
    print("\n--- Graph decoding ---")
    if not _graph_capture_supported(device):
        print(f"  skipped ({device.type} has no graph capture backend).")
        return

    torch.manual_seed(0)
    model = MixerModel(128, n_layer=3, rms_norm=True,
                       ssm_cfg=dict(d_state=32, headdim=32)).to(device)
    model.eval().float()
    batch, nsteps = 6, 10
    frames = [torch.randn(batch, 1, 128, device=device) for _ in range(nsteps)]

    # Reference: no graph, plain frame-by-frame forward.
    params = InferenceParams(max_seqlen=64, max_batch_size=batch)
    ref = []
    for x in frames:
        ref.append(model(x, inference_params=params).clone())
        params.seqlen_offset += 1

    cache = update_graph_cache(model, None, batch, 0, 64)
    got = [cache.run(x) for x in frames]
    err = max((a - b).abs().max().item() for a, b in zip(ref, got))
    print(f"  {nsteps} frames, max|d| = {err:.3e}")
    assert err < 1e-5, "graph replay differs from call-by-call forward"

    # After a full reset, replaying the first frame should match starting from
    # a zero state.
    initialize_states(cache.inference_params)
    assert (cache.run(frames[0]) - ref[0]).abs().max().item() < 1e-5
    print("  first frame matches again after initialize_states")

    # Per-sample reset: zero the first 3, keep the rest (used for clip
    # boundaries and state dropout).
    mask = torch.zeros(batch, dtype=torch.bool, device=device)
    mask[:3] = True
    initialize_states(cache.inference_params, mask)
    ssm = [s[1] for s in cache.inference_params.key_value_memory_dict.values()]
    assert all(s[:3].abs().max().item() == 0 for s in ssm)
    assert any(s[3:].abs().max().item() > 0 for s in ssm)
    print("  per-sample initialize_states works (first 3 zeroed, rest kept)")


def benchmark(device) -> None:
    if device.type == "cpu":
        print("\nSkipping benchmarks (requires an accelerator).")
        return

    import time

    def timeit(fn, iters=20, warmup=5) -> float:
        for _ in range(warmup):
            fn()
        torch.accelerator.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.accelerator.synchronize()
        return (time.perf_counter() - start) / iters * 1000.0

    print()
    print("=" * 72)
    print(f"Benchmark  B=8, d_model=384, headdim=64, d_state=64  "
          f"({_device_label(device)})")
    print("=" * 72)
    print(f"{'config':<14}{'seqlen':>8}{'recurrent':>12}{'chunked':>12}"
          f"{'speedup':>10}{'train':>10}")
    print("-" * 72)

    for is_mimo, rank in ((False, 1), (True, 2)):
        tag = f"MIMO(R={rank})" if is_mimo else "SISO"
        mixer = Mamba3(d_model=384, d_state=64, headdim=64, is_mimo=is_mimo,
                       mimo_rank=rank, chunk_size=64 // rank,
                       layer_idx=0).to(device).float()

        for seqlen in (32, 128, 512):
            u = torch.randn(8, seqlen, mixer.d_model, device=device)
            args, state = _scan_inputs(device, seqlen=seqlen, rank=rank)
            t_rec = timeit(lambda: _recurrent_scan(*args, *state))
            t_chunk = timeit(lambda: _chunk_scan(*args, *state, chunk_size=mixer.chunk_size))

            def train_step():
                mixer.zero_grad(set_to_none=True)
                mixer(u).square().mean().backward()

            t_train = timeit(train_step, iters=10, warmup=3)
            print(f"{tag:<14}{seqlen:>8}{t_rec:>10.2f}ms{t_chunk:>10.2f}ms"
                  f"{t_rec / t_chunk:>9.1f}x{t_train:>8.2f}ms")

    # Graph replay is a CUDA/XPU column: elsewhere there is no capture backend
    # to compare the plain forward against, so the table narrows instead.
    graphs = _graph_capture_supported(device)

    print()
    print("=" * 72)
    print(f"Single-step decode latency: plain forward"
          f"{' vs graph replay' if graphs else ''} "
          f"(3 layers, batch 64, d_model=320)")
    if not graphs:
        print(f"({device.type} has no graph capture backend; "
              f"the 60FPS budget is measured against the plain forward)")
    print("=" * 72)
    if graphs:
        print(f"{'d_state':>9}{'forward':>14}{'graph replay':>14}{'speedup':>10}"
              f"{'of 60FPS':>12}")
    else:
        print(f"{'d_state':>9}{'forward':>14}{'of 60FPS':>12}")
    print("-" * 72)

    for d_state in (16, 32, 64):
        model = MixerModel(320, n_layer=3, rms_norm=True,
                           ssm_cfg=dict(d_state=d_state, headdim=64)).to(device)
        model.eval().float()
        x = torch.randn(64, 1, 320, device=device)

        with torch.inference_mode():
            params = InferenceParams(max_seqlen=64, max_batch_size=64)
            params.key_value_memory_dict = model.allocate_inference_cache(64, 64)
            params.seqlen_offset = 1
            t_eager = timeit(lambda: model(x, inference_params=params), iters=100, warmup=30)

            if graphs:
                cache = update_graph_cache(model, None, 64, 0, 64)
                t_graph = timeit(lambda: cache.run(x), iters=100, warmup=30)

        if graphs:
            print(f"{d_state:>9}{t_eager:>12.3f}ms{t_graph:>12.3f}ms"
                  f"{t_eager / t_graph:>9.1f}x{t_graph / 16.7 * 100:>11.2f}%")
        else:
            print(f"{d_state:>9}{t_eager:>12.3f}ms"
                  f"{t_eager / 16.7 * 100:>11.2f}%")


if __name__ == "__main__":
    dev = _pick_device()

    print("=" * 72)
    print(f"Numerical self-check  (device={dev}, torch={torch.__version__})")
    print("=" * 72)

    # Disable TF32 for the exact comparisons: the chunked path uses matmul and
    # the recurrent one uses einsum, and TF32 widens the gap between them.
    _tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(0)
        test_scan(dev)
        test_mamba3(dev)
        test_step_dedicated_path(dev)
        test_official_conventions(dev)
        test_backend_parity(dev)
        test_graph_decoding(dev)
        test_compiled_decode(dev)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32

    print("\nAll self-checks passed.")
    benchmark(dev)
