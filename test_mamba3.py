"""Numerical self-checks and benchmarks for mamba3.py.

Run directly: python test_mamba3.py

The self-checks cover:
  1. The chunked parallel scan agrees with the step-by-step recurrence in both
     forward values and gradients (training takes the chunked path, decoding
     takes the recurrent one)
  2. Segmented resumption through inference_params, and the official step()
     path, both reproduce a single full-sequence forward
  3. Key conventions line up with the official mamba_ssm: state shapes, bias
     layouts, norm semantics, Block return values
  4. CUDA graph decoding matches call-by-call forwards, and states can be reset
     per sample
"""

import torch
import torch.nn.functional as F

from mamba3 import (
    Block,
    GatedMLP,
    InferenceParams,
    Mamba3,
    MixerModel,
    _chunk_scan,
    _recurrent_scan,
    create_block,
    initialize_states,
    mamba3_siso_combined,
    rms_norm_ref,
    update_graph_cache,
)


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


@torch.inference_mode()
def test_cuda_graph(device) -> None:
    """CUDA graph decoding must give the same results as calling forward directly."""
    print("\n--- CUDA graph decoding ---")
    if device.type != "cuda":
        print("  skipped (requires CUDA).")
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
    assert err < 1e-5, "CUDA graph result differs from call-by-call forward"

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
    if device.type != "cuda":
        print("\nSkipping benchmarks (requires CUDA).")
        return

    import time

    def timeit(fn, iters=20, warmup=5) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1000.0

    print()
    print("=" * 72)
    print(f"Benchmark  B=8, d_model=384, headdim=64, d_state=64  "
          f"({torch.cuda.get_device_name(0)})")
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

    print()
    print("=" * 72)
    print("Single-step decode latency: plain forward vs CUDA graph "
          "(3 layers, batch 64, d_model=320)")
    print("=" * 72)
    print(f"{'d_state':>9}{'forward':>14}{'CUDA graph':>14}{'speedup':>10}"
          f"{'of 60FPS':>12}")
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

            cache = update_graph_cache(model, None, 64, 0, 64)
            t_graph = timeit(lambda: cache.run(x), iters=100, warmup=30)

        print(f"{d_state:>9}{t_eager:>12.3f}ms{t_graph:>12.3f}ms"
              f"{t_eager / t_graph:>9.1f}x{t_graph / 16.7 * 100:>11.2f}%")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        test_official_conventions(dev)
        test_cuda_graph(dev)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32

    print("\nAll self-checks passed.")
    benchmark(dev)
