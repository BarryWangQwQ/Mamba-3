"""Measure and plot the benchmark figures shown in README.md.

    python bench_figures.py            # measure on GPU, then plot
    python bench_figures.py --replot   # redraw from assets/results.json only

Writes assets/*.png and assets/results.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba3 import (
    InferenceParams,
    Mamba3,
    MixerModel,
    _chunk_scan,
    _recurrent_scan,
    initialize_states,
    update_graph_cache,
)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
INK = "#1A1D23"
MUTED = "#5C6370"
HAIR = "#D5D8DE"
GRID = "#EEF0F3"
TEAL = "#0F7B6C"
CORAL = "#C45C26"
NAVY = "#1E3A5F"
BLUE = "#1D4ED8"
SLATE = "#64748B"
PLUM = "#7C3AED"

# Figures are read on GitHub, often on a phone where the image is scaled to the
# device width. A narrow canvas with large type survives that; a wide one does not.
WIDTH = 6.8

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 12.0,
    "axes.titlesize": 13.0,
    "axes.labelsize": 12.0,
    "axes.labelcolor": INK,
    "axes.edgecolor": HAIR,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "legend.fontsize": 10.8,
    "lines.linewidth": 2.2,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.22,
})


def _style(ax, *, log=False):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, which="major")
    if log:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, which="minor")
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", pad=4)


def _single(height=4.2):
    return plt.subplots(figsize=(WIDTH, height))


def _stacked(height=7.0, ratios=(1, 1), share_x=False):
    """Two panels one above the other.

    Stacking keeps the image close to square, so a phone that scales it to the
    device width does not shrink the type as much as a side-by-side layout would.
    """
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, height), sharex=share_x,
                             gridspec_kw={"height_ratios": list(ratios)})
    fig.subplots_adjust(hspace=0.30 if share_x else 0.44)
    return fig, axes


def _heading(fig, title: str, *subtitle: str, axes_title: bool = True) -> None:
    """Title and subtitle in the margin reserved above the axes.

    The reserved band includes room for the first panel's own title, which is
    drawn above its axes and would otherwise land on the subtitle.
    """
    h = fig.get_figheight()
    fig.subplots_adjust(top=1.0 - (0.30 + 0.32 * axes_title + 0.21 * len(subtitle)) / h)
    y = 1.0 - 0.04 / h
    fig.text(0.0, y, title, fontsize=15.0, fontweight="bold", color=INK, va="top")
    for i, line in enumerate(subtitle):
        y -= (0.30 if i == 0 else 0.21) / h
        fig.text(0.0, y, line, fontsize=11.0, color=MUTED, va="top")


def _footer(fig, *lines: str) -> None:
    h = fig.get_figheight()
    y = -0.06 / h
    for line in lines:
        fig.text(0.0, y, line, fontsize=9.6, color=MUTED, va="top")
        y -= 0.17 / h


def _save(fig, name: str) -> None:
    fig.savefig(ASSETS / name)
    plt.close(fig)
    print(f"wrote {name}")


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------
def timeit(fn, iters=20, warmup=5, reps=5, budget_s=2.0) -> float:
    """Median of `reps` timed blocks of `iters` calls, in ms per call.

    One block is not enough at these sizes. Most calls here are launch bound and
    land near a millisecond, where GPU clock drift alone moves a single block by
    more than 20% — enough to invent a regression that is not in the code. The
    median across blocks is what makes a number reproducible run to run.

    `budget_s` caps the extra cost: a callable slow enough to have averaged out
    on its own stops after three blocks instead of being re-timed for minutes.

    This does not make sub-millisecond numbers exact. What is left after the
    median is drift on a scale of tens of seconds, which more blocks do not
    remove because neighbouring blocks are correlated; expect the last digit of
    a ~1 ms reading to move by about 10% between runs.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times, spent = [], 0.0
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        block = time.perf_counter() - start
        spent += block
        times.append(block / iters * 1000.0)
        if spent > budget_s and len(times) >= 3:
            break
    return statistics.median(times)


def peak_mem_mb(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def scan_inputs(device, seqlen=40, rank=1, batch=3, nheads=4, headdim=16, d_state=32,
                requires_grad=False):
    randn = lambda *s: torch.randn(*s, device=device)  # noqa: E731
    trap = torch.sigmoid(randn(batch, seqlen, nheads))
    dt = F.softplus(randn(batch, seqlen, nheads)) * 0.1 + 1e-4
    tensors = [
        randn(batch, seqlen, rank, nheads, headdim),
        randn(batch, seqlen, rank, nheads, d_state),
        randn(batch, seqlen, rank, nheads, d_state),
        -F.softplus(randn(batch, seqlen, nheads)) * 0.1 - 1e-4,
        dt * (1.0 - 0.5 * trap),
        dt * (0.5 * trap),
    ]
    if requires_grad:
        tensors = [t.detach().requires_grad_(True) for t in tensors]
    state = (
        randn(batch, nheads, headdim, d_state),
        randn(batch, rank, nheads, headdim),
        randn(batch, rank, nheads, d_state),
    )
    return tensors, state


class CausalAttention(nn.Module):
    """Causal MHA through PyTorch SDPA (FlashAttention when available)."""

    def __init__(self, d_model: int, nheads: int):
        super().__init__()
        self.nheads = nheads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.nheads, D // self.nheads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, L, D))


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def measure(device) -> dict:
    torch.manual_seed(0)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False

    print("-- scan alignment (chunked vs recurrence)")
    scan_alignment = []
    for rank in (1, 2, 4):
        args, state = scan_inputs(device, rank=rank)
        y_ref, h_ref = _recurrent_scan(*args, *state)
        y, h = _chunk_scan(*args, *state, chunk_size=8)
        args_a, state = scan_inputs(device, rank=rank, requires_grad=True)
        args_b = [t.detach().clone().requires_grad_(True) for t in args_a]
        wy, wh = torch.randn_like(args_a[0]), torch.randn_like(state[0])
        y_a, h_a = _recurrent_scan(*args_a, *state)
        g_a = torch.autograd.grad((y_a * wy).sum() + (h_a * wh).sum(), args_a)
        y_b, h_b = _chunk_scan(*args_b, *state, chunk_size=8)
        g_b = torch.autograd.grad((y_b * wy).sum() + (h_b * wh).sum(), args_b)
        gerr = max((a - b).abs().max().item() for a, b in zip(g_a, g_b))
        gscale = max(a.abs().max().item() for a in g_a)
        scan_alignment.append({
            "rank": rank,
            "fwd_y": (y - y_ref).abs().max().item(),
            "fwd_h": (h - h_ref).abs().max().item(),
            "grad": gerr,
            "grad_scale": gscale,
            "grad_rel": gerr / gscale,
        })
        print(f"   R={rank}  {scan_alignment[-1]}")

    print("-- chunk-size sweep")
    chunk_sweep = []
    for rank in (1, 2, 4):
        args, state = scan_inputs(device, seqlen=256, rank=rank)
        y_ref, h_ref = _recurrent_scan(*args, *state)
        for q in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            y, h = _chunk_scan(*args, *state, chunk_size=q)
            chunk_sweep.append({
                "rank": rank,
                "chunk_size": q,
                "err": max((y - y_ref).abs().max().item(), (h - h_ref).abs().max().item()),
            })
    chunk_speed = []
    for rank in (1, 2):
        args, state = scan_inputs(device, seqlen=512, rank=rank, nheads=6, headdim=64, d_state=64,
                                  batch=8)
        for q in (8, 16, 32, 64, 128, 256):
            t = timeit(lambda: _chunk_scan(*args, *state, chunk_size=q), iters=20, warmup=6)
            chunk_speed.append({"rank": rank, "chunk_size": q, "ms": t})
            print(f"   R={rank} Q={q:3d}  {t:.3f} ms")

    print("-- layer alignment")
    variants = [
        ("SISO", dict()),
        ("SISO + outproj_norm", dict(is_outproj_norm=True)),
        ("SISO + rope_fraction=1", dict(rope_fraction=1.0)),
        ("MIMO(R=2)", dict(is_mimo=True, mimo_rank=2, chunk_size=32)),
        ("MIMO(R=4) + norm", dict(is_mimo=True, mimo_rank=4, chunk_size=16, is_outproj_norm=True)),
        ("MIMO(R=4) unfused", dict(is_mimo=True, mimo_rank=4, chunk_size=16,
                                   is_outproj_norm=True, fuse_pregate_headwise_norm=False)),
    ]
    layer_alignment = []
    for tag, over in variants:
        mixer = Mamba3(d_model=128, d_state=32, headdim=32, layer_idx=0, **over).to(device).float()
        u = torch.randn(3, 40, mixer.d_model, device=device)
        with torch.no_grad():
            y_full = mixer(u)
            p = InferenceParams(max_seqlen=40, max_batch_size=3)
            y_a = mixer(u[:, :17], inference_params=p)
            p.seqlen_offset += 17
            y_b = mixer(u[:, 17:], inference_params=p)
            e_seg = (torch.cat([y_a, y_b], dim=1) - y_full).abs().max().item()
            p = InferenceParams(max_seqlen=40, max_batch_size=3)
            states = mixer._get_states_from_cache(p, 3)
            outs = [mixer.step(u[:, t], *states)[0] for t in range(u.shape[1])]
            e_step = (torch.stack(outs, dim=1) - y_full).abs().max().item()
        layer_alignment.append({"tag": tag, "segmented": e_seg, "stepwise": e_step})
        print(f"   {tag:<24s} seg={e_seg:.2e} step={e_step:.2e}")

    print("-- graph replay")
    model = MixerModel(128, n_layer=3, rms_norm=True,
                       ssm_cfg=dict(d_state=32, headdim=32)).to(device).eval().float()
    frames = [torch.randn(6, 1, 128, device=device) for _ in range(10)]
    p = InferenceParams(max_seqlen=64, max_batch_size=6)
    with torch.inference_mode():
        ref = []
        for x in frames:
            ref.append(model(x, inference_params=p).clone())
            p.seqlen_offset += 1
        cache = update_graph_cache(model, None, 6, 0, 64)
        got = [cache.run(x) for x in frames]
        cg_err = max((a - b).abs().max().item() for a, b in zip(ref, got))
        initialize_states(cache.inference_params)
        cg_reset = (cache.run(frames[0]) - ref[0]).abs().max().item()
    print(f"   graph={cg_err:.2e}  reset={cg_reset:.2e}")

    print("-- error vs sequence length")
    stability_len = []
    for L in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        args, state = scan_inputs(device, seqlen=L, rank=1)
        y_ref, h_ref = _recurrent_scan(*args, *state)
        y, h = _chunk_scan(*args, *state, chunk_size=64)
        stability_len.append({
            "seqlen": L,
            "err_y": (y - y_ref).abs().max().item(),
            "err_h": (h - h_ref).abs().max().item(),
            "scale_y": y_ref.abs().max().item(),
            "scale_h": h_ref.abs().max().item(),
        })
        r = stability_len[-1]
        print(f"   L={L:5d}  Y={r['err_y']:.2e}  H={r['err_h']:.2e}")

    print("-- streaming decode drift")
    drift_steps = 1024
    drift_mixer = Mamba3(d_model=128, d_state=32, headdim=32, layer_idx=0).to(device).float()
    u = torch.randn(2, drift_steps, 128, device=device)
    with torch.no_grad():
        y_full = drift_mixer(u)
        p = InferenceParams(max_seqlen=drift_steps, max_batch_size=2)
        states = drift_mixer._get_states_from_cache(p, 2)
        drift = [(drift_mixer.step(u[:, t], *states)[0] - y_full[:, t]).abs().max().item()
                 for t in range(drift_steps)]
    decode_drift = {"steps": drift_steps, "err": drift, "scale": y_full.abs().max().item()}
    print(f"   {drift_steps} steps: first={drift[0]:.2e}  last={drift[-1]:.2e}  "
          f"max={max(drift):.2e}")
    del u, y_full
    torch.cuda.empty_cache()

    torch.backends.cuda.matmul.allow_tf32 = tf32

    print("-- scan bench")
    scan_bench = []
    seqlens = (32, 64, 128, 256, 512)
    for is_mimo, rank in ((False, 1), (True, 2)):
        tag = f"MIMO(R={rank})" if is_mimo else "SISO"
        chunk = 64 // rank
        for seqlen in seqlens:
            args, state = scan_inputs(device, seqlen=seqlen, rank=rank)
            t_rec = timeit(lambda: _recurrent_scan(*args, *state))
            t_chunk = timeit(lambda: _chunk_scan(*args, *state, chunk_size=chunk))
            scan_bench.append({
                "tag": tag, "seqlen": seqlen,
                "recurrent_ms": t_rec, "chunked_ms": t_chunk, "speedup": t_rec / t_chunk,
            })
            print(f"   {tag:10s} L={seqlen:3d} rec={t_rec:6.2f} chunk={t_chunk:5.2f} "
                  f"{t_rec / t_chunk:5.1f}x")

    print("-- training step")
    train_bench = []
    for is_mimo, rank in ((False, 1), (True, 2)):
        tag = f"MIMO(R={rank})" if is_mimo else "SISO"
        mixer = Mamba3(d_model=384, d_state=64, headdim=64, is_mimo=is_mimo, mimo_rank=rank,
                       chunk_size=64 // rank, layer_idx=0).to(device).float()
        for seqlen in (128, 256, 512, 1024, 2048):
            u = torch.randn(8, seqlen, mixer.d_model, device=device)

            def step():
                mixer.zero_grad(set_to_none=True)
                mixer(u).square().mean().backward()

            t = timeit(step, iters=10, warmup=4)
            per_tok = t * 1e3 / (8 * seqlen)
            train_bench.append({"tag": tag, "seqlen": seqlen, "ms": t, "us_per_token": per_tok})
            print(f"   {tag:10s} L={seqlen:4d} {t:7.2f} ms  {per_tok:.3f} us/token")
            del u
            torch.cuda.empty_cache()

    print("-- decode vs attention with a KV cache")
    d_model, nheads, dec_batch = 384, 6, 4
    mamba = Mamba3(d_model=d_model, d_state=64, headdim=64, layer_idx=0).to(device).eval().float()
    attn = CausalAttention(d_model, nheads).to(device).eval().float()
    n_mamba = sum(p.numel() for p in mamba.parameters())
    n_attn = sum(p.numel() for p in attn.parameters())
    contexts = (256, 512, 1024, 2048, 4096, 8192, 16384)
    decode_scaling = []
    token = torch.randn(dec_batch, 1, d_model, device=device)
    with torch.inference_mode():
        for ctx in contexts:
            # SSM: one fixed-size recurrent state, independent of context length
            p = InferenceParams(max_seqlen=ctx + 2, max_batch_size=dec_batch)
            p.key_value_memory_dict[0] = mamba.allocate_inference_cache(dec_batch, ctx + 2)
            p.seqlen_offset = ctx
            state_mb = sum(s.numel() * s.element_size()
                           for s in p.key_value_memory_dict[0]) / 2**20
            t_m = timeit(lambda: mamba(token, inference_params=p), iters=100, warmup=30)

            # Attention: q for one token against a KV cache of ctx entries
            hd = d_model // nheads
            k_cache = torch.randn(dec_batch, nheads, ctx, hd, device=device)
            v_cache = torch.randn(dec_batch, nheads, ctx, hd, device=device)
            cache_mb = (k_cache.numel() + v_cache.numel()) * k_cache.element_size() / 2**20

            def attn_step():
                qkv = attn.qkv(token).reshape(dec_batch, 1, 3, nheads, hd)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                ks = torch.cat([k_cache, k], dim=2)
                vs = torch.cat([v_cache, v], dim=2)
                o = F.scaled_dot_product_attention(q, ks, vs)
                return attn.proj(o.transpose(1, 2).reshape(dec_batch, 1, d_model))

            t_a = timeit(attn_step, iters=100, warmup=30)
            decode_scaling.append({
                "context": ctx,
                "mamba_ms": t_m, "attn_ms": t_a,
                "mamba_state_mb": state_mb, "attn_cache_mb": cache_mb,
            })
            print(f"   ctx={ctx:6d}  ssm={t_m:6.3f}ms/{state_mb:8.3f}MB   "
                  f"attn={t_a:6.3f}ms/{cache_mb:8.2f}MB")
            del k_cache, v_cache
            torch.cuda.empty_cache()

    print("-- decode bench")
    decode_bench = []
    dec_bs, dec_len = 64, 64
    for d_state in (16, 32, 64):
        m = MixerModel(320, n_layer=3, rms_norm=True,
                       ssm_cfg=dict(d_state=d_state, headdim=64)).to(device).eval().float()
        x = torch.randn(dec_bs, 1, 320, device=device)
        with torch.inference_mode():
            p = InferenceParams(max_seqlen=dec_len, max_batch_size=dec_bs)
            p.key_value_memory_dict = m.allocate_inference_cache(dec_bs, dec_len)
            p.seqlen_offset = 1
            t_eager = timeit(lambda: m(x, inference_params=p), iters=100, warmup=30)
            cache = update_graph_cache(m, None, dec_bs, 0, dec_len)
            t_graph = timeit(lambda: cache.run(x), iters=100, warmup=30)
        decode_bench.append({
            "d_state": d_state, "eager_ms": t_eager, "graph_ms": t_graph,
            "speedup": t_eager / t_graph, "pct_of_60fps": t_graph / 16.7 * 100,
        })
        print(f"   N={d_state:2d} eager={t_eager:.3f} graph={t_graph:.3f} "
              f"{t_eager / t_graph:.1f}x")
    decode_bench_params = {"batch": dec_bs, "n_layer": 3, "d_model": 320, "headdim": 64}

    print("-- compile bench")
    # The four configurations of the decode path, since compilation and replay
    # are independent: replay removes launch overhead, compilation removes the
    # launches. reduce-overhead is measured too because it brings its own CUDA
    # graphs, which would be a second layer under our capture -- except that the
    # in-place state updates make inductor skip them, so it buys nothing here.
    compile_bench = []
    import torch._dynamo as dynamo
    for batch in (1, 64):
        row = {"batch": batch}
        for tag, mode, graph in (("eager", None, False),
                                 ("compile", "default", False),
                                 ("compile_ro", "reduce-overhead", False),
                                 ("graph", None, True),
                                 ("compile_graph", "default", True),
                                 ("compile_ro_graph", "reduce-overhead", True)):
            dynamo.reset()
            torch.accelerator.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m = MixerModel(320, n_layer=3, rms_norm=True,
                           ssm_cfg=dict(d_state=32, headdim=64)).to(device).eval().float()
            x = torch.randn(batch, 1, 320, device=device)
            try:
                with torch.inference_mode():
                    fn = m if mode is None else torch.compile(m, mode=mode, fullgraph=True)
                    p = InferenceParams(max_seqlen=dec_len, max_batch_size=batch)
                    p.key_value_memory_dict = m.allocate_inference_cache(batch, dec_len)
                    p.seqlen_offset = 1
                    for _ in range(5):
                        fn(x, inference_params=p)
                    if graph:
                        cache = update_graph_cache(fn, None, batch, 0, dec_len)
                        call = lambda: cache.run(x)
                    else:
                        call = lambda: fn(x, inference_params=p)
                    row[tag] = timeit(call, iters=100, warmup=30)
                row[tag + "_mib"] = torch.cuda.max_memory_allocated() / 2**20
            except Exception as exc:
                row[tag] = row[tag + "_mib"] = None
                print(f"   B={batch} {tag} failed: {type(exc).__name__}")
        compile_bench.append(row)
        print(f"   B={batch} " + "  ".join(
            f"{k}={row[k]:.3f}/{row[k + '_mib']:.0f}MiB"
            for k in row if not k.endswith("_mib") and k != "batch" and row[k] is not None))
    dynamo.reset()

    print("-- trainable length: chunked scan vs the naive recurrence")
    # The recurrence is what a readable reference implementation does. Backprop through
    # L Python steps is what makes long-sequence training impractical, in time and in
    # the size of the autograd graph.
    long_context = []
    for L in (128, 256, 512, 1024, 2048, 4096):
        args, state = scan_inputs(device, seqlen=L, rank=1, batch=4, nheads=6, headdim=64,
                                  d_state=64, requires_grad=True)

        def step(fn, **kw):
            def run():
                for t in args:
                    t.grad = None
                y, h = fn(*args, *state, **kw)
                (y.square().mean() + h.square().mean()).backward()
            return run

        row = {"seqlen": L}
        for tag, fn, kw in (("chunked", _chunk_scan, dict(chunk_size=64)),
                            ("recurrent", _recurrent_scan, {})):
            try:
                row[f"{tag}_ms"] = timeit(step(fn, **kw), iters=5, warmup=2)
                row[f"{tag}_mb"] = peak_mem_mb(step(fn, **kw))
            except torch.OutOfMemoryError:
                row[f"{tag}_ms"] = row[f"{tag}_mb"] = None
                torch.cuda.empty_cache()
                print(f"   L={L:5d}  {tag}: out of memory")
        long_context.append(row)
        print(f"   L={L:5d}  chunked={row['chunked_ms']:7.2f}ms/{row['chunked_mb']:8.1f}MB   "
              f"recurrent={row['recurrent_ms']:8.2f}ms/{row['recurrent_mb']:9.1f}MB")
        del args, state
        torch.cuda.empty_cache()
    long_context_params = {"batch": 4, "nheads": 6, "headdim": 64, "d_state": 64,
                           "chunk_size": 64}

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "official_package": "mamba_ssm not installed — official fused kernels not compared",
        "scan_alignment": scan_alignment,
        "chunk_sweep": chunk_sweep,
        "chunk_speed": chunk_speed,
        "layer_alignment": layer_alignment,
        "graph_replay_max_abs_diff": cg_err,
        "graph_replay_reset_max_abs_diff": cg_reset,
        "compile_bench": compile_bench,
        "scan_bench": scan_bench,
        "train_bench": train_bench,
        "decode_scaling": decode_scaling,
        "decode_scaling_params": {"mamba3": n_mamba, "attention": n_attn,
                                  "d_model": d_model, "nheads": nheads, "batch": dec_batch},
        "decode_bench": decode_bench,
        "decode_bench_params": decode_bench_params,
        "stability_len": stability_len,
        "decode_drift": decode_drift,
        "long_context": long_context,
        "long_context_params": long_context_params,
    }
    (ASSETS / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_scan(D) -> None:
    rows = D["scan_bench"]
    seqlens = [32, 64, 128, 256, 512]
    fig, (ax_t, ax_s) = _stacked(height=7.1, share_x=True)

    for tag, color in (("SISO", TEAL), ("MIMO(R=2)", CORAL)):
        sel = [r for r in rows if r["tag"] == tag]
        xs = [r["seqlen"] for r in sel]
        ax_t.plot(xs, [r["recurrent_ms"] for r in sel], color=color, ls=(0, (3.2, 2.2)), lw=1.9,
                  marker="o", ms=6.5, mfc="white", mew=1.8, label=f"{tag}  recurrent")
        ax_t.plot(xs, [r["chunked_ms"] for r in sel], color=color,
                  marker="s", ms=5.8, label=f"{tag}  chunked")
        ax_s.plot(xs, [r["speedup"] for r in sel], color=color,
                  marker="o", ms=7.0, mfc="white", mew=1.9, label=tag)

    for ax in (ax_t, ax_s):
        _style(ax)
        ax.set_xticks(seqlens)
        ax.set_xticklabels(["32", "", "128", "256", "512"])
        ax.set_xlim(16, 568)
    ax_s.set_xlabel("Sequence length")
    ax_t.set_title("Scan time", loc="left")
    ax_s.set_title("Speedup over the recurrence", loc="left")
    ax_t.set_ylabel("Milliseconds")
    ax_s.set_ylabel("× faster")
    ax_t.set_ylim(0, 80)
    ax_s.set_ylim(0, 78)
    ax_t.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.8)
    ax_s.legend(loc="upper left")
    ax_t.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_s.yaxis.set_major_locator(mticker.MultipleLocator(20))

    tail = [r["speedup"] for r in rows if r["seqlen"] == seqlens[-1]]
    lo, hi = round(min(tail)), round(max(tail))
    ax_s.annotate(f"{lo}×" if lo == hi else f"{lo}–{hi}×",
                  (seqlens[-1], max(tail)), textcoords="offset points", xytext=(-8, 10),
                  ha="right", color=INK, fontsize=12, fontweight="bold")

    _heading(fig, "Chunked GEMM vs step-by-step recurrence",
             "Recurrent cost grows linearly with L; the chunked path stays near 1 ms")
    _footer(fig, f"{D['gpu']}  ·  fp32, B=3, H=4, P=16, N=32",
            "SISO chunk=64, MIMO(R=2) chunk=32")
    _save(fig, "scan.png")


def plot_decode_scaling(D) -> None:
    rows = D["decode_scaling"]
    meta = D["decode_scaling_params"]
    xs = [r["context"] for r in rows]
    fig, (ax_t, ax_m) = _stacked(height=7.1, share_x=True)

    ax_t.plot(xs, [r["attn_ms"] for r in rows], color=PLUM, marker="o", ms=6.5,
              mfc="white", mew=1.9, label="Attention + KV cache")
    ax_t.plot(xs, [r["mamba_ms"] for r in rows], color=TEAL, marker="s", ms=5.8,
              label="Mamba-3 recurrent state")
    ax_m.plot(xs, [r["attn_cache_mb"] for r in rows], color=PLUM, marker="o", ms=6.5,
              mfc="white", mew=1.9, label="KV cache")
    ax_m.plot(xs, [r["mamba_state_mb"] for r in rows], color=TEAL, marker="s", ms=5.8,
              label="SSM state")

    last = rows[-1]
    ax_m.annotate(f"{last['attn_cache_mb'] / last['mamba_state_mb']:.0f}× smaller at 16k",
                  (xs[-1], last["mamba_state_mb"]), textcoords="offset points", xytext=(-8, 14),
                  ha="right", color=TEAL, fontsize=11.5, fontweight="bold")

    for ax in (ax_t, ax_m):
        _style(ax, log=True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v // 1024}k" if v >= 1024 else str(v) for v in xs])
        ax.legend(loc="upper left")
    ax_m.set_xlabel("Context already consumed  (tokens)")
    ax_t.set_ylabel("Decode latency  (ms)")
    ax_m.set_ylabel("Per layer  (MiB, log)")
    ax_m.set_yscale("log")
    ax_t.set_ylim(0, max(r["attn_ms"] for r in rows) * 1.12)
    ax_t.set_title("One more token, whatever the context", loc="left")
    ax_m.set_title("What has to be kept around", loc="left")

    _heading(fig, "Constant state vs a growing KV cache",
             "Decode cost and memory do not depend on how much context came before")
    _footer(fig, f"{D['gpu']}  ·  fp32, one layer, B={meta['batch']}, "
                 f"d_model={meta['d_model']}, {meta['nheads']} heads",
            "Baseline: PyTorch SDPA over a preallocated cache. Eager path — graph replay is faster.")
    _save(fig, "decode_scaling.png")


def plot_chunk(D) -> None:
    sweep, speed = D["chunk_sweep"], D["chunk_speed"]
    fig, (ax_e, ax_t) = _stacked(height=7.3)

    for rank, color in ((1, TEAL), (2, CORAL), (4, NAVY)):
        sel = [r for r in sweep if r["rank"] == rank]
        ax_e.plot([r["chunk_size"] for r in sel], [r["err"] for r in sel], color=color, lw=2.0,
                  marker="o", ms=6.0, mfc="white", mew=1.7, label=f"R={rank}")
    ax_e.axhline(1e-4, color=SLATE, lw=1.0, ls=(0, (3, 2)))
    ax_e.text(1.05, 1.2e-4, "test tolerance  1×10⁻⁴", color=MUTED, fontsize=10.5, va="bottom")

    for i, (rank, color) in enumerate(((1, TEAL), (2, CORAL))):
        sel = [r for r in speed if r["rank"] == rank]
        ax_t.plot([r["chunk_size"] for r in sel], [r["ms"] for r in sel], color=color,
                  marker="s", ms=6.0, label=f"R={rank}")
        best = min(sel, key=lambda r: r["ms"])
        ax_t.text(0.35, 0.92 - 0.11 * i, f"R={rank} fastest at Q={best['chunk_size']}",
                  transform=ax_t.transAxes, color=color, fontsize=11.5, fontweight="bold")

    for ax in (ax_e, ax_t):
        _style(ax, log=True)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Chunk size Q")
    ax_e.legend(loc="center left", ncol=3, columnspacing=1.2, handlelength=1.8)
    ax_t.legend(loc="lower left")
    ax_e.set_yscale("log")
    ax_e.set_xticks([1, 4, 16, 64, 256])
    ax_e.set_xticklabels(["1", "4", "16", "64", "256"])
    ax_e.set_ylim(2e-6, 3e-4)
    ax_e.set_ylabel("max |Δ|  vs recurrence")
    ax_e.set_title("Every chunk size agrees", loc="left")
    ax_t.set_xticks([8, 16, 32, 64, 128, 256])
    ax_t.set_xticklabels(["8", "16", "32", "64", "128", "256"])
    ax_t.set_ylabel("Milliseconds")
    ax_t.set_title("Only the speed moves", loc="left")

    _heading(fig, "Chunk size is a performance knob, not a correctness one",
             "Q partitions the same algebra, so the result holds while the GEMM shape changes",
             "— and the fastest Q sits at or just above the default 64 / rank")
    _footer(fig, f"{D['gpu']}  ·  fp32, TF32 off  ·  error: B=3, L=256, H=4, P=16, N=32",
            "Timing: B=8, L=512, H=6, P=64, N=64")
    _save(fig, "chunk_size.png")


def plot_training(D) -> None:
    rows = D["train_bench"]
    fig, (ax_t, ax_p) = _stacked(height=7.1, share_x=True)

    for tag, color in (("SISO", TEAL), ("MIMO(R=2)", CORAL)):
        sel = [r for r in rows if r["tag"] == tag]
        xs = [r["seqlen"] for r in sel]
        ax_t.plot(xs, [r["ms"] for r in sel], color=color, marker="o", ms=6.5,
                  mfc="white", mew=1.9, label=tag)
        ax_p.plot(xs, [r["us_per_token"] for r in sel], color=color, marker="s", ms=5.8,
                  label=tag)

    for ax in (ax_t, ax_p):
        _style(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 256, 512, 1024, 2048])
        ax.set_xticklabels(["128", "256", "512", "1k", "2k"])
    ax_t.legend(loc="upper left")
    ax_p.legend(loc="upper center", ncol=2, columnspacing=1.4)
    ax_p.set_xlabel("Sequence length")
    ax_t.set_ylabel("Fwd + bwd  (ms)")
    ax_p.set_ylabel("Microseconds / token")
    ax_t.set_ylim(bottom=0)
    ax_p.set_ylim(bottom=0)
    ax_t.set_title("Training step", loc="left")
    ax_p.set_title("Cost per token", loc="left")

    _heading(fig, "Autograd straight through the chunked scan",
             "Per-token cost bottoms out near L≈512–1024: launch overhead is amortized,",
             "chunk intermediates still fit comfortably")
    _footer(fig, f"{D['gpu']}  ·  fp32, B=8, d_model=384, N=64, P=64",
            "One zero_grad + forward + backward per step, no custom kernel")
    _save(fig, "training.png")


def plot_alignment(D) -> None:
    rows = D["layer_alignment"]
    scan = D["scan_alignment"]
    labels = ["SISO", "SISO\n+ norm", "SISO\nrope = 1",
              "MIMO\nR=2", "MIMO R=4\n+ norm", "MIMO R=4\nunfused"]

    fig, (ax_l, ax_s) = _stacked(height=7.6, ratios=(1.15, 1.0))

    x = np.arange(len(rows))
    w = 0.38
    ax_l.bar(x - w / 2, [r["segmented"] for r in rows], w, color=NAVY,
             label="Segmented resume", zorder=2)
    ax_l.bar(x + w / 2, [r["stepwise"] for r in rows], w, color=TEAL,
             label="Official step()", zorder=2)
    _style(ax_l, log=True)
    ax_l.set_yscale("log")
    ax_l.set_ylim(8e-8, 3.2e-6)
    ax_l.axhline(1e-6, color=SLATE, lw=1.0, ls=(0, (3, 2)))
    ax_l.text(len(rows) - 0.42, 1.15e-6, "1×10⁻⁶", color=MUTED, fontsize=10.5,
              ha="right", va="bottom")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=10.0)
    ax_l.set_ylabel("max |Δ|  vs full forward")
    ax_l.set_title("Layer configs — resumed state vs one full forward", loc="left")
    ax_l.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.5)

    xr = np.arange(len(scan))
    ax_s.bar(xr - w / 2, [r["fwd_y"] for r in scan], w, color=CORAL, label="Forward", zorder=2)
    ax_s.bar(xr + w / 2, [r["grad_rel"] for r in scan], w, color=PLUM,
             label="Gradient (relative)", zorder=2)
    _style(ax_s, log=True)
    ax_s.set_yscale("log")
    ax_s.set_ylim(1e-8, 8e-5)
    ax_s.set_xticks(xr)
    ax_s.set_xticklabels([f"R={r['rank']}" for r in scan])
    ax_s.set_ylabel("max |Δ|  vs recurrence")
    ax_s.set_title("Chunked scan — forward and backward", loc="left")
    ax_s.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.5)

    cg = D["graph_replay_max_abs_diff"]
    _heading(fig, "Numerical agreement",
             f"Every alternative path reproduces the reference to ~10⁻⁶ or better.",
             f"Graph replay vs eager decode: {cg:.1e}")
    _footer(fig, f"{D['gpu']}  ·  fp32, TF32 disabled  ·  B=3, L=40",
            "Top: d_model=128, N=32.  Bottom: H=4, P=16, N=32, chunk=8")
    _save(fig, "alignment.png")


def plot_decode(D) -> None:
    rows = D["decode_bench"]
    fig, ax = _single(height=4.4)
    x = np.arange(len(rows))
    w = 0.38
    eager = [r["eager_ms"] for r in rows]
    graph = [r["graph_ms"] for r in rows]
    b1 = ax.bar(x - w / 2, eager, w, color=SLATE, label="Eager step", zorder=2)
    b2 = ax.bar(x + w / 2, graph, w, color=BLUE, label="Graph replay", zorder=2)

    for bars, vals in ((b1, eager), (b2, graph)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.07, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    for i, r in enumerate(rows):
        ax.text(x[i] + w / 2, r["graph_ms"] + 0.42, f"{r['speedup']:.1f}×",
                ha="center", va="bottom", fontsize=11, color=BLUE, fontweight="bold")

    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d_state = {r['d_state']}" for r in rows])
    ax.set_ylabel("Per-frame latency  (ms)")
    ax.set_ylim(0, max(eager) * 1.26)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.5)

    lo = min(r["speedup"] for r in rows)
    hi = max(r["speedup"] for r in rows)
    p_lo = min(r["pct_of_60fps"] for r in rows)
    p_hi = max(r["pct_of_60fps"] for r in rows)
    _heading(fig, "Single-step decode",
             f"Graph replay is {lo:.1f}–{hi:.1f}× faster than eager, "
             f"{p_lo:.1f}–{p_hi:.1f}% of a 16.7 ms frame", axes_title=False)
    m = D.get("decode_bench_params", {"batch": 64, "n_layer": 3, "d_model": 320})
    _footer(fig, f"{D['gpu']}  ·  fp32, {m['n_layer']} layers, batch {m['batch']}, "
                 f"d_model={m['d_model']}",
            "100 timed steps after 30 warmup")
    _save(fig, "decode.png")


def plot_stability(D) -> None:
    rows, drift = D["stability_len"], D["decode_drift"]
    fig, (ax_l, ax_d) = _stacked(height=7.3)

    xs = [r["seqlen"] for r in rows]
    ax_l.plot(xs, [r["err_y"] for r in rows], color=TEAL, marker="s", ms=6.0, label="Output Y")
    ax_l.plot(xs, [r["err_h"] for r in rows], color=CORAL, lw=2.0, marker="o", ms=6.5,
              mfc="white", mew=1.8, label="Final state h")
    ax_l.axhline(1e-6, color=SLATE, lw=1.0, ls=(0, (3, 2)))
    ax_l.text(xs[0], 1.2e-6, "1×10⁻⁶", color=MUTED, fontsize=10.5, va="bottom")
    _style(ax_l, log=True)
    ax_l.set_xscale("log", base=2)
    ax_l.set_yscale("log")
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels([f"{v // 1024}k" if v >= 1024 else str(v) for v in xs])
    ax_l.set_ylim(1e-7, 3e-5)
    ax_l.set_xlabel("Sequence length")
    ax_l.set_ylabel("max |Δ|  vs recurrence")
    ax_l.set_title("Chunked scan, L = 32 … 4k", loc="left")
    ax_l.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.8)

    err = np.maximum(np.asarray(drift["err"], dtype=float), 1e-12)
    steps = np.arange(1, len(err) + 1)
    running = np.maximum.accumulate(err)
    ax_d.plot(steps, err, color="#C3C8D1", lw=0.8, label="per step")
    ax_d.plot(steps, running, color=BLUE, label="running max")
    ax_d.annotate(f"{running[-1]:.1e} after {len(err)} steps",
                  (steps[-1], running[-1]), textcoords="offset points", xytext=(-8, -20),
                  ha="right", va="top", color=BLUE, fontsize=11.5, fontweight="bold")
    _style(ax_d, log=True)
    ax_d.set_yscale("log")
    ax_d.set_ylim(2e-7, 6e-6)
    ax_d.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_d.set_xlim(0, len(err))
    ax_d.set_xlabel("Decode step")
    ax_d.set_ylabel("max |Δ|  vs full forward")
    ax_d.set_title("Streaming step(), one state carried forward", loc="left")
    ax_d.legend(loc="lower right", ncol=2, columnspacing=1.2, handlelength=1.8)

    _heading(fig, "Error does not grow with length or with time",
             "Chunking splits the same algebra, so a longer sequence adds no drift —",
             "and neither does replaying a streaming state a thousand steps in a row")
    _footer(fig, f"{D['gpu']}  ·  fp32, TF32 off  ·  top: B=3, H=4, P=16, N=32, chunk=64",
            f"Bottom: B=2, d_model=128, N=32, P=32, {drift['steps']} consecutive steps")
    _save(fig, "stability.png")


def plot_long_context(D) -> None:
    rows, meta = D["long_context"], D["long_context_params"]
    xs = [r["seqlen"] for r in rows]
    fig, (ax_t, ax_m) = _stacked(height=7.1, share_x=True)

    def series(key):
        return ([r["seqlen"] for r in rows if r[key] is not None],
                [r[key] for r in rows if r[key] is not None])

    for ax, kc, kr, ylabel, title in (
        (ax_t, "chunked_ms", "recurrent_ms", "Fwd + bwd  (ms, log)", "Training step"),
        (ax_m, "chunked_mb", "recurrent_mb", "Peak allocated  (MiB)", "Autograd footprint"),
    ):
        ax.plot(*series(kr), color=CORAL, lw=2.0, marker="o", ms=6.5, mfc="white", mew=1.8,
                ls=(0, (3.2, 2.2)), label="Step-by-step recurrence")
        ax.plot(*series(kc), color=TEAL, marker="s", ms=6.0, label="Chunked scan")
        _style(ax, log=ax is ax_t)
        ax.set_xscale("log", base=2)
        if ax is ax_t:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 3900)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v // 1024}k" if v >= 1024 else str(v) for v in xs])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left")
    ax_m.set_xlabel("Sequence length")

    last = rows[-1]
    for ax, kc, kr, unit in ((ax_t, "chunked_ms", "recurrent_ms", "faster"),
                             (ax_m, "chunked_mb", "recurrent_mb", "smaller")):
        if last[kc] and last[kr]:
            ax.annotate(f"{last[kr] / last[kc]:.0f}× {unit} at {last['seqlen'] // 1024}k",
                        (last["seqlen"], last[kc]), textcoords="offset points",
                        xytext=(-8, 12), ha="right", color=TEAL, fontsize=11.5,
                        fontweight="bold")

    _heading(fig, "What makes long sequences trainable at all",
             "Backprop through L Python steps is the wall a readable reference hits —",
             "in wall-clock time and in the size of the autograd graph")
    _footer(fig, f"{D['gpu']}  ·  fp32, B={meta['batch']}, H={meta['nheads']}, "
                 f"P={meta['headdim']}, N={meta['d_state']}, chunk={meta['chunk_size']}",
            "One zero_grad + forward + backward through the scan itself")
    _save(fig, "long_context.png")


def plot_compile(D) -> None:
    rows = D.get("compile_bench")
    if not rows:
        return
    tags = [("eager", "Eager"), ("compile", "torch.compile"),
            ("graph", "Graph replay"), ("compile_graph", "Both")]
    fig, ax = _single(height=4.3)
    y = np.arange(len(tags))
    h = 0.36

    # A linear axis, not log: the point of the figure is that the last bar is a
    # sliver, and a log axis would flatten exactly that.
    ax.axhspan(y[-1] - 0.5, y[-1] + 0.5, color="#F3F6FD", zorder=0)
    span = max(r["eager"] for r in rows)
    for i, (r, color) in enumerate(zip(rows, (SLATE, BLUE))):
        vals = [r[t] for t, _ in tags]
        bars = ax.barh(y + (i - 0.5) * h, vals, h, color=color, zorder=2,
                       label=f"batch {r['batch']}")
        for rect, v, (tag, _) in zip(bars, vals, tags):
            gain = "" if tag == "eager" else f"   {r['eager'] / v:.0f}×"
            ax.text(v + span * 0.014, rect.get_y() + rect.get_height() / 2,
                    f"{v:.3f} ms{gain}", va="center", ha="left", fontsize=10.5,
                    color=INK, fontweight="bold" if tag == "compile_graph" else "normal")

    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in tags])
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.46)
    ax.set_xlabel("Per-token latency  (ms)")
    # Right of the middle rows: the only block of space the labels leave free,
    # and it keeps the legend off the highlighted bottom row.
    ax.legend(loc="center right", ncol=1, handlelength=1.5, labelspacing=0.5)

    best = max(r["eager"] / r["compile_graph"] for r in rows)
    worst = min(r["eager"] / r["compile_graph"] for r in rows)
    _heading(fig, "Compilation and replay compose",
             f"Each removes a different cost; together {worst:.0f}–{best:.0f}× "
             f"faster than eager", axes_title=False)
    _footer(fig, f"{D['gpu']}  ·  fp32, 3 layers, d_model=320, d_state=32",
            "100 timed steps after 30 warmup")
    _save(fig, "compile.png")


def plot_all(D) -> None:
    plot_scan(D)
    plot_decode_scaling(D)
    plot_chunk(D)
    plot_training(D)
    plot_alignment(D)
    plot_decode(D)
    plot_compile(D)
    plot_stability(D)
    plot_long_context(D)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="redraw from assets/results.json")
    args = ap.parse_args()

    if args.replot:
        D = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
    else:
        assert torch.cuda.is_available(), "measuring needs a CUDA device; use --replot"
        torch.backends.cudnn.benchmark = True
        D = measure(torch.device("cuda"))
    plot_all(D)
    print("done")


if __name__ == "__main__":
    main()
