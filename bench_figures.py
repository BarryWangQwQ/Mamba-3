"""Measure and plot the benchmark figures shown in README.md.

    python bench_figures.py            # measure on GPU, then plot
    python bench_figures.py --replot   # redraw from assets/results.json only

Writes assets/*.png and assets/results.json.
"""

from __future__ import annotations

import argparse
import json
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

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": HAIR,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.24,
})


def _style(ax, *, log=False):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, which="major")
    if log:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, which="minor")
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", pad=4)


def _heading(fig, title: str, subtitle: str) -> None:
    """Title and subtitle above everything, in figure coordinates.

    Kept off the axes so they can never collide with per-axes titles.
    """
    fig.text(0.0, 1.055, title, fontsize=14, fontweight="semibold", color=INK, va="bottom")
    fig.text(0.0, 1.005, subtitle, fontsize=9.0, color=MUTED, va="bottom")


def _footer(fig, text: str) -> None:
    fig.text(0.0, -0.035, text, fontsize=8.3, color=MUTED, va="top")


def _save(fig, name: str) -> None:
    fig.savefig(ASSETS / name)
    plt.close(fig)
    print(f"wrote {name}")


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------
def timeit(fn, iters=20, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


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

    print("-- cuda graph")
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
    for d_state in (16, 32, 64):
        m = MixerModel(320, n_layer=3, rms_norm=True,
                       ssm_cfg=dict(d_state=d_state, headdim=64)).to(device).eval().float()
        x = torch.randn(67, 1, 320, device=device)
        with torch.inference_mode():
            p = InferenceParams(max_seqlen=64, max_batch_size=67)
            p.key_value_memory_dict = m.allocate_inference_cache(67, 64)
            p.seqlen_offset = 1
            t_eager = timeit(lambda: m(x, inference_params=p), iters=100, warmup=30)
            cache = update_graph_cache(m, None, 67, 0, 64)
            t_graph = timeit(lambda: cache.run(x), iters=100, warmup=30)
        decode_bench.append({
            "d_state": d_state, "eager_ms": t_eager, "graph_ms": t_graph,
            "speedup": t_eager / t_graph, "pct_of_60fps": t_graph / 16.7 * 100,
        })
        print(f"   N={d_state:2d} eager={t_eager:.3f} graph={t_graph:.3f} "
              f"{t_eager / t_graph:.1f}x")

    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "official_package": "mamba_ssm not installed — official fused kernels not compared",
        "scan_alignment": scan_alignment,
        "chunk_sweep": chunk_sweep,
        "chunk_speed": chunk_speed,
        "layer_alignment": layer_alignment,
        "cuda_graph_max_abs_diff": cg_err,
        "cuda_graph_reset_max_abs_diff": cg_reset,
        "scan_bench": scan_bench,
        "train_bench": train_bench,
        "decode_scaling": decode_scaling,
        "decode_scaling_params": {"mamba3": n_mamba, "attention": n_attn,
                                  "d_model": d_model, "nheads": nheads, "batch": dec_batch},
        "decode_bench": decode_bench,
    }
    (ASSETS / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_scan(D) -> None:
    rows = D["scan_bench"]
    seqlens = [32, 64, 128, 256, 512]
    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    fig.subplots_adjust(wspace=0.26)

    for tag, color in (("SISO", TEAL), ("MIMO(R=2)", CORAL)):
        sel = [r for r in rows if r["tag"] == tag]
        xs = [r["seqlen"] for r in sel]
        ax_t.plot(xs, [r["recurrent_ms"] for r in sel], color=color, ls=(0, (3.2, 2.2)), lw=1.7,
                  marker="o", ms=5.5, mfc="white", mew=1.6, label=f"{tag}  recurrent")
        ax_t.plot(xs, [r["chunked_ms"] for r in sel], color=color, lw=2.0,
                  marker="s", ms=5.0, label=f"{tag}  chunked")
        spd = [r["speedup"] for r in sel]
        ax_s.plot(xs, spd, color=color, lw=2.0, marker="o", ms=6.0, mfc="white", mew=1.7, label=tag)

    for ax in (ax_t, ax_s):
        _style(ax)
        ax.set_xticks(seqlens)
        ax.set_xlim(20, 564)
        ax.set_xlabel("Sequence length")
    ax_t.set_title("Scan time", loc="left")
    ax_s.set_title("Speedup", loc="left")
    ax_t.set_ylabel("Milliseconds")
    ax_s.set_ylabel("× faster than recurrence")
    ax_t.set_ylim(0, 76)
    ax_s.set_ylim(0, 84)
    ax_t.legend(loc="upper left", fontsize=8.6)
    ax_s.legend(loc="upper left", fontsize=8.6)
    ax_t.yaxis.set_major_locator(mticker.MultipleLocator(15))
    ax_s.yaxis.set_major_locator(mticker.MultipleLocator(20))

    tail = [r["speedup"] for r in rows if r["seqlen"] == seqlens[-1]]
    lo, hi = round(min(tail)), round(max(tail))
    label = f"{lo}× at L={seqlens[-1]}" if lo == hi else f"{lo}–{hi}× at L={seqlens[-1]}"
    ax_s.annotate(label,
                  (seqlens[-1], max(tail)), textcoords="offset points", xytext=(-10, 12),
                  ha="right", color=INK, fontsize=9.5, fontweight="semibold")

    _heading(fig, "Selective scan — chunked GEMM vs step-by-step recurrence",
             "Recurrent cost grows linearly with L; the chunked path stays near 1 ms")
    _footer(fig, f"{D['gpu']}  ·  B=3, H=4, P=16, N=32  ·  SISO chunk=64, MIMO(R=2) chunk=32")
    _save(fig, "scan.png")


def plot_decode_scaling(D) -> None:
    rows = D["decode_scaling"]
    meta = D["decode_scaling_params"]
    xs = [r["context"] for r in rows]
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    fig.subplots_adjust(wspace=0.26)

    ax_t.plot(xs, [r["attn_ms"] for r in rows], color=PLUM, lw=2.0, marker="o", ms=5.5,
              mfc="white", mew=1.7, label="Attention + KV cache")
    ax_t.plot(xs, [r["mamba_ms"] for r in rows], color=TEAL, lw=2.0, marker="s", ms=5.0,
              label="Mamba3 recurrent state")
    ax_m.plot(xs, [r["attn_cache_mb"] for r in rows], color=PLUM, lw=2.0, marker="o", ms=5.5,
              mfc="white", mew=1.7, label="KV cache")
    ax_m.plot(xs, [r["mamba_state_mb"] for r in rows], color=TEAL, lw=2.0, marker="s", ms=5.0,
              label="SSM state")

    last = rows[-1]
    ax_m.annotate(f"{last['attn_cache_mb'] / last['mamba_state_mb']:.0f}× smaller\nat 16k context",
                  (xs[-1], last["mamba_state_mb"]), textcoords="offset points", xytext=(-6, 16),
                  ha="right", color=TEAL, fontsize=9.5, fontweight="semibold")

    for ax in (ax_t, ax_m):
        _style(ax, log=True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v // 1024}k" if v >= 1024 else str(v) for v in xs])
        ax.set_xlabel("Context already consumed  (tokens)")
        ax.legend(loc="upper left", fontsize=8.6)
    ax_t.set_ylabel("Per-token decode latency  (ms)")
    ax_m.set_ylabel("Cache / state per layer  (MiB, log)")
    ax_m.set_yscale("log")
    ax_t.set_ylim(bottom=0)
    ax_t.set_title("Decode latency", loc="left")
    ax_m.set_title("What has to be kept", loc="left")

    _heading(fig, "Constant state vs a growing KV cache — one layer, one token",
             "Decoding cost and memory do not depend on how much context was already consumed")
    _footer(fig, f"{D['gpu']}  ·  fp32, B={meta['batch']}, d_model={meta['d_model']}, "
                 f"{meta['nheads']} heads  ·  attention baseline is PyTorch SDPA on a "
                 f"preallocated cache  ·  for batched prefill, fused attention kernels stay "
                 f"competitive; the SSM advantage is here")
    _save(fig, "decode_scaling.png")


def plot_chunk(D) -> None:
    sweep, speed = D["chunk_sweep"], D["chunk_speed"]
    fig, (ax_e, ax_t) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    fig.subplots_adjust(wspace=0.26)

    for rank, color in ((1, TEAL), (2, CORAL), (4, NAVY)):
        sel = [r for r in sweep if r["rank"] == rank]
        ax_e.plot([r["chunk_size"] for r in sel], [r["err"] for r in sel], color=color, lw=1.9,
                  marker="o", ms=5.0, mfc="white", mew=1.5, label=f"R={rank}")
    ax_e.axhline(1e-4, color=HAIR, lw=0.9, ls=(0, (3, 2)))
    ax_e.text(1.05, 1.15e-4, "test tolerance  1×10⁻⁴", color=MUTED, fontsize=8.4, va="bottom")

    for i, (rank, color) in enumerate(((1, TEAL), (2, CORAL))):
        sel = [r for r in speed if r["rank"] == rank]
        ax_t.plot([r["chunk_size"] for r in sel], [r["ms"] for r in sel], color=color, lw=2.0,
                  marker="s", ms=5.0, label=f"R={rank}")
        best = min(sel, key=lambda r: r["ms"])
        ax_t.text(0.40, 0.93 - 0.085 * i, f"R={rank}  fastest at Q={best['chunk_size']}",
                  transform=ax_t.transAxes, color=color, fontsize=9.2, fontweight="semibold")

    for ax in (ax_e, ax_t):
        _style(ax, log=True)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Chunk size Q")
    ax_e.legend(loc="lower right", fontsize=8.6)
    ax_t.legend(loc="lower left", fontsize=8.6)
    ax_e.set_yscale("log")
    ax_e.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
    ax_e.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
    ax_e.set_ylim(1e-6, 4e-4)
    ax_e.set_ylabel("max |Δ|  vs recurrence")
    ax_e.set_title("Every chunk size agrees with the recurrence", loc="left")
    ax_t.set_xticks([8, 16, 32, 64, 128, 256])
    ax_t.set_xticklabels(["8", "16", "32", "64", "128", "256"])
    ax_t.set_ylabel("Milliseconds")
    ax_t.set_title("Chunk size only changes speed", loc="left")

    _heading(fig, "Chunk size is a performance knob, not a correctness one",
             "Q only partitions the same algebra, so the result holds while the GEMM shape "
             "changes — and the measured optimum lands on the default 64 / rank")
    _footer(fig, f"{D['gpu']}  ·  error: B=3, L=256, H=4, P=16, N=32, fp32, TF32 off  ·  "
                 f"timing: B=8, L=512, H=6, P=64, N=64")
    _save(fig, "chunk_size.png")


def plot_training(D) -> None:
    rows = D["train_bench"]
    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(11.6, 4.5))
    fig.subplots_adjust(wspace=0.26)

    for tag, color in (("SISO", TEAL), ("MIMO(R=2)", CORAL)):
        sel = [r for r in rows if r["tag"] == tag]
        xs = [r["seqlen"] for r in sel]
        ax_t.plot(xs, [r["ms"] for r in sel], color=color, lw=2.0, marker="o", ms=5.5,
                  mfc="white", mew=1.7, label=tag)
        ax_p.plot(xs, [r["us_per_token"] for r in sel], color=color, lw=2.0, marker="s", ms=5.0,
                  label=tag)

    for ax in (ax_t, ax_p):
        _style(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 256, 512, 1024, 2048])
        ax.set_xticklabels(["128", "256", "512", "1024", "2048"])
        ax.set_xlabel("Sequence length")
        ax.legend(loc="upper left" if ax is ax_t else "upper right", fontsize=8.6)
    ax_t.set_ylabel("Forward + backward  (ms)")
    ax_p.set_ylabel("Microseconds per token")
    ax_t.set_ylim(bottom=0)
    ax_p.set_ylim(bottom=0)
    ax_t.set_title("Training step", loc="left")
    ax_p.set_title("Cost per token", loc="left")

    _heading(fig, "Training — autograd straight through the chunked scan",
             "Per-token cost bottoms out near L≈512–1024, where launch overhead is amortized "
             "but chunk intermediates still fit comfortably")
    _footer(fig, f"{D['gpu']}  ·  fp32, B=8, d_model=384, N=64, P=64  ·  "
                 f"one zero_grad + forward + backward per step")
    _save(fig, "training.png")


def plot_alignment(D) -> None:
    rows = D["layer_alignment"]
    scan = D["scan_alignment"]
    labels = ["SISO", "SISO\n+ outproj norm", "SISO\n+ rope = 1",
              "MIMO  R=2", "MIMO  R=4\n+ norm", "MIMO  R=4\n+ unfused"]

    fig, (ax_l, ax_s) = plt.subplots(1, 2, figsize=(12.2, 4.5),
                                     gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.subplots_adjust(wspace=0.24)

    x = np.arange(len(rows))
    w = 0.36
    ax_l.bar(x - w / 2, [r["segmented"] for r in rows], w, color=NAVY,
             label="Segmented resume", zorder=2)
    ax_l.bar(x + w / 2, [r["stepwise"] for r in rows], w, color=TEAL,
             label="Official step()", zorder=2)
    _style(ax_l, log=True)
    ax_l.set_yscale("log")
    ax_l.set_ylim(8e-8, 2.4e-6)
    ax_l.axhline(1e-6, color=HAIR, lw=0.9, ls=(0, (3, 2)))
    ax_l.text(len(rows) - 0.45, 1.1e-6, "1×10⁻⁶", color=MUTED, fontsize=8.4, ha="right", va="bottom")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=8.5)
    ax_l.set_ylabel("max |Δ|  vs full-sequence forward")
    ax_l.set_title("Layer configs — state paths vs one full forward", loc="left")
    ax_l.legend(loc="upper left", fontsize=8.6, ncol=2)

    xr = np.arange(len(scan))
    ax_s.bar(xr - w / 2, [r["fwd_y"] for r in scan], w, color=CORAL, label="Forward", zorder=2)
    ax_s.bar(xr + w / 2, [r["grad_rel"] for r in scan], w, color=PLUM,
             label="Gradient (relative)", zorder=2)
    _style(ax_s, log=True)
    ax_s.set_yscale("log")
    ax_s.set_ylim(1e-8, 3e-5)
    ax_s.set_xticks(xr)
    ax_s.set_xticklabels([f"R={r['rank']}" for r in scan])
    ax_s.set_ylabel("max |Δ|  vs recurrence")
    ax_s.set_title("Scan — chunked vs recurrence", loc="left")
    ax_s.legend(loc="upper left", fontsize=8.6)

    cg = D["cuda_graph_max_abs_diff"]
    _heading(fig, "Numerical agreement",
             f"Every alternative path reproduces the reference to ~10⁻⁶ or better  ·  "
             f"CUDA-graph decode vs eager: {cg:.1e}")
    _footer(fig, f"{D['gpu']}  ·  fp32, TF32 disabled  ·  "
                 f"left: B=3, L=40, d_model=128, N=32  ·  right: B=3, L=40, chunk=8")
    _save(fig, "alignment.png")


def plot_decode(D) -> None:
    rows = D["decode_bench"]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    x = np.arange(len(rows))
    w = 0.36
    eager = [r["eager_ms"] for r in rows]
    graph = [r["graph_ms"] for r in rows]
    b1 = ax.bar(x - w / 2, eager, w, color=SLATE, label="Eager step", zorder=2)
    b2 = ax.bar(x + w / 2, graph, w, color=BLUE, label="CUDA graph", zorder=2)

    for bars, vals in ((b1, eager), (b2, graph)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.07, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="semibold")
    for i, r in enumerate(rows):
        ax.text(x[i] + w / 2, r["graph_ms"] + 0.36, f"{r['speedup']:.1f}×",
                ha="center", va="bottom", fontsize=8.4, color=BLUE)

    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d_state = {r['d_state']}" for r in rows])
    ax.set_ylabel("Per-frame latency  (ms)")
    ax.set_ylim(0, max(eager) * 1.22)
    ax.legend(loc="upper left", fontsize=9)

    lo = min(r["speedup"] for r in rows)
    hi = max(r["speedup"] for r in rows)
    p_lo = min(r["pct_of_60fps"] for r in rows)
    p_hi = max(r["pct_of_60fps"] for r in rows)
    _heading(fig, "Single-step decode",
             f"CUDA graph is {lo:.1f}–{hi:.1f}× faster than eager and uses "
             f"{p_lo:.1f}–{p_hi:.1f}% of a 16.7 ms frame")
    _footer(fig, f"{D['gpu']}  ·  3 layers, 67 joints, d_model=320  ·  "
                 f"100 timed steps after 30 warmup")
    _save(fig, "decode.png")


def plot_all(D) -> None:
    plot_scan(D)
    plot_decode_scaling(D)
    plot_chunk(D)
    plot_training(D)
    plot_alignment(D)
    plot_decode(D)


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
