"""Measure and plot the benchmark figures shown in README.md.

    python scripts/bench_figures.py            # measure on GPU, then plot
    python scripts/bench_figures.py --platform # measure the portable subset only
    python scripts/bench_figures.py --replot   # redraw from the stored json

Writes assets/*.png, assets/results.json and assets/platform_<key>.json.

Measuring runs on whatever accelerator the torch build exposes, and on CPU if
there is none. What a backend cannot do is recorded as unavailable rather than
skipped silently, because the cross-platform figure is partly about which paths
exist where.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import textwrap
import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from figstyle import (BLUE, CORAL, FAINT, FOOT, GRID, INK, LABEL, META, MUTED,
                      NAVY, NOTE, PLUM, SEMI, SLATE, SUB, TEAL, TITLE, use_style)

ROOT = Path(__file__).resolve().parent.parent
# python puts this file's directory on sys.path, not the one it was launched
# from, so point at the repo root to reach mamba3.py.
sys.path.insert(0, str(ROOT))

from mamba3 import (
    InferenceParams,
    Mamba3,
    MixerModel,
    _chunk_scan,
    _new_graph,
    _recurrent_scan,
    initialize_states,
    update_graph_cache,
)

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
# Narrow canvas, large type: the image is nearly always scaled down to a column
# width, and that shrinks the text with it. See figstyle.
WIDTH = 6.4

use_style()


def _style(ax, *, log=False):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0, which="major")
    if log:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, which="minor")
    ax.xaxis.grid(False)


def _single(height=4.2):
    return plt.subplots(figsize=(WIDTH, height))


def _stacked(height=7.0, ratios=(1, 1), share_x=False):
    """Two panels one above the other.

    Stacking keeps the image close to square, so a phone that scales it to the
    device width does not shrink the type as much as a side-by-side layout would.
    """
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, height), sharex=share_x,
                             gridspec_kw={"height_ratios": list(ratios)})
    fig.subplots_adjust(hspace=0.34 if share_x else 0.50)
    return fig, axes


def _left(fig) -> float:
    """Left edge of the axes including their tick labels, in figure fractions.

    Headings sit here rather than at the canvas edge. The axes are inset by
    whatever the y tick labels need, which varies per figure, so a heading at
    x=0 lands a ragged distance to the left of the plot it belongs to. Aligning
    to the tick labels puts the title, the panel titles and the leftmost label
    on one line.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return min(ax.get_tightbbox(renderer).x0 for ax in fig.axes) / fig.bbox.width


def _heading(fig, title: str, subtitle: str = "", *, axes_title: bool = True) -> None:
    """Title, and at most one line under it, above the axes.

    One line by design. A figure competes with the prose around it, and a
    caption long enough to explain the figure is a caption the reader will skip
    along with the figure; the explaining belongs in the README.

    The reserved band includes room for the first panel's own title, which is
    drawn above its axes and would otherwise land on the subtitle.
    """
    h = fig.get_figheight()
    fig.subplots_adjust(top=1.0 - (0.40 + 0.40 * axes_title + 0.26 * bool(subtitle)) / h)
    x = _left(fig)

    # Panel titles default to the spine, which is inset from the tick labels the
    # heading lines up with. Pull them out so every left edge in the figure is
    # the same one.
    for ax in fig.axes:
        label = ax.get_title(loc="left")
        if label:
            pos = ax.get_position()
            ax.set_title(label, loc="left", x=(x - pos.x0) / pos.width)

    fig.text(x, 1.0 - 0.04 / h, title, fontsize=TITLE, fontweight="bold",
             color=INK, va="top")
    if subtitle:
        fig.text(x, 1.0 - 0.35 / h, subtitle, fontsize=SUB, color=MUTED, va="top")


def _footer(fig, setup: str) -> None:
    """The setup needed to read the numbers: hardware and shape.

    Deliberately the quietest thing on the canvas — small and pale, set well
    below the axes. It is the same line on every figure, so at full strength it
    reads as part of the plot and competes with it; at this weight it stays
    findable by anyone who wants to know what the numbers were measured on, and
    invisible to everyone else. Method notes are in the README, which is where
    someone questioning a number actually looks.

    Placed under whatever the bottom axes actually ends at, measured rather than
    assumed: a panel with an x label reaches further down than one with only
    tick labels, and a fixed offset that clears the first collides with the
    second.

    Wrapped to the width of the axes, because savefig crops to the widest thing
    on the canvas — so a footer allowed to overrun does not spill, it shrinks
    the figure it belongs to.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    y = min(b.y0 for b in boxes) / fig.bbox.height - 0.26 / fig.get_figheight()
    width = max(b.x1 for b in boxes) - min(b.x0 for b in boxes)
    x = _left(fig)

    line = fig.text(x, y, setup, fontsize=META, color=FAINT, va="top")
    # A few percent over is not worth a second line: the figure grows by less
    # than the line costs.
    if line.get_window_extent(renderer).width <= width * 1.06:
        return
    line.remove()
    per_char = len(setup) / max(1.0, _text_width(fig, setup, renderer))
    for i, part in enumerate(textwrap.wrap(setup, int(width * per_char))):
        fig.text(x, y - i * 0.14 / fig.get_figheight(), part, fontsize=META,
                 color=FAINT, va="top")


def _text_width(fig, text: str, renderer) -> float:
    probe = fig.text(0.0, -1.0, text, fontsize=META)
    width = probe.get_window_extent(renderer).width
    probe.remove()
    return width


def _save(fig, name: str) -> None:
    fig.savefig(ASSETS / name)
    plt.close(fig)
    print(f"wrote {name}")


# --------------------------------------------------------------------------
# Backend capabilities
# --------------------------------------------------------------------------
# Three things differ between backends and none of them are the math: how you
# wait for queued work, whether peak allocation is tracked, and whether graphs
# can be captured. Each is probed once here so the measurement code below reads
# the same on every device.
def pick_device() -> torch.device:
    """The accelerator this build actually exposes, else CPU.

    current_accelerator() names the compiled-in backend whether or not it can be
    reached, so is_available() is the part that decides.
    """
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()
    return torch.device("cpu")


def device_label(device) -> str:
    """Hardware name for the figure footers, as specific as the backend allows."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        # Metal exposes no device name; the chip is the useful identifier and
        # sysctl is where macOS keeps it.
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            chip = ""
        return f"{chip} (MPS)" if chip else "Apple Silicon (MPS)"
    return "CPU"


def platform_key(device) -> str:
    return device.type


def sync() -> None:
    """Wait for queued accelerator work. A no-op on CPU, where there is none."""
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


@lru_cache(maxsize=1)
def _cache_trim():
    """The callable that releases cached device blocks, or None.

    torch.accelerator.empty_cache is the neutral spelling but it reaches for a
    DeviceAllocator, which MPS does not register: the call raises an internal
    assert there while torch.mps.empty_cache works. Probing beats branching on
    the device name, and doing it once keeps the failure out of the timed loops.
    """
    if not torch.accelerator.is_available():
        return None
    try:
        torch.accelerator.empty_cache()
        return torch.accelerator.empty_cache
    except Exception:
        pass
    if torch.backends.mps.is_available():
        return torch.mps.empty_cache
    return None


def empty_cache() -> None:
    trim = _cache_trim()
    if trim is not None:
        trim()


@lru_cache(maxsize=1)
def peak_mem_supported() -> bool:
    """Whether this backend tracks peak allocation.

    MPS reports current and driver-allocated bytes but keeps no high-water mark,
    so the memory columns are genuinely unavailable there rather than zero.
    """
    if not torch.accelerator.is_available():
        return False
    try:
        torch.accelerator.reset_peak_memory_stats()
        torch.accelerator.max_memory_allocated()
    except Exception:
        return False
    return True


def graphs_supported(device) -> bool:
    """Whether this backend can capture an accelerator graph at all.

    Asks the same _new_graph that capture_graph uses, so the answer tracks what
    mamba3.py can actually reach. Only CUDA and XPU register an implementation;
    on MPS the neutral API reports "Graph is not supported on device type: mps"
    and the CUDA fallback cannot be instantiated.
    """
    if device.type == "cpu":
        return False
    try:
        _new_graph()
    except Exception:
        return False
    return True


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
    sync()

    times, spent = [], 0.0
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        block = time.perf_counter() - start
        spent += block
        times.append(block / iters * 1000.0)
        if spent > budget_s and len(times) >= 3:
            break
    return statistics.median(times)


def peak_mem_mb(fn):
    """Peak allocation during fn, in MiB, or None where the backend has no counter."""
    if not peak_mem_supported():
        fn()
        sync()
        return None
    empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    fn()
    sync()
    return torch.accelerator.max_memory_allocated() / 2**20


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


DECODE_BATCH, DECODE_LEN = 64, 64


# --------------------------------------------------------------------------
# Portable measurements
# --------------------------------------------------------------------------
# The quantities the cross-platform figure compares, in functions that both
# measure() and measure_platform() call. Sharing the code is the whole point: a
# column produced by a near-copy of the measurement would not be comparable to
# one produced by the original, and the difference would not be visible in the
# numbers.
def measure_scan_alignment(device) -> list:
    print("-- scan alignment (chunked vs recurrence)")
    rows = []
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
        rows.append({
            "rank": rank,
            "fwd_y": (y - y_ref).abs().max().item(),
            "fwd_h": (h - h_ref).abs().max().item(),
            "grad": gerr,
            "grad_scale": gscale,
            "grad_rel": gerr / gscale,
        })
        print(f"   R={rank}  {rows[-1]}")
    return rows


def measure_layer_alignment(device) -> list:
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
    rows = []
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
        rows.append({"tag": tag, "segmented": e_seg, "stepwise": e_step})
        print(f"   {tag:<24s} seg={e_seg:.2e} step={e_step:.2e}")
    return rows


def measure_stability_len(device) -> list:
    print("-- error vs sequence length")
    rows = []
    for L in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        args, state = scan_inputs(device, seqlen=L, rank=1)
        y_ref, h_ref = _recurrent_scan(*args, *state)
        y, h = _chunk_scan(*args, *state, chunk_size=64)
        rows.append({
            "seqlen": L,
            "err_y": (y - y_ref).abs().max().item(),
            "err_h": (h - h_ref).abs().max().item(),
            "scale_y": y_ref.abs().max().item(),
            "scale_h": h_ref.abs().max().item(),
        })
        print(f"   L={L:5d}  Y={rows[-1]['err_y']:.2e}  H={rows[-1]['err_h']:.2e}")
    return rows


def measure_backend_parity(device):
    """The same weights on CPU and on the accelerator, all three paths.

    The most direct statement of portability there is: not that each backend
    agrees with a reference computed on itself, but that two backends agree with
    each other. Meaningless without an accelerator, hence None on CPU.
    """
    if device.type == "cpu":
        return None
    print("-- cpu vs accelerator parity")
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
            p = InferenceParams(max_seqlen=128, max_batch_size=2)
            layer(x0.to(dev), inference_params=p)
            p.seqlen_offset = x0.shape[1]
            step = layer(token0.to(dev), inference_params=p)
        out[dev.type] = (y.detach().cpu(), x.grad.cpu(), step.cpu())

    row = {name: (out["cpu"][i] - out[device.type][i]).abs().max().item()
           for i, name in enumerate(("forward", "backward", "decode_step"))}
    print(f"   {row}")
    return row


def measure_scan_bench(device) -> list:
    print("-- scan bench")
    rows = []
    for is_mimo, rank in ((False, 1), (True, 2)):
        tag = f"MIMO(R={rank})" if is_mimo else "SISO"
        chunk = 64 // rank
        for seqlen in (32, 64, 128, 256, 512):
            args, state = scan_inputs(device, seqlen=seqlen, rank=rank)
            t_rec = timeit(lambda: _recurrent_scan(*args, *state))
            t_chunk = timeit(lambda: _chunk_scan(*args, *state, chunk_size=chunk))
            rows.append({
                "tag": tag, "seqlen": seqlen,
                "recurrent_ms": t_rec, "chunked_ms": t_chunk, "speedup": t_rec / t_chunk,
            })
            print(f"   {tag:10s} L={seqlen:3d} rec={t_rec:6.2f} chunk={t_chunk:5.2f} "
                  f"{t_rec / t_chunk:5.1f}x")
    return rows


def measure_train_bench(device) -> list:
    print("-- training step")
    rows = []
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
            rows.append({"tag": tag, "seqlen": seqlen, "ms": t, "us_per_token": per_tok})
            print(f"   {tag:10s} L={seqlen:4d} {t:7.2f} ms  {per_tok:.3f} us/token")
            del u
            empty_cache()
    return rows


def measure_decode_bench(device, graphs: bool) -> list:
    """Eager single-step decode, and graph replay where the backend has it.

    graph_ms is None rather than absent where capture is unavailable, so a reader
    of the json can tell "not supported here" from "not measured".
    """
    print("-- decode bench")
    rows = []
    for d_state in (16, 32, 64):
        m = MixerModel(320, n_layer=3, rms_norm=True,
                       ssm_cfg=dict(d_state=d_state, headdim=64)).to(device).eval().float()
        x = torch.randn(DECODE_BATCH, 1, 320, device=device)
        with torch.inference_mode():
            p = InferenceParams(max_seqlen=DECODE_LEN, max_batch_size=DECODE_BATCH)
            p.key_value_memory_dict = m.allocate_inference_cache(DECODE_BATCH, DECODE_LEN)
            p.seqlen_offset = 1
            t_eager = timeit(lambda: m(x, inference_params=p), iters=100, warmup=30)
            t_graph = None
            if graphs:
                cache = update_graph_cache(m, None, DECODE_BATCH, 0, DECODE_LEN)
                t_graph = timeit(lambda: cache.run(x), iters=100, warmup=30)
        rows.append({
            "d_state": d_state, "eager_ms": t_eager, "graph_ms": t_graph,
            "speedup": t_eager / t_graph if t_graph else None,
            "pct_of_60fps": (t_graph or t_eager) / 16.7 * 100,
        })
        print(f"   N={d_state:2d} eager={t_eager:.3f} graph="
              + (f"{t_graph:.3f} {t_eager / t_graph:.1f}x" if t_graph else "unsupported"))
    return rows


def measure_platform(device) -> dict:
    """The portable subset, for one backend, written to assets/platform_<key>.json.

    A cross-platform column needs the same measurement on every machine, and a
    full measure() run needs a graph backend and a peak-memory counter that not
    every device has. This is the part that runs anywhere.
    """
    torch.manual_seed(0)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    graphs = graphs_supported(device)
    try:
        payload = {
            "platform": device_label(device),
            "key": platform_key(device),
            "device": device.type,
            "torch": torch.__version__,
            "graphs": graphs,
            "peak_mem": peak_mem_supported(),
            "scan_alignment": measure_scan_alignment(device),
            "layer_alignment": measure_layer_alignment(device),
            "stability_len": measure_stability_len(device),
            "backend_parity": measure_backend_parity(device),
            "scan_bench": measure_scan_bench(device),
            "train_bench": measure_train_bench(device),
            "decode_bench": measure_decode_bench(device, graphs),
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32

    path = ASSETS / f"platform_{payload['key']}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path.name}")
    return payload


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def measure(device) -> dict:
    torch.manual_seed(0)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    graphs = graphs_supported(device)

    scan_alignment = measure_scan_alignment(device)

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

    layer_alignment = measure_layer_alignment(device)

    print("-- graph replay")
    cg_err = cg_reset = None
    if not graphs:
        print(f"   unsupported ({device.type} has no graph capture backend)")
    else:
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

    stability_len = measure_stability_len(device)
    backend_parity = measure_backend_parity(device)

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
    empty_cache()

    torch.backends.cuda.matmul.allow_tf32 = tf32

    scan_bench = measure_scan_bench(device)
    train_bench = measure_train_bench(device)

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
            empty_cache()

    decode_bench = measure_decode_bench(device, graphs)
    decode_bench_params = {"batch": DECODE_BATCH, "n_layer": 3, "d_model": 320, "headdim": 64}

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
            if graph and not graphs:
                row[tag] = row[tag + "_mib"] = None
                continue
            dynamo.reset()
            empty_cache()
            if peak_mem_supported():
                torch.accelerator.reset_peak_memory_stats()
            m = MixerModel(320, n_layer=3, rms_norm=True,
                           ssm_cfg=dict(d_state=32, headdim=64)).to(device).eval().float()
            x = torch.randn(batch, 1, 320, device=device)
            try:
                with torch.inference_mode():
                    fn = m if mode is None else torch.compile(m, mode=mode, fullgraph=True)
                    p = InferenceParams(max_seqlen=DECODE_LEN, max_batch_size=batch)
                    p.key_value_memory_dict = m.allocate_inference_cache(batch, DECODE_LEN)
                    p.seqlen_offset = 1
                    for _ in range(5):
                        fn(x, inference_params=p)
                    if graph:
                        cache = update_graph_cache(fn, None, batch, 0, DECODE_LEN)
                        call = lambda: cache.run(x)
                    else:
                        call = lambda: fn(x, inference_params=p)
                    row[tag] = timeit(call, iters=100, warmup=30)
                row[tag + "_mib"] = (torch.accelerator.max_memory_allocated() / 2**20
                                     if peak_mem_supported() else None)
            except Exception as exc:
                row[tag] = row[tag + "_mib"] = None
                print(f"   B={batch} {tag} failed: {type(exc).__name__}")
        compile_bench.append(row)
        done = [k for k in row
                if not k.endswith("_mib") and k != "batch" and row[k] is not None]
        print(f"   B={batch} " + "  ".join(
            f"{k}={row[k]:.3f}" + (f"/{row[k + '_mib']:.0f}MiB" if row[k + "_mib"] else "")
            for k in done))
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
                empty_cache()
                print(f"   L={L:5d}  {tag}: out of memory")
        long_context.append(row)
        shown = "   ".join(
            f"{tag}={row[f'{tag}_ms']:7.2f}ms"
            + (f"/{row[f'{tag}_mb']:8.1f}MB" if row[f"{tag}_mb"] is not None else "")
            for tag in ("chunked", "recurrent") if row[f"{tag}_ms"] is not None)
        print(f"   L={L:5d}  {shown}")
        del args, state
        empty_cache()
    long_context_params = {"batch": 4, "nheads": 6, "headdim": 64, "d_state": 64,
                           "chunk_size": 64}

    payload = {
        "gpu": device_label(device),
        "device": device.type,
        "torch": torch.__version__,
        "graphs": graphs,
        "peak_mem": peak_mem_supported(),
        "official_package": "mamba_ssm not installed — official fused kernels not compared",
        "backend_parity": backend_parity,
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
        ax_t.plot(xs, [r["recurrent_ms"] for r in sel], color=color, ls=(0, (3.2, 2.2)),
                  lw=1.9, marker="o", ms=6.5, mfc="white", mew=1.8, label=tag)
        ax_t.plot(xs, [r["chunked_ms"] for r in sel], color=color, marker="s", ms=5.8)
        ax_s.plot(xs, [r["speedup"] for r in sel], color=color,
                  marker="o", ms=7.0, mfc="white", mew=1.9)

    # The legend carries colour only, so line style is spelled out once, on the
    # title line where no data can run through it. The lower panel then needs no
    # legend at all: same two colours, directly below.
    ax_t.text(1.0, 1.015, "dashed step-by-step   ·   solid chunked", ha="right",
              va="bottom", transform=ax_t.transAxes, color=MUTED, fontsize=FOOT)

    for ax in (ax_t, ax_s):
        _style(ax)
        ax.set_xticks(seqlens)
        ax.set_xticklabels(["32", "", "128", "256", "512"])
        ax.set_xlim(16, 568)
    ax_s.set_xlabel("Sequence length")
    ax_t.set_title("Scan time  (ms)", loc="left")
    ax_s.set_title("Speedup over the recurrence  (×)", loc="left")
    ax_t.set_ylim(0, 80)
    ax_s.set_ylim(0, 78)
    ax_t.legend(loc="upper left", ncol=2, columnspacing=1.4, handlelength=1.8)
    ax_t.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_s.yaxis.set_major_locator(mticker.MultipleLocator(20))

    tail = [r["speedup"] for r in rows if r["seqlen"] == seqlens[-1]]
    lo, hi = round(min(tail)), round(max(tail))
    ax_s.annotate(f"{lo}×" if lo == hi else f"{lo}–{hi}×",
                  (seqlens[-1], max(tail)), textcoords="offset points", xytext=(-8, 12),
                  ha="right", color=INK, fontsize=NOTE, fontweight="bold")

    _heading(fig, "Chunked GEMM vs step-by-step recurrence",
             "Recurrent cost grows with L; the chunked path stays near 1 ms")
    _footer(fig, f"{D['gpu']}  ·  fp32, B=3, H=4, P=16, N=32  ·  chunk 64 / 32")
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
                  (xs[-1], last["mamba_state_mb"]), textcoords="offset points", xytext=(-8, 16),
                  ha="right", color=TEAL, fontsize=NOTE, fontweight="bold")

    for ax in (ax_t, ax_m):
        _style(ax, log=True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v // 1024}k" if v >= 1024 else str(v) for v in xs])
    ax_t.legend(loc="upper left")
    ax_m.set_xlabel("Context already consumed  (tokens)")
    ax_m.set_yscale("log")
    ax_t.set_ylim(0, max(r["attn_ms"] for r in rows) * 1.12)
    ax_t.set_title("One more token, whatever the context  (ms)", loc="left")
    ax_m.set_title("What has to be kept around  (MiB per layer)", loc="left")

    _heading(fig, "Constant state vs a growing KV cache",
             "Decode cost and memory do not depend on the context behind them")
    _footer(fig, f"{D['gpu']}  ·  fp32, one layer, B={meta['batch']}, "
                 f"d_model={meta['d_model']}, {meta['nheads']} heads  ·  "
                 f"baseline: PyTorch SDPA")
    _save(fig, "decode_scaling.png")


def plot_chunk(D) -> None:
    sweep, speed = D["chunk_sweep"], D["chunk_speed"]
    fig, (ax_e, ax_t) = _stacked(height=7.3)

    for rank, color in ((1, TEAL), (2, CORAL), (4, NAVY)):
        sel = [r for r in sweep if r["rank"] == rank]
        ax_e.plot([r["chunk_size"] for r in sel], [r["err"] for r in sel], color=color, lw=2.0,
                  marker="o", ms=6.0, mfc="white", mew=1.7, label=f"R={rank}")
    ax_e.axhline(1e-4, color=SLATE, lw=1.0, ls=(0, (3, 2)))
    # Exponents are written 1e-4 rather than 1×10⁻⁴: the figure font has the
    # superscript digits but no superscript minus, and e-notation matches the
    # formatted values elsewhere in these figures anyway.
    ax_e.text(1.05, 1.25e-4, "test tolerance  1e-4", color=MUTED, fontsize=FOOT,
              va="bottom")

    for i, (rank, color) in enumerate(((1, TEAL), (2, CORAL))):
        sel = [r for r in speed if r["rank"] == rank]
        ax_t.plot([r["chunk_size"] for r in sel], [r["ms"] for r in sel], color=color,
                  marker="s", ms=6.0, label=f"R={rank}")
        best = min(sel, key=lambda r: r["ms"])
        ax_t.text(0.33, 0.90 - 0.13 * i, f"R={rank} fastest at Q={best['chunk_size']}",
                  transform=ax_t.transAxes, color=color, fontsize=NOTE,
                  fontweight=SEMI)

    for ax in (ax_e, ax_t):
        _style(ax, log=True)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Chunk size Q")
    # Only the upper panel gets a legend; below, the two coloured callouts name
    # their own series.
    ax_e.legend(loc="center left", ncol=3, columnspacing=1.2, handlelength=1.8)
    ax_e.set_yscale("log")
    ax_e.set_xticks([1, 4, 16, 64, 256])
    ax_e.set_xticklabels(["1", "4", "16", "64", "256"])
    ax_e.set_ylim(2e-6, 3e-4)
    ax_e.set_title("Every chunk size agrees  (max |Δ| vs recurrence)", loc="left")
    ax_t.set_xticks([8, 16, 32, 64, 128, 256])
    ax_t.set_xticklabels(["8", "16", "32", "64", "128", "256"])
    ax_t.set_title("Only the speed moves  (ms)", loc="left")

    _heading(fig, "Chunk size is a performance knob, not a correctness one",
             "Q partitions the same algebra; only the GEMM shape changes")
    _footer(fig, f"{D['gpu']}  ·  fp32, TF32 off  ·  "
                 f"error at L=256, timing at L=512")
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
    ax_t.legend(loc="upper left")            # same two colours below, said once
    ax_p.set_xlabel("Sequence length")
    ax_t.set_ylim(bottom=0)
    ax_p.set_ylim(bottom=0)
    ax_t.set_title("Forward + backward  (ms)", loc="left")
    ax_p.set_title("Cost per token  (µs)", loc="left")

    _heading(fig, "Autograd straight through the chunked scan",
             "Per-token cost bottoms out near L ≈ 512–1024")
    _footer(fig, f"{D['gpu']}  ·  fp32, B=8, d_model=384, N=64, P=64")
    _save(fig, "training.png")


def plot_alignment(D) -> None:
    rows = D["layer_alignment"]
    scan = D["scan_alignment"]
    # "MIMO" is dropped from the last two: R=4 only exists there, and spelling it
    # out makes those two tick labels touch.
    labels = ["SISO", "SISO\n+ norm", "SISO\nrope = 1",
              "MIMO\nR=2", "R=4\n+ norm", "R=4\nunfused"]

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
    ax_l.text(len(rows) - 0.42, 1.2e-6, "1e-6", color=MUTED, fontsize=FOOT,
              ha="right", va="bottom")
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=11.2)
    ax_l.set_title("Resumed state vs one full forward  (max |Δ|)", loc="left")
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
    ax_s.set_title("Chunked scan vs recurrence  (max |Δ|)", loc="left")
    ax_s.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.5)

    _heading(fig, "Numerical agreement",
             "Every path reproduces the reference to ~1e-6 or better")
    _footer(fig, f"{D['gpu']}  ·  fp32, TF32 off  ·  B=3, L=40")
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
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.08, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=NOTE, color=INK,
                    fontweight=SEMI)
    for i, r in enumerate(rows):
        ax.text(x[i] + w / 2, r["graph_ms"] + 0.50, f"{r['speedup']:.1f}×",
                ha="center", va="bottom", fontsize=NOTE, color=BLUE, fontweight="bold")

    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"d_state = {r['d_state']}" for r in rows])
    ax.set_title("Per-frame latency  (ms)", loc="left")
    ax.set_ylim(0, max(eager) * 1.30)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.5)

    lo = min(r["speedup"] for r in rows)
    hi = max(r["speedup"] for r in rows)
    _heading(fig, "Single-step decode",
             f"Graph replay removes the launch overhead: {lo:.1f}–{hi:.1f}× faster")
    m = D.get("decode_bench_params", {"batch": 64, "n_layer": 3, "d_model": 320})
    _footer(fig, f"{D['gpu']}  ·  fp32, {m['n_layer']} layers, batch {m['batch']}, "
                 f"d_model={m['d_model']}")
    _save(fig, "decode.png")


def plot_stability(D) -> None:
    rows, drift = D["stability_len"], D["decode_drift"]
    fig, (ax_l, ax_d) = _stacked(height=7.3)

    xs = [r["seqlen"] for r in rows]
    ax_l.plot(xs, [r["err_y"] for r in rows], color=TEAL, marker="s", ms=6.0, label="Output Y")
    ax_l.plot(xs, [r["err_h"] for r in rows], color=CORAL, lw=2.0, marker="o", ms=6.5,
              mfc="white", mew=1.8, label="Final state h")
    ax_l.axhline(1e-6, color=SLATE, lw=1.0, ls=(0, (3, 2)))
    ax_l.text(xs[0], 1.25e-6, "1e-6", color=MUTED, fontsize=FOOT, va="bottom")
    _style(ax_l, log=True)
    ax_l.set_xscale("log", base=2)
    ax_l.set_yscale("log")
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels([f"{v // 1024}k" if v >= 1024 else str(v) for v in xs])
    ax_l.set_ylim(1e-7, 3e-5)
    ax_l.set_xlabel("Sequence length")
    ax_l.set_title("Chunked scan vs recurrence  (max |Δ|)", loc="left")
    ax_l.legend(loc="upper left", ncol=2, columnspacing=1.2, handlelength=1.8)

    err = np.maximum(np.asarray(drift["err"], dtype=float), 1e-12)
    steps = np.arange(1, len(err) + 1)
    running = np.maximum.accumulate(err)
    ax_d.plot(steps, err, color="#C3C8D1", lw=0.8, label="per step")
    ax_d.plot(steps, running, color=BLUE, label="running max")
    ax_d.annotate(f"{running[-1]:.1e} after {len(err)} steps",
                  (steps[-1], running[-1]), textcoords="offset points", xytext=(-8, 10),
                  ha="right", va="bottom", color=BLUE, fontsize=NOTE, fontweight="bold")
    _style(ax_d, log=True)
    ax_d.set_yscale("log")
    ax_d.set_ylim(2e-7, 6e-6)
    ax_d.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_d.set_xlim(0, len(err))
    ax_d.set_xlabel("Decode step")
    ax_d.set_title("Streaming step() vs one full forward  (max |Δ|)", loc="left")
    # Below the noise band, which bottoms out well above the axis; the callout
    # has the space above it.
    ax_d.legend(loc="lower right", ncol=2, columnspacing=1.2, handlelength=1.8)

    _heading(fig, "Error does not grow with length or with time",
             "Neither a longer sequence nor a thousand streamed steps add drift")
    _footer(fig, f"{D['gpu']}  ·  fp32, TF32 off  ·  chunk=64")
    _save(fig, "stability.png")


def plot_long_context(D) -> None:
    rows, meta = D["long_context"], D["long_context_params"]
    xs = [r["seqlen"] for r in rows]
    fig, (ax_t, ax_m) = _stacked(height=7.1, share_x=True)

    def series(key):
        return ([r["seqlen"] for r in rows if r[key] is not None],
                [r[key] for r in rows if r[key] is not None])

    for ax, kc, kr, title in (
        (ax_t, "chunked_ms", "recurrent_ms", "Forward + backward  (ms, log)"),
        (ax_m, "chunked_mb", "recurrent_mb", "Autograd footprint  (MiB)"),
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
        ax.set_title(title, loc="left")
    ax_t.legend(loc="upper left")            # same two colours below, said once
    ax_m.set_xlabel("Sequence length")

    last = rows[-1]
    # Above the chunked line in the log panel, where the gap between the two
    # series is wide; below it in the linear one, where the recurrence curve
    # sweeps through that space.
    for ax, kc, kr, unit, dy in ((ax_t, "chunked_ms", "recurrent_ms", "faster", 14),
                                 (ax_m, "chunked_mb", "recurrent_mb", "smaller", -18)):
        if last[kc] and last[kr]:
            ax.annotate(f"{last[kr] / last[kc]:.0f}× {unit} at {last['seqlen'] // 1024}k",
                        (last["seqlen"], last[kc]), textcoords="offset points",
                        xytext=(-8, dy), ha="right",
                        va="bottom" if dy > 0 else "top",
                        color=TEAL, fontsize=NOTE, fontweight="bold")

    _heading(fig, "What makes long sequences trainable at all",
             "Backprop through L Python steps is the wall a plain reference hits")
    _footer(fig, f"{D['gpu']}  ·  fp32, B={meta['batch']}, H={meta['nheads']}, "
                 f"P={meta['headdim']}, N={meta['d_state']}, chunk={meta['chunk_size']}")
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
            ax.text(v + span * 0.016, rect.get_y() + rect.get_height() / 2,
                    f"{v:.3f} ms{gain}", va="center", ha="left", fontsize=FOOT + 0.6,
                    color=INK, fontweight="bold" if tag == "compile_graph" else "normal")

    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in tags], fontsize=LABEL)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.50)
    ax.set_xlabel("Per-token latency  (ms)")
    # Right of the middle rows: the only block of space the labels leave free,
    # and it keeps the legend off the highlighted bottom row.
    ax.legend(loc="center right", ncol=1, handlelength=1.5, labelspacing=0.5)

    best = max(r["eager"] / r["compile_graph"] for r in rows)
    worst = min(r["eager"] / r["compile_graph"] for r in rows)
    _heading(fig, "Compilation and replay compose",
             f"Each removes a different cost; together {worst:.0f}–{best:.0f}× "
             f"faster than eager", axes_title=False)
    _footer(fig, f"{D['gpu']}  ·  fp32, 3 layers, d_model=320, d_state=32")
    _save(fig, "compile.png")


# Drawn in this order wherever a platform axis appears, so the figure does not
# reshuffle when a machine is added.
PLATFORM_ORDER = ("cuda", "xpu", "mps", "cpu")
PLATFORM_COLORS = {"cuda": TEAL, "xpu": PLUM, "mps": CORAL, "cpu": NAVY}

# Checks every backend runs against its own reference. The direct CPU-vs-
# accelerator comparison is deliberately not here: it does not exist for CPU and
# was not recorded on the CUDA run, so it would draw as one lone bar in a group
# of three and read as missing data. It belongs in the README table, where the
# gap can be named.
CONSISTENCY_CHECKS = (
    ("chunked", "Chunked\nvs recur."),
    ("grad", "Gradients\n(rel.)"),
    ("segmented", "Segmented\nresume"),
    ("stepwise", "step()\npath"),
)


def _platform_view(payload):
    """The comparable subset of a results.json or platform_<key>.json payload.

    Both files carry these fields for the same quantities because the same
    functions wrote them, which is what makes a column from one machine
    comparable to a column from another.
    """
    key = payload.get("key") or payload.get("device")
    if key is None:
        # results.json predates these keys. It records the build it ran on, and
        # a +cu wheel is a CUDA run; nothing else in the file needs guessing.
        key = "cuda" if "+cu" in payload.get("torch", "") else None
        if key is None:
            return None
    return {
        "key": key,
        "label": payload.get("platform") or payload.get("gpu") or key,
        "torch": payload.get("torch", ""),
        "graphs": payload.get("graphs"),
        "scan_alignment": payload.get("scan_alignment"),
        "layer_alignment": payload.get("layer_alignment"),
        "backend_parity": payload.get("backend_parity"),
        "scan_bench": payload.get("scan_bench"),
        "train_bench": payload.get("train_bench"),
        "decode_bench": payload.get("decode_bench"),
    }


def load_platforms() -> list:
    """Every backend measured so far, in PLATFORM_ORDER.

    The full run in results.json counts as one of them, so the machine that
    draws all the other figures needs no second pass. A --platform re-run of the
    same backend does not displace it: the full run measured strictly more.
    """
    views = {}
    for path in [ASSETS / "results.json", *sorted(ASSETS.glob("platform_*.json"))]:
        if not path.exists():
            continue
        view = _platform_view(json.loads(path.read_text(encoding="utf-8")))
        if view is not None:
            views.setdefault(view["key"], view)
    return [views[k] for k in PLATFORM_ORDER if k in views]


def _consistency(view) -> dict:
    """max |Δ| per check for one backend; None where the check was not recorded."""
    scan = view.get("scan_alignment") or []
    layer = view.get("layer_alignment") or []
    return {
        "chunked": max((r["fwd_y"] for r in scan), default=None),
        "grad": max((r["grad_rel"] for r in scan), default=None),
        "segmented": max((r["segmented"] for r in layer), default=None),
        "stepwise": max((r["stepwise"] for r in layer), default=None),
    }


def plot_cross_platform(platforms) -> None:
    """Agreement and portable speedup, side by side across backends.

    Colour means backend in both panels, which is why the upper one groups by
    check rather than colouring by it: one legend, one meaning, read once.

    The lower panel plots the speedup rather than the times on purpose. A 4090
    and a laptop chip cannot be compared in milliseconds without the figure
    turning into a hardware review, but the ratio of the two code paths on one
    machine is a property of the code, and that is the claim being made.
    """
    if len(platforms) < 2:
        print("skipping cross_platform.png (needs two backends; "
              "run --platform on another device)")
        return

    fig, (ax_c, ax_s) = _stacked(height=7.6, ratios=(1.15, 1.0))

    rows = {p["key"]: _consistency(p) for p in platforms}
    x = np.arange(len(CONSISTENCY_CHECKS))
    w = 0.78 / len(platforms)
    for i, p in enumerate(platforms):
        off = (i - (len(platforms) - 1) / 2) * w
        vals = [rows[p["key"]][key] for key, _ in CONSISTENCY_CHECKS]
        # A missing check leaves a gap rather than a zero-height bar, which on a
        # log axis would draw as a full-height one.
        ax_c.bar([xi + off for xi, v in zip(x, vals) if v is not None],
                 [v for v in vals if v is not None], w,
                 color=PLATFORM_COLORS.get(p["key"], SLATE),
                 label=p["label"], zorder=2)

    _style(ax_c, log=True)
    ax_c.set_yscale("log")
    # Headroom for two rows of annotation above the tallest bar: the legend, and
    # the tolerance line with its label.
    ax_c.set_ylim(4e-8, 4e-3)
    ax_c.axhline(1e-4, color=SLATE, lw=1.0, ls=(0, (3, 2)))
    ax_c.text(len(CONSISTENCY_CHECKS) - 0.45, 1.2e-4, "self-check tolerance  1e-4",
              color=MUTED, fontsize=FOOT, ha="right", va="bottom")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([label for _, label in CONSISTENCY_CHECKS], fontsize=11.2)
    ax_c.set_title("Same weights, same answers  (max |Δ|)", loc="left")
    ax_c.legend(loc="upper left", ncol=len(platforms), columnspacing=1.1,
                handlelength=1.4, fontsize=11.4)

    for p in platforms:
        sel = [r for r in (p.get("scan_bench") or []) if r["tag"] == "SISO"]
        if not sel:
            continue
        ax_s.plot([r["seqlen"] for r in sel], [r["speedup"] for r in sel],
                  color=PLATFORM_COLORS.get(p["key"], SLATE), marker="o", ms=6.2,
                  mfc="white", mew=1.8)
        last = sel[-1]
        ax_s.annotate(f"{last['speedup']:.0f}×", (last["seqlen"], last["speedup"]),
                      textcoords="offset points", xytext=(-6, 9), ha="right",
                      color=PLATFORM_COLORS.get(p["key"], SLATE), fontsize=NOTE,
                      fontweight="bold")

    _style(ax_s, log=True)
    ax_s.set_xscale("log", base=2)
    ax_s.set_yscale("log")
    ax_s.set_xticks([32, 64, 128, 256, 512])
    ax_s.set_xticklabels(["32", "64", "128", "256", "512"])
    # A decade locator would label 10× and nothing else across a range that only
    # spans 2×–70×, so the ticks are named outright.
    ax_s.set_yticks([2, 5, 10, 20, 50])
    ax_s.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax_s.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_s.set_xlabel("Sequence length")
    ax_s.set_title("Chunked scan over the recurrence, SISO  (×)", loc="left")

    _heading(fig, "The same file on every backend",
             "Agreement to ~1e-6 everywhere; the chunked win is not CUDA-specific")
    _footer(fig, "  ·  ".join(f"{p['label']}, torch {p['torch']}" for p in platforms)
                 + "  ·  fp32, TF32 off")
    _save(fig, "cross_platform.png")


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


def _guard_results(device, force: bool) -> None:
    """Refuse to replace a recorded run from another backend by accident.

    results.json backs almost every figure in the README, and the prose around
    them names the machine they were measured on. Measuring no longer requires
    CUDA, so without this a run on a laptop would quietly replace that dataset
    with one the surrounding text no longer describes. --platform is the way to
    add a backend without touching it.
    """
    path = ASSETS / "results.json"
    if force or not path.exists():
        return
    old = json.loads(path.read_text(encoding="utf-8"))
    was = old.get("device") or ("cuda" if "+cu" in old.get("torch", "") else None)
    if was is None or was == device.type:
        return
    raise SystemExit(
        f"assets/results.json holds a {was} run ({old.get('gpu', 'unknown')}), and "
        f"measuring on {device.type} would replace it.\n"
        f"Use --platform to add {device.type} to the cross-platform figure, or "
        f"--force to overwrite anyway."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="redraw from the stored json")
    ap.add_argument("--platform", action="store_true",
                    help="measure only the portable subset, for the cross-platform figure")
    # The CPU column of the cross-platform figure has to be measurable on a
    # machine that has an accelerator, or it never gets measured at all.
    ap.add_argument("--device", default=None,
                    help="measure on this device instead of the one torch picks")
    ap.add_argument("--force", action="store_true",
                    help="overwrite results.json even if it holds another backend's run")
    args = ap.parse_args()

    if args.platform:
        device = torch.device(args.device) if args.device else pick_device()
        print(f"portable subset on {device_label(device)}  ({device.type})")
        measure_platform(device)
        plot_cross_platform(load_platforms())
        print("done")
        return

    if args.replot:
        D = json.loads((ASSETS / "results.json").read_text(encoding="utf-8"))
    else:
        device = torch.device(args.device) if args.device else pick_device()
        assert device.type != "cpu", "measuring needs an accelerator; use --replot"
        _guard_results(device, args.force)
        torch.backends.cudnn.benchmark = True
        D = measure(device)
    plot_all(D)
    plot_cross_platform(load_platforms())
    print("done")


if __name__ == "__main__":
    main()
