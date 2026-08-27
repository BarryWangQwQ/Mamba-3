<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-only-ee4c2c?style=flat-square" alt="PyTorch only">
  <img src="https://img.shields.io/badge/CUDA-no%20nvcc-0f7b6c?style=flat-square" alt="No nvcc">
  <img src="https://img.shields.io/badge/API-mamba__ssm-1e3a5f?style=flat-square" alt="Official API">
  <img src="https://img.shields.io/badge/paper-arXiv%3A2603.15569-b31b1b?style=flat-square" alt="Paper">
</p>

<h1 align="center">Mamba-3</h1>

<p align="center">
  The whole of <a href="https://arxiv.org/abs/2603.15569">Mamba-3</a> in one file —
  official <a href="https://github.com/state-spaces/mamba"><code>mamba_ssm</code></a> API,
  pure PyTorch, no kernels to build.
</p>

<p align="center">
  <img src="assets/architecture.png" width="560" alt="The data path of one Mamba-3 layer">
</p>

Copy [`mamba3.py`](mamba3.py) into your project, `pip install torch`, and you have Mamba-3 running on the GPU: SISO and MIMO, chunked prefill, single-step decode, CUDA-graph replay, gradients included.

**Nothing to compile.** No nvcc, no CUDA Toolkit, no MSVC, no Triton, no TileLang, no einops, no `mamba-ssm`. A CUDA-enabled PyTorch build is the entire dependency list — which is what makes this work on Windows, inside a locked-down container, or on a cluster where you cannot install a toolchain.

**Not a lookalike.** This file replaces the official *kernel backend*, not the interface. Module names, arguments, attributes, state layouts and calling conventions follow `mamba_ssm` exactly, and parameter names line up one-to-one with the official modules — so upstream code and docs apply to it unchanged.

<p align="center">
  <img src="assets/headline.png" width="560" alt="72x faster scan, 0.51 ms decode, 247x smaller state, 1.1e-6 max error">
</p>

---

## Install

```bash
pip install torch
```

```python
import torch
from mamba3 import Mamba3, MixerModel, InferenceParams, update_graph_cache

layer = Mamba3(d_model=256, d_state=64, headdim=64).cuda()
y = layer(torch.randn(2, 128, 256, device="cuda"))          # (B, L, D) → (B, L, D)

model = MixerModel(256, n_layer=4, rms_norm=True,
                   ssm_cfg=dict(d_state=64, headdim=64)).cuda().eval()
```

SISO is the default. MIMO and the Mamba-3 extras use the same argument names as upstream:

```python
Mamba3(
    d_model=256, d_state=64, headdim=64,
    is_mimo=True, mimo_rank=4,
    chunk_size=16,              # 64 for SISO, 64 / mimo_rank for MIMO
    is_outproj_norm=True,
    rope_fraction=0.5,          # 0.5 or 1.0
)
```

## Prefill, decode, CUDA graph

`inference_params` follows the official convention: `seqlen_offset == 0` is the chunked prefill; `seqlen_offset > 0` and `seqlen == 1` is single-step decode. Segmented forwards resume from the cached state.

```python
params = InferenceParams(max_seqlen=1024, max_batch_size=2)
y = model(x, inference_params=params)       # prefill
params.seqlen_offset += x.shape[1]

token = torch.randn(2, 1, 256, device="cuda")
y_t = model(token, inference_params=params) # eager decode
params.seqlen_offset += 1

cache = update_graph_cache(model, None, batch_size=2, seqlen_og=0, max_seqlen=1024)
y_t = cache.run(token)                      # one launch for the whole stack
```

---

## Alignment with official

Official fused kernels (Triton SISO / TileLang MIMO) are **not** compared: they need `mamba-ssm` plus nvcc / Triton / TileLang, which this file is written to avoid. What is compared is every path in this file against the one reference that does not depend on those kernels — the plain step-by-step recurrence, and a single full-sequence forward.

RTX 4090, PyTorch 2.12.1+cu132, fp32, TF32 off:

<div align="center">

| | Compared against | max \|Δ\| |
|---|---|---:|
| Chunked scan, R = 1 / 2 / 4 | Recurrence (Y, final state) | ≤ 5.7×10⁻⁶ |
| Chunked-scan gradients | Recurrence, d(Y,h)/d(V,K,Q,ADT,α,β) | rel. ~3×10⁻⁷ |
| Chunked scan, Q = 1 … 256 | Recurrence, every chunk size | ≤ 1.4×10⁻⁵ |
| Chunked scan, L = 32 … 4096 | Recurrence, every length | ≤ 4.8×10⁻⁶ |
| Segmented `inference_params` | One full forward | ≤ 6.0×10⁻⁷ |
| Official `step()` path | One full forward | ≤ 1.1×10⁻⁶ |
| 1024 consecutive `step()` calls | One full forward | ≤ 2.9×10⁻⁶ |
| CUDA graph, then after reset | Eager decode | 3.6×10⁻⁷ / 1.2×10⁻⁷ |
| State shapes, B/C bias, `in_proj`, RoPE, `Block`, `rms_norm_ref` | `mamba_ssm` conventions | match |

</div>

Both state paths reproduce a single full forward to ~10⁻⁶, and the backward pass agrees with the recurrence to ~3×10⁻⁷ relative — so the file is usable for training, not only for inference.

<p align="center">
  <img src="assets/alignment.png" width="600" alt="Numerical agreement with a full forward">
</p>

The error also does not accumulate. It is flat from `L=32` to `L=4096`, and a streaming state replayed a thousand steps in a row does not drift away from the batch forward.

<p align="center">
  <img src="assets/stability.png" width="600" alt="Error vs sequence length and vs decode step">
</p>

`chunk_size` is a tuning knob only: `Q` partitions the same algebra, so the result holds across `Q = 1 … 256` while the GEMM shape — and the speed — changes. The fastest `Q` sits at or just above the documented default of `64 / mimo_rank`.

<p align="center">
  <img src="assets/chunk_size.png" width="600" alt="Error and time vs chunk size">
</p>

Two deviations are intentional — the official counterparts are language-model only:

- **`MixerModel` drops the embedding.** This stack takes `(batch, seqlen, d_model)` features directly.
- **`capture_graph` uses `hidden_states`.** Official static buffers are `input_ids` / `position_ids`.

Everything else follows the official code line by line. Not implemented: `cu_seqlens`, `seq_idx`, tensor parallelism, kernel-level fusion.

---

## Benchmarks

Same machine as above. Every number and figure comes from `python bench_figures.py`.

### Chunked scan vs the recurrence

Recurrent time grows linearly with `L`; chunked time stays near 1 ms, because the inner work is cuBLAS GEMMs.

<p align="center">
  <img src="assets/scan.png" width="600" alt="Selective scan: time and speedup">
</p>

<div align="center">

| | L=32 | L=128 | L=512 |
|---|---:|---:|---:|
| SISO chunked | 0.71 ms | 0.76 ms | 0.86 ms |
| SISO vs recurrence | 5.4× | 20.2× | **71.9×** |
| MIMO(R=2) chunked | 0.77 ms | 0.88 ms | 1.07 ms |
| MIMO(R=2) vs recurrence | 6.3× | 21.5× | **69.2×** |

</div>

### Why long sequences are trainable at all

Backprop through `L` Python steps is the wall a readable reference implementation hits — in wall-clock time and in the size of the autograd graph.

<p align="center">
  <img src="assets/long_context.png" width="600" alt="Training step time and autograd footprint vs length">
</p>

<div align="center">

| Sequence length | 128 | 1k | 4k |
|---|---:|---:|---:|
| Recurrence, fwd + bwd | 79 ms | 652 ms | 2615 ms |
| Chunked scan, fwd + bwd | 2.9 ms | 4.9 ms | **12.9 ms** |
| Recurrence, peak allocated | 273 MiB | 998 MiB | 3460 MiB |
| Chunked scan, peak allocated | 192 MiB | 356 MiB | **916 MiB** |

</div>

### Constant state vs a growing KV cache

The reason to reach for an SSM. Decoding one token costs the same whether 256 or 16384 tokens came before, and one fixed-size state replaces a cache that grows with context. The baseline is a causal-attention layer of the same width decoding against a preallocated KV cache through PyTorch SDPA.

<p align="center">
  <img src="assets/decode_scaling.png" width="600" alt="Decode latency and state size vs context length">
</p>

<div align="center">

| Context | 256 | 1k | 4k | 16k |
|---|---:|---:|---:|---:|
| Attention + KV cache | 0.08 ms | 0.21 ms | 0.78 ms | 3.09 ms |
| Mamba-3 recurrent state | 1.03 ms | 1.06 ms | 0.99 ms | **0.98 ms** |
| KV cache, per layer | 3.0 MiB | 12.0 MiB | 48.0 MiB | 192.0 MiB |
| SSM state, per layer | 0.78 MiB | 0.78 MiB | 0.78 MiB | **0.78 MiB** |

</div>

Latency crosses over near 5k tokens; memory is flat throughout, 247× smaller at 16k. Two caveats worth stating plainly: that flat line is the *eager* path, so a CUDA graph moves it further down, and for batched **prefill** the fused attention kernels stay ahead of a pure-PyTorch scan. The structural win is in streaming decode.

### Eager vs CUDA-graph decode

Decoding one token at a time, 3 layers, batch 64, `d_model=320`. At this size a step is almost pure kernel-launch overhead, which a graph replay removes.

<p align="center">
  <img src="assets/decode.png" width="600" alt="Eager vs CUDA-graph decode">
</p>

<div align="center">

| `d_state` | Eager | CUDA graph | Speedup | of 60 FPS |
|---:|---:|---:|---:|---:|
| 16 | 3.52 ms | **0.51 ms** | 6.9× | 3.1% |
| 32 | 3.54 ms | **0.55 ms** | 6.5× | 3.3% |
| 64 | 3.66 ms | **0.75 ms** | 4.9× | 4.5% |

</div>

### Training throughput

Autograd runs straight through the chunked scan — no custom backward, no recompute hooks. Per-token cost bottoms out near `L ≈ 512–1024`, where launch overhead is amortized but chunk intermediates still fit comfortably.

<p align="center">
  <img src="assets/training.png" width="600" alt="Forward + backward time and per-token cost">
</p>

### Reproducing

```bash
python test_mamba3.py      # numerical self-checks + benchmarks
python bench_figures.py    # re-measure and redraw every figure above
python arch_figure.py      # redraw the diagram at the top
```

`bench_figures.py --replot` redraws from `assets/results.json` without touching the GPU.

---

## API map

<div align="center">

| Official `mamba_ssm` | This file |
|---|---|
| `modules.mamba3.Mamba3` | `Mamba3` |
| `modules.mamba3.heavy_tail_activation` | `heavy_tail_activation` |
| `modules.block.Block` | `Block` |
| `models.mixer_seq_simple.create_block` | `create_block` |
| `models.mixer_seq_simple.MixerModel` | `MixerModel` |
| `modules.mlp.GatedMLP` | `GatedMLP` |
| `ops.triton.layernorm_gated.RMSNorm` | `RMSNorm` (`RMSNormGated`) |
| `ops.triton.layernorm_gated.rms_norm_ref` | `rms_norm_ref` |
| `ops.triton.mamba3.mamba3_siso_combined` | `mamba3_siso_combined` |
| `ops.tilelang.mamba3.mamba3_mimo` | `mamba3_mimo_combined` |
| `utils.generation.InferenceParams` | `InferenceParams` |
| `utils.generation.DecodingCGCache` | `DecodingCGCache` |
| `utils.generation.capture_graph` | `capture_graph` |
| `utils.generation.update_graph_cache` | `update_graph_cache` |

</div>

## Implementation notes

**Chunked parallel scan.** Unrolling the linear recurrence gives

```
h_t = exp(cs_t) · h_{-1} + Σ_{s≤t} exp(cs_t - cs_s) · u_s ,
cs_t = Σ_{k≤t} A_k · dt_k
```

Split into chunks of length `Q`: every `(i, j)` pair inside a chunk is one matmul, and only `L/Q` states move serially. Training uses this path; decoding uses the step-by-step recurrence.

**CUDA graph decoding.** Single-step decode is almost pure kernel-launch overhead. `capture_graph` records the whole call chain into one graph. States update in place with `copy_`, so a replay reads and writes the addresses fixed at capture time.

**No einops.** On this path Python itself is hot; `rearrange` pattern parsing measured +10.8% on eager single-step. The file uses `reshape` / `permute` / `einsum`.

### Mamba-3 vs Mamba-2

1. **Exponential-trapezoidal discretization**

   ```
   h_t = exp(A·dt_t)·h_{t-1} + dt_t·[(1-tr_t)·Bx_t + tr_t·(Bx_t + Bx_{t-1})/2]
   ```

   `tr_t` is a learnable sigmoid gate: `tr=0` is Euler/ZOH, `tr=1` is full trapezoidal integration. Equivalent to `u_t = α_t·Bx_t + β_t·Bx_{t-1}` with `α_t = dt_t·(1 - tr_t/2)` and `β_t = dt_t·tr_t/2`, which keeps the recurrence linear in `(X, B)`.

2. **Complex (rotary) state space.** RoPE on `B` / `C`, angles scaled by `dt` and accumulated over time, so a real-valued state can express periodic dependencies.

3. **MIMO.** A rank-`R` projection turns the state write into a matmul. Persistent state stays `(B, H, P, N)`; `R` appears only on the read/write paths.

```
B batch | L seqlen | H nheads | P headdim | N d_state
G num_bc_heads | R mimo_rank | S num_rope_angles
Q = C readout | K = B write | V = x values
```

## Citation

```bibtex
@article{lahoti2026mamba3,
  title   = {Mamba-3: Improved Sequence Modeling using State Space Principles},
  author  = {Lahoti, Aakash and Li, Kevin Y. and Chen, Berlin and Wang, Caitlin
             and Bick, Aviv and Kolter, J. Zico and Dao, Tri and Gu, Albert},
  journal = {arXiv preprint arXiv:2603.15569},
  year    = {2026}
}
```
