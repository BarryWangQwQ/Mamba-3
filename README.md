<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-only-ee4c2c?style=flat-square" alt="PyTorch only">
  <img src="https://img.shields.io/badge/CUDA-no%20nvcc-0f7b6c?style=flat-square" alt="No nvcc">
  <img src="https://img.shields.io/badge/API-mamba__ssm-1e3a5f?style=flat-square" alt="Official API">
  <img src="https://img.shields.io/badge/paper-arXiv%3A2603.15569-b31b1b?style=flat-square" alt="Paper">
</p>

<h1 align="center">Mamba3</h1>

<p align="center">
  Single-file <a href="https://arxiv.org/abs/2603.15569">Mamba-3</a>.
  Official <a href="https://github.com/state-spaces/mamba"><code>mamba_ssm</code></a> API.
  PyTorch is the only dependency.
</p>

Drop [`mamba3.py`](mamba3.py) into a project. No nvcc, CUDA Toolkit, MSVC, Triton, TileLang, einops or `mamba-ssm`. A CUDA-enabled PyTorch build is enough for GPU acceleration.

This file replaces the official **kernel backend**, not the interface. Names, arguments, attributes, state layouts and calling conventions match `mamba_ssm`, so the official docs apply as-is and weights are interchangeable.

<table>
  <tr>
    <td align="center" width="33%"><strong>69.8×</strong><br><sub>SISO scan at L=512 vs recurrence</sub></td>
    <td align="center" width="33%"><strong>0.50 ms</strong><br><sub>CUDA-graph decode, 67 joints, d_state=16</sub></td>
    <td align="center" width="33%"><strong>≤ 10⁻⁶</strong><br><sub>max |Δ| vs a full forward, six layer configs</sub></td>
  </tr>
</table>

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

This file replaces the official kernel backend. Official fused kernels (Triton SISO / TileLang MIMO) are **not** compared: they need `mamba-ssm` plus nvcc / Triton / TileLang, which this file is written to avoid.

What *is* checked, on an RTX 4090 / PyTorch 2.12.1+cu132, TF32 off:

| | Compared against | max \|Δ\| |
|---|---|---:|
| Chunked scan, R = 1 / 2 / 4 | Step-by-step recurrence (Y, final state) | ≤ 5.7×10⁻⁶ |
| Chunked-scan gradients | Same recurrence, d(Y,h)/d(V,K,Q,ADT,α,β) | ≤ 1.8×10⁻⁴ &nbsp;·&nbsp; rel. ~2×10⁻⁷ |
| Segmented `inference_params` | One full forward | ≤ 7.2×10⁻⁷ |
| Official `step()` path | One full forward | ≤ 1.3×10⁻⁶ |
| CUDA graph, then after reset | Eager decode | 4.8×10⁻⁷ &nbsp;/&nbsp; 7.5×10⁻⁹ |
| State shapes, B/C bias, `in_proj`, RoPE, `Block`, `rms_norm_ref` | `mamba_ssm` conventions | match |

Two deviations are intentional — the official counterparts are language-model only:

- **`MixerModel` drops the embedding.** This stack takes `(batch, seqlen, d_model)` features directly.
- **`capture_graph` uses `hidden_states`.** Official static buffers are `input_ids` / `position_ids`.

Everything else follows the official code line by line. Not implemented: `cu_seqlens`, `seq_idx`, tensor parallelism, kernel-level fusion.

<p align="center">
  <img src="assets/alignment.png" width="920" alt="Numerical agreement with a full forward">
</p>

---

## Benchmarks

Chunked parallel scan vs the step-by-step recurrence. Recurrent time grows linearly with `L`; chunked time stays around 1 ms because the inner work is cuBLAS GEMMs.

<p align="center">
  <img src="assets/scan.png" width="920" alt="Selective scan: time and speedup">
</p>

| | L=32 | L=64 | L=128 | L=256 | L=512 |
|---|---:|---:|---:|---:|---:|
| SISO chunked | 0.66 ms | 0.67 ms | 0.69 ms | 0.74 ms | 0.81 ms |
| SISO vs recurrence | 5.5× | 10.8× | 20.8× | 38.7× | **69.8×** |
| MIMO(R=2) chunked | 0.68 ms | 0.72 ms | 0.76 ms | 0.84 ms | 1.04 ms |
| MIMO(R=2) vs recurrence | 6.4× | 12.0× | 22.5× | 40.6× | **65.8×** |

Single-step decode in the streaming-pose setting: 3 layers, 67 joints, `d_model=320`. CUDA graph removes kernel-launch overhead. Both stay far below a 16.7 ms / 60 FPS frame.

<p align="center">
  <img src="assets/decode.png" width="780" alt="Eager vs CUDA-graph decode">
</p>

| `d_state` | Eager | CUDA graph | Speedup | of 60 FPS |
|---:|---:|---:|---:|---:|
| 16 | 3.31 ms | **0.50 ms** | 6.6× | 3.0% |
| 32 | 3.41 ms | **0.55 ms** | 6.2× | 3.3% |
| 64 | 3.48 ms | **0.76 ms** | 4.6× | 4.5% |

```bash
python test_mamba3.py
```

Re-runs the numerical checks and prints the same class of benchmarks if CUDA is available.

---

## API map

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
