# Mamba3

Single-file [Mamba-3](https://arxiv.org/abs/2603.15569) with the official [`mamba_ssm`](https://github.com/state-spaces/mamba) API. PyTorch is the only dependency.

No nvcc, CUDA Toolkit, MSVC, Triton, TileLang, einops or `mamba-ssm`. A CUDA-enabled PyTorch build is enough for GPU acceleration.

This file replaces the official **kernel backend**, not the interface. Class names, arguments, attributes, state layouts and calling conventions match `mamba_ssm`, so the official docs apply as-is and weights are interchangeable.

## Install

```bash
pip install torch
```

Drop [`mamba3.py`](mamba3.py) into your project and import it.

```python
import torch
from mamba3 import Mamba3, MixerModel, InferenceParams, update_graph_cache

# One layer, drop-in replacement for self-attention: (B, L, D) -> (B, L, D)
layer = Mamba3(d_model=256, d_state=64, headdim=64).cuda()
y = layer(torch.randn(2, 128, 256, device="cuda"))

# A stack of layers for continuous features (no token embedding)
model = MixerModel(256, n_layer=4, rms_norm=True,
                   ssm_cfg=dict(d_state=64, headdim=64)).cuda().eval()
```

SISO is the default. MIMO and the Mamba-3 extras use the same argument names as upstream:

```python
Mamba3(
    d_model=256,
    d_state=64,
    headdim=64,
    is_mimo=True,
    mimo_rank=4,
    chunk_size=16,          # 64 for SISO, 64 / mimo_rank for MIMO
    is_outproj_norm=True,
    rope_fraction=0.5,      # 0.5 or 1.0
)
```

## Prefill, decode, CUDA graph

`inference_params` follows the official convention: `seqlen_offset == 0` is the chunked prefill; `seqlen_offset > 0` and `seqlen == 1` is single-step decode. Segmented forwards resume from the cached state.

```python
params = InferenceParams(max_seqlen=1024, max_batch_size=2)
y = model(x, inference_params=params)   # prefill
params.seqlen_offset += x.shape[1]

# Single-step decode (eager)
token = torch.randn(2, 1, 256, device="cuda")
y_t = model(token, inference_params=params)
params.seqlen_offset += 1

# Single-step decode (CUDA graph): one launch for the whole stack
cache = update_graph_cache(model, None, batch_size=2, seqlen_og=0, max_seqlen=1024)
y_t = cache.run(token)
```

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

## Deviations

Two changes, both because the official counterparts are language-model only and have no non-LM version to follow:

- **`MixerModel` drops the embedding.** Official code maps `input_ids` through a lookup. This stack takes `(batch, seqlen, d_model)` features directly.
- **`capture_graph` uses `hidden_states`.** Official static buffers are `input_ids` / `position_ids`. Here they are `(batch, decoding_seqlen, d_model)`.

Everything else follows the official code line by line.

Not implemented: variable-length sequences (`cu_seqlens`), `seq_idx`, tensor parallelism, kernel-level operator fusion.

## Implementation notes

**Chunked parallel scan.** Unrolling the linear recurrence gives

```
h_t = exp(cs_t) · h_{-1} + Σ_{s≤t} exp(cs_t - cs_s) · u_s,
cs_t = Σ_{k≤t} A_k · dt_k
```

Split into chunks of length `Q`: every `(i, j)` pair inside a chunk is one matmul, and only `L/Q` states move serially across chunks. The bulk of the work is cuBLAS GEMMs and stays differentiable end to end. Training uses this path; decoding uses the step-by-step recurrence.

**CUDA graph decoding.** Single-step decode is almost pure kernel-launch overhead. `capture_graph` records the whole call chain into one graph that replays with a single submission. States are updated in place with `copy_`, so a replay reads and writes the addresses fixed at capture time.

**No einops.** On this path Python itself is hot, and `rearrange` pattern parsing showed up as about +10.8% on eager single-step. The file uses `reshape` / `permute` / `einsum` throughout.

## Mamba-3 vs Mamba-2

1. **Exponential-trapezoidal discretization**

   ```
   h_t = exp(A·dt_t)·h_{t-1} + dt_t·[(1-tr_t)·Bx_t + tr_t·(Bx_t + Bx_{t-1})/2]
   ```

   `tr_t` is a learnable sigmoid gate: `tr=0` is Euler/ZOH, `tr=1` is full trapezoidal integration. Equivalent to `u_t = alpha_t·Bx_t + beta_t·Bx_{t-1}` with `alpha_t = dt_t·(1 - tr_t/2)` and `beta_t = dt_t·tr_t/2`, which keeps the recurrence linear in `(X, B)` and is what makes the parallel rewrite above possible.

2. **Complex (rotary) state space.** RoPE is applied to `B` / `C` with rotation angles scaled by `dt` and accumulated over time, so a real-valued state can express periodic dependencies such as counting and parity.

3. **MIMO.** A rank-`R` projection turns the state write from an outer product into a matrix multiply, raising arithmetic intensity. The persistent state is always `(B, H, P, N)`; `R` appears only on the read/write paths, never in `ssm_state`.

Tensor notation (same as the official kernels):

```
B batch | L seqlen | H nheads | P headdim | N d_state
G num_bc_heads (ngroups) | R mimo_rank | S num_rope_angles
Q = C readout | K = B write | V = x values
```

## Tests

```bash
python test_mamba3.py
```

The script checks that the chunked scan matches the recurrence in both values and gradients, that segmented and stepwise decode reproduce a full forward, that state / bias / norm layouts match `mamba_ssm`, and that CUDA-graph decode matches eager decode. It then prints a small benchmark if CUDA is available.

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
