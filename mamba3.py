"""Single-file Mamba-3, with PyTorch as its only dependency.

Paper: https://arxiv.org/abs/2603.15569
Official: https://github.com/state-spaces/mamba

Names, arguments, state layouts and calling conventions follow the official
mamba_ssm package, so weights and documentation carry over. There is nothing
to build and no backend to select: the compute is ordinary PyTorch ops, so it
runs wherever PyTorch does. README.md covers the API map, the deviations from
upstream, and the implementation notes.

Self-check and benchmark: python scripts/test_mamba3.py
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    # Normalization and activation
    "heavy_tail_activation",
    "rms_norm_ref",
    "RMSNorm",
    "RMSNormGated",
    # Scans
    "mamba3_siso_combined",
    "mamba3_mimo_combined",
    # Modules
    "Mamba3",
    "GatedMLP",
    "Block",
    "create_block",
    "MixerModel",
    # Inference
    "InferenceParams",
    "initialize_states",
    "DecodingCGCache",
    "capture_graph",
    "update_graph_cache",
]


# =============================================================================
# 1. Normalization and activation
# =============================================================================


def heavy_tail_activation(x: Tensor) -> Tensor:
    """Heavy-tail activation for data-dependent A.

    Using this activation can improve stability during WSD training and at
    higher learning rates.

        f(x) = 1 + x        if x >= 0
             = 1 / (1 - x)  if x < 0

    The function is positive, continuous, and differentiable at x = 0.
    """
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def rms_norm_ref(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True
):
    """Equivalent of the official layernorm_gated.rms_norm_ref, without einops."""
    dtype = x.dtype
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = x.unflatten(-1, (-1, group_size))
        rstd = 1 / torch.sqrt(x_group.square().mean(dim=-1, keepdim=True) + eps)
        out = (x_group * rstd).flatten(-2) * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out = out * F.silu(z)
    return out.to(dtype)


class RMSNorm(nn.Module):
    """Counterpart of mamba_ssm.ops.triton.layernorm_gated.RMSNorm.

    If group_size is not None, we do GroupNorm with each group having group_size
    elements. group_size=None is equivalent to group_size=hidden_size (i.e. there's
    only 1 group).
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        return rms_norm_ref(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


# Mirrors `from ...layernorm_gated import RMSNorm as RMSNormGated` in official mamba3.py
RMSNormGated = RMSNorm


# =============================================================================
# 2. Selective scan
# =============================================================================
#
# Recurrence:
#     a_t = exp(A_t · dt_t)
#     u_t = alpha_t · Bx_t + beta_t · Bx_{t-1}
#     h_t = a_t · h_{t-1} + u_t
#     y_{t,r} = C_{t,r}ᵀ · h_t
# where Bx_t[p, n] = Σ_r V_{t,r}[p] · K_{t,r}[n]
#       alpha_t = dt_t·(1 - tr_t/2), beta_t = dt_t·tr_t/2

# Below this length the reshape overhead of chunking outweighs its benefit (the
# measured crossover is around L≈6), so we fall back to the step-by-step
# recurrence. Mirrors the official split between batched forward and single-step
# decode kernels.
_RECURRENT_MAX_LEN = 4


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor, rotate_pairwise: bool) -> Tensor:
    """Apply a 2-D rotation to pairs taken from the last dimension of x.

    x:        (..., 2*S)
    cos, sin: (..., S), broadcastable to x with its last dimension dropped
    rotate_pairwise: True  pairs (0,1), (2,3), ... and interleaves the output,
                           the official SISO convention
                     False pairs (i, i+S) and concatenates the two halves, the
                           official MIMO convention (the TileLang kernel permutes
                           the block rotation matrix so that entry i pairs with
                           entry i+N//2)

    The two pairings differ only by a permutation of the state dimension and are
    equally expressive; they are kept apart so official weights load correctly.
    cos/sin are passed in precomputed rather than the angles themselves: under
    MIMO, Q/K carry a rank axis while the rotation angle does not depend on rank,
    so the trigonometry is evaluated once over (B, L, H, S).
    """
    if rotate_pairwise:
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        )
        return rotated.flatten(-2)

    x_first, x_second = x.chunk(2, dim=-1)
    return torch.cat(
        [x_first * cos - x_second * sin, x_first * sin + x_second * cos], dim=-1
    )


def _recurrent_scan(V, K, Q, ADT, alpha, beta, ssm_state, v_state, k_state):
    """Step-by-step recurrence. Used for very short sequences, and as the
    numerical reference for the chunked implementation.

    V: (B, L, R, H, P)   K, Q: (B, L, R, H, N)   ADT, alpha, beta: (B, L, H)
    Returns (Y, ssm_state) with Y: (B, L, R, H, P).
    """
    batch, seqlen, rank, nheads, headdim = V.shape
    d_state = K.shape[-1]

    if ssm_state is None:
        ssm_state = V.new_zeros(batch, nheads, headdim, d_state)
    if v_state is None or k_state is None:
        kv_prev = V.new_zeros(batch, nheads, headdim, d_state)
    else:
        kv_prev = torch.einsum("brhp,brhn->bhpn", v_state, k_state)

    decay = torch.exp(ADT)
    ys = []
    for t in range(seqlen):
        kv = torch.einsum("brhp,brhn->bhpn", V[:, t], K[:, t])
        ssm_state = (
            decay[:, t, :, None, None] * ssm_state
            + alpha[:, t, :, None, None] * kv
            + beta[:, t, :, None, None] * kv_prev
        )
        ys.append(torch.einsum("brhn,bhpn->brhp", Q[:, t], ssm_state))
        kv_prev = kv

    return torch.stack(ys, dim=1), ssm_state


def _chunk_scan(V, K, Q, ADT, alpha, beta, ssm_state, v_state, k_state, chunk_size):
    """Chunked parallel scan. Built from matmuls and elementwise ops, fully
    differentiable, with the compute landing on cuBLAS.

    Since u_s = alpha_s·KV_s + beta_s·KV_{s-1} is linear in (V, K), the whole scan
    splits into the sum of two structurally identical paths: the current path
    (V, K, alpha) and the shifted path (V_shift, K_shift, beta). Within a chunk
    each path becomes a quadratic form with a decay mask, so one matmul covers
    every (i, j) pair; across chunks only the state is passed serially (L/Q steps,
    usually a single digit).
    """
    batch, seqlen, rank, nheads, headdim = V.shape
    d_state = K.shape[-1]
    dev, dtp = V.device, V.dtype

    # ---- Shifted path: move (V, K) of the previous step to the current one ----
    if v_state is None:
        v_state = V.new_zeros(batch, rank, nheads, headdim)
    if k_state is None:
        k_state = K.new_zeros(batch, rank, nheads, d_state)
    V_shift = torch.cat([v_state.unsqueeze(1), V[:, :-1]], dim=1)
    K_shift = torch.cat([k_state.unsqueeze(1), K[:, :-1]], dim=1)

    # ---- Pad up to a whole multiple of the chunk length ----
    # The chunk never exceeds the actual sequence length, so short sequences are
    # not padded into several times the work. Padded steps take A·dt = 0 (decay
    # of 1) with zero coefficients, leaving the state untouched: the end-of-chunk
    # state stays correct and the surplus outputs are simply truncated.
    chunk = max(1, min(chunk_size, seqlen))
    pad = (chunk - seqlen % chunk) % chunk
    if pad:
        wide = (0, 0, 0, 0, 0, 0, 0, pad)
        V, K, Q = F.pad(V, wide), F.pad(K, wide), F.pad(Q, wide)
        V_shift, K_shift = F.pad(V_shift, wide), F.pad(K_shift, wide)
        ADT = F.pad(ADT, (0, 0, 0, pad))
        alpha, beta = F.pad(alpha, (0, 0, 0, pad)), F.pad(beta, (0, 0, 0, pad))
    padded_len = seqlen + pad
    nchunks = padded_len // chunk

    # (B, L, R, H, X) -> (B, nchunks, H, chunk*R, X), flat index a = i*R + r
    def to_chunks(t: Tensor) -> Tensor:
        t = t.reshape(batch, nchunks, chunk, rank, nheads, t.shape[-1])
        return t.permute(0, 1, 4, 2, 3, 5).reshape(
            batch, nchunks, nheads, chunk * rank, -1
        )

    Vc, Vsc = to_chunks(V), to_chunks(V_shift)
    Kc, Ksc = to_chunks(K), to_chunks(K_shift)
    Qc = to_chunks(Q)

    # (B, L, H) -> (B, nchunks, H, chunk)
    def to_scalar_chunks(t: Tensor) -> Tensor:
        return t.reshape(batch, nchunks, chunk, nheads).permute(0, 1, 3, 2)

    ADTc = to_scalar_chunks(ADT)
    alphac, betac = to_scalar_chunks(alpha), to_scalar_chunks(beta)

    cl = ADTc.cumsum(dim=-1)                       # (B, nchunks, H, chunk), inclusive
    diff = cl.unsqueeze(-1) - cl.unsqueeze(-2)     # diff[i, j] = cl_i - cl_j
    causal = torch.ones(chunk, chunk, device=dev, dtype=torch.bool).tril()
    # For i >= j, diff <= 0 (A·dt is always negative) so exp cannot overflow;
    # i < j is masked out outright.
    decay_mat = torch.exp(diff.masked_fill(~causal, float("-inf")))

    def expand_rank(t: Tensor) -> Tensor:
        """(B, nchunks, H, chunk, chunk) -> (..., chunk*R, chunk*R), constant over R."""
        if rank == 1:
            return t
        return (
            t[..., :, None, :, None]
            .expand(batch, nchunks, nheads, chunk, rank, chunk, rank)
            .reshape(batch, nchunks, nheads, chunk * rank, chunk * rank)
        )

    # ---- Intra-chunk term: one matmul covers every (i, j) pair in the chunk ----
    mask_a = expand_rank(decay_mat * alphac.unsqueeze(-2))  # weighted by source j
    mask_b = expand_rank(decay_mat * betac.unsqueeze(-2))
    Y = (Qc @ Kc.transpose(-1, -2)) * mask_a @ Vc
    Y = Y + (Qc @ Ksc.transpose(-1, -2)) * mask_b @ Vsc

    # ---- Net contribution of each chunk to the state ----
    tail_decay = torch.exp(cl[..., -1:] - cl)               # (B, nchunks, H, chunk)

    def rank_weight(w: Tensor) -> Tensor:
        return w.repeat_interleave(rank, dim=-1).unsqueeze(-1)

    delta = (Vc * rank_weight(tail_decay * alphac)).transpose(-1, -2) @ Kc
    delta = delta + (Vsc * rank_weight(tail_decay * betac)).transpose(-1, -2) @ Ksc

    # ---- Inter-chunk: propagate serially over chunks (nchunks is small, so the
    # Python loop overhead is negligible) ----
    if ssm_state is None:
        ssm_state = torch.zeros(batch, nheads, headdim, d_state, device=dev, dtype=dtp)
    chunk_decay = torch.exp(cl[..., -1])                    # (B, nchunks, H)
    starts = []
    for n in range(nchunks):
        starts.append(ssm_state)
        ssm_state = chunk_decay[:, n, :, None, None] * ssm_state + delta[:, n]
    state_start = torch.stack(starts, dim=1)                # (B, nchunks, H, P, N)

    # ---- Inter-chunk term: the chunk's initial state, decayed, read out by Q ----
    Y = Y + (Qc @ state_start.transpose(-1, -2)) * rank_weight(torch.exp(cl))

    Y = Y.reshape(batch, nchunks, nheads, chunk, rank, headdim).permute(0, 1, 3, 4, 2, 5)
    return Y.reshape(batch, padded_len, rank, nheads, headdim)[:, :seqlen], ssm_state


def _mamba3_combined(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z,
    chunk_size, rotate_pairwise, Input_States, MIMO_V, MIMO_Z, MIMO_Out,
    outproj_norm_weight, outproj_norm_eps,
):
    """Shared body of the SISO and MIMO paths. See mamba3_siso_combined for the
    conventions.

    Q, K: (B, L, R, G, N)   V: (B, L, H, P)   ADT, DT, Trap: (B, H, L)
    Angles: (B, L, H, S)    Q_bias, K_bias: (H, R, N)
    """
    batch, seqlen, rank = V.shape[0], V.shape[1], Q.shape[2]
    nheads, headdim = V.shape[2], V.shape[3]
    d_state = K.shape[-1]

    # The state recurrence is precision sensitive and drifts noticeably in bf16,
    # so the scan runs in float32 internally.
    ADT = ADT.transpose(1, 2).float()               # (B, L, H)
    DT = DT.transpose(1, 2).float()
    trap = torch.sigmoid(Trap.transpose(1, 2).float())
    alpha = DT * (1.0 - 0.5 * trap)
    beta = DT * (0.5 * trap)

    angle_state, ssm_state, k_state, v_state = (
        Input_States if Input_States is not None else (None, None, None, None)
    )

    # ---- Expand to nheads, add bias, then rotate (order matches official) ----
    Q = Q.expand(batch, seqlen, rank, nheads, d_state).float() + Q_bias.permute(1, 0, 2)
    K = K.expand(batch, seqlen, rank, nheads, d_state).float() + K_bias.permute(1, 0, 2)

    # RoPE: angles are scaled by dt and accumulated over time, and can be resumed
    # across calls.
    angles = (Angles.float() * DT.unsqueeze(-1)).cumsum(dim=1)
    if angle_state is not None:
        angles = angles + angle_state.unsqueeze(1)
    nxt_angle_state = angles[:, -1]

    rot = 2 * Angles.shape[-1]
    cos, sin = torch.cos(angles).unsqueeze(2), torch.sin(angles).unsqueeze(2)
    Q = torch.cat(
        [_apply_rotary(Q[..., :rot], cos, sin, rotate_pairwise), Q[..., rot:]], dim=-1
    )
    K = torch.cat(
        [_apply_rotary(K[..., :rot], cos, sin, rotate_pairwise), K[..., rot:]], dim=-1
    )
    nxt_k_state = K[:, -1]
    nxt_v_state = V[:, -1]

    # ---- MIMO expands V / Z along rank; SISO just inserts a rank axis of 1 ----
    V = V.float()
    V_rank = (
        V.unsqueeze(2) * MIMO_V.permute(1, 0, 2).float()
        if MIMO_V is not None else V.unsqueeze(2)
    )
    v_state_rank = None
    if v_state is not None:
        v_state = v_state.float()
        v_state_rank = (
            v_state.unsqueeze(1) * MIMO_V.permute(1, 0, 2).float()
            if MIMO_V is not None else v_state.unsqueeze(1)
        )

    scan = _recurrent_scan if seqlen <= _RECURRENT_MAX_LEN else partial(
        _chunk_scan, chunk_size=chunk_size
    )
    Y, nxt_ssm_state = scan(
        V_rank, K, Q, ADT, alpha, beta,
        None if ssm_state is None else ssm_state.float(),
        v_state_rank,
        None if k_state is None else k_state.float(),
    )

    # ---- Skip term: broadcast the pre-expansion V across the rank axis ----
    # MIMO_Out starts at 1/R, so summing over rank recovers exactly D·V and stays
    # continuous with SISO behaviour. Using the expanded V_rank instead would
    # shrink the skip by an extra factor of 1/R at initialization.
    Y = Y + D.float().view(1, 1, 1, -1, 1) * V.unsqueeze(2)

    # ---- Output gate / normalization ----
    if Z is not None:
        Z = Z.float().unsqueeze(2)
        if MIMO_Z is not None:
            Z = Z * MIMO_Z.permute(1, 0, 2).float()
        if outproj_norm_weight is not None:
            Y = rms_norm_ref(
                Y.flatten(-2), outproj_norm_weight, None, z=Z.flatten(-2),
                eps=outproj_norm_eps, group_size=headdim, norm_before_gate=True,
            ).unflatten(-1, (nheads, headdim))
        else:
            Y = Y * F.silu(Z)

    if MIMO_Out is not None:
        Y = (Y * MIMO_Out.permute(1, 0, 2).float()).sum(dim=2)
    elif rank == 1:
        Y = Y.squeeze(2)

    return Y, nxt_angle_state, nxt_ssm_state, nxt_k_state, nxt_v_state


def _outer_kv(V_rank, K, rank):
    """Σ_r V[r] ⊗ K[r] → (B, H, P, N)."""
    if rank == 1:
        # A sum over a single term, so the outer product alone gives the same
        # numbers without the reshape and permute traffic einsum emits.
        return V_rank.squeeze(1).unsqueeze(-1) * K.squeeze(1).unsqueeze(-2)
    return torch.einsum("brhp,brhn->bhpn", V_rank, K)


def _mamba3_step(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z,
    rotate_pairwise, Input_States, MIMO_V, MIMO_Z, MIMO_Out,
    outproj_norm_weight, outproj_norm_eps,
):
    """Single-token specialization of _mamba3_combined, for decoding.

    _mamba3_combined is written for (B, L, ...) and stays correct at L = 1, but a
    decode step is dispatch bound, not compute bound: going through the general
    path costs ~350 aten calls for ~90 kernels, and most of the remainder is
    metadata churn on a length axis of one — a cumsum over a single element, a
    stack of a single tensor, and a long tail of unsqueeze/select/slice. This
    version drops the axis rather than degenerating it.

    Arguments are exactly what Mamba3._preprocess returns, with no length axis:

        Q, K:    (B, R, H, N)    V: (B, H, P)    ADT, DT, Trap: (B, H)
        Angles:  (B, H, S)       Q_bias, K_bias: (H, R, N)    D: (H,)
        Z:       (B, H, P) or None

    Returns (Y, angle_state, ssm_state, k_state, v_state), the L = 1 outputs of
    _mamba3_combined with the length axis squeezed out. Operations are kept in
    the same order and dtype, so results are bit-identical; test_mamba3.py
    asserts that with torch.equal.
    """
    batch, rank = Q.shape[0], Q.shape[1]
    nheads, headdim = V.shape[1], V.shape[2]
    d_state = K.shape[-1]

    ADT = ADT.float()
    DT = DT.float()
    trap = torch.sigmoid(Trap.float())
    half = 0.5 * trap
    alpha = DT * (1.0 - half)
    beta = DT * half

    angle_state, ssm_state, k_state, v_state = (
        Input_States if Input_States is not None else (None, None, None, None)
    )

    Q = Q.float() + Q_bias.permute(1, 0, 2)
    K = K.float() + K_bias.permute(1, 0, 2)

    # Over one token the running sum of dt-scaled angles is just this token's.
    angles = Angles.float() * DT.unsqueeze(-1)
    if angle_state is not None:
        angles = angles + angle_state
    nxt_angle_state = angles

    rot = 2 * Angles.shape[-1]
    cos, sin = torch.cos(angles).unsqueeze(1), torch.sin(angles).unsqueeze(1)
    if rot == d_state:
        # Nothing is left unrotated, so the concatenation would copy for nothing.
        Q = _apply_rotary(Q, cos, sin, rotate_pairwise)
        K = _apply_rotary(K, cos, sin, rotate_pairwise)
    else:
        Q = torch.cat(
            [_apply_rotary(Q[..., :rot], cos, sin, rotate_pairwise), Q[..., rot:]],
            dim=-1,
        )
        K = torch.cat(
            [_apply_rotary(K[..., :rot], cos, sin, rotate_pairwise), K[..., rot:]],
            dim=-1,
        )
    nxt_k_state = K
    nxt_v_state = V

    V = V.float()
    mimo_v = MIMO_V.permute(1, 0, 2).float() if MIMO_V is not None else None
    V_rank = V.unsqueeze(1) if mimo_v is None else V.unsqueeze(1) * mimo_v

    if ssm_state is None:
        ssm_state = V.new_zeros(batch, nheads, headdim, d_state)
    else:
        ssm_state = ssm_state.float()

    if v_state is None or k_state is None:
        kv_prev = V.new_zeros(batch, nheads, headdim, d_state)
    else:
        v_state = v_state.float()
        v_state_rank = (
            v_state.unsqueeze(1) if mimo_v is None else v_state.unsqueeze(1) * mimo_v
        )
        kv_prev = _outer_kv(v_state_rank, k_state.float(), rank)

    kv = _outer_kv(V_rank, K, rank)
    ssm_state = (
        torch.exp(ADT)[:, :, None, None] * ssm_state
        + alpha[:, :, None, None] * kv
        + beta[:, :, None, None] * kv_prev
    )
    Y = torch.einsum("brhn,bhpn->brhp", Q, ssm_state)

    Y = Y + D.float().view(1, 1, -1, 1) * V.unsqueeze(1)

    if Z is not None:
        Z = Z.float().unsqueeze(1)
        if MIMO_Z is not None:
            Z = Z * MIMO_Z.permute(1, 0, 2).float()
        if outproj_norm_weight is not None:
            Y = rms_norm_ref(
                Y.flatten(-2), outproj_norm_weight, None, z=Z.flatten(-2),
                eps=outproj_norm_eps, group_size=headdim, norm_before_gate=True,
            ).unflatten(-1, (nheads, headdim))
        else:
            Y = Y * F.silu(Z)

    if MIMO_Out is not None:
        Y = (Y * MIMO_Out.permute(1, 0, 2).float()).sum(dim=1)
    elif rank == 1:
        Y = Y.squeeze(1)

    return Y, nxt_angle_state, ssm_state, nxt_k_state, nxt_v_state


def mamba3_siso_combined(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z=None,
    chunk_size=64, Input_States=None, return_final_states=False, cu_seqlens=None,
):
    """Mamba-3 SISO combined scan. Signature matches the official
    mamba3_siso_combined.

        Q, K:      (B, L, G, N)   RMSNorm'd readout / write projections
        V:         (B, L, H, P)   values
        ADT, DT:   (B, H, L)      A·dt and dt
        Trap:      (B, H, L)      trapezoidal gate logit (sigmoid applied inside)
        Q_bias,
        K_bias:    (H, N)         added after the expand, before the rotation
        Angles:    (B, L, H, S)   angular rates, scaled by dt and accumulated
                                  over time internally
        D:         (H,)           skip coefficient
        Z:         (B, L, H, P)   output gate; when None the caller normalizes
        Input_States: optional 4-tuple (angle_dt_state, ssm_state, k_state,
                      v_state) for resuming across segments; same name as in the
                      official kernel
        cu_seqlens:   variable-length sequences, not implemented

    With return_final_states=False returns y: (B, L, H, P); otherwise returns
    (y, last_angle, last_state, last_k, last_v) where last_k is (B, H, N).
    """
    if cu_seqlens is not None:
        raise NotImplementedError("variable-length sequences (cu_seqlens)")

    y, angle, state, k, v = _mamba3_combined(
        Q.unsqueeze(2), K.unsqueeze(2), V, ADT, DT, Trap,
        Q_bias.unsqueeze(1), K_bias.unsqueeze(1), Angles, D, Z,
        chunk_size=chunk_size, rotate_pairwise=True, Input_States=Input_States,
        MIMO_V=None, MIMO_Z=None, MIMO_Out=None,
        outproj_norm_weight=None, outproj_norm_eps=1e-5,
    )
    if not return_final_states:
        return y
    return y, angle, state, k.squeeze(1), v


def mamba3_mimo_combined(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Angles, D,
    Z=None, chunk_size=16, rotary_dim_divisor=4, dtype=None, return_state=False,
    cu_seqlens=None, fuse_pregate_headwise_rms_norm=False,
    outproj_norm_weight=None, outproj_norm_eps=1e-5, Input_States=None,
):
    """Mamba-3 MIMO combined scan. Signature matches the official
    mamba3_mimo_combined.

    Differences from SISO: Q/K carry a rank axis (B, L, R, G, N), the biases are
    (H, R, N), MIMO_V / MIMO_Z / MIMO_Out are all (H, R, P), and the rotation
    pairs by halves. When MIMO_Out is None the rank axis is left unreduced and
    (B, L, R, H, P) is returned for the caller to handle.
    """
    if cu_seqlens is not None:
        raise NotImplementedError("variable-length sequences (cu_seqlens)")

    y, angle, state, k, v = _mamba3_combined(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z,
        chunk_size=chunk_size, rotate_pairwise=False, Input_States=Input_States,
        MIMO_V=MIMO_V, MIMO_Z=MIMO_Z, MIMO_Out=MIMO_Out,
        outproj_norm_weight=outproj_norm_weight if fuse_pregate_headwise_rms_norm else None,
        outproj_norm_eps=outproj_norm_eps,
    )
    if dtype is not None:
        y = y.to(dtype)
    if not return_state:
        return y
    return y, angle, state, k, v


# =============================================================================
# 3. Mamba3
# =============================================================================


class Mamba3(nn.Module):
    """Counterpart of mamba_ssm.modules.mamba3.Mamba3, argument for argument and
    attribute for attribute.

    Input and output are both (batch, seqlen, d_model), so it drops straight into
    a Transformer in place of self-attention.
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        # ----------------------------------------
        # Mamba-3 configs
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        fuse_pregate_headwise_norm=True,
        # -------------------------------------------
        # Fused kernel and sharding options
        chunk_size=64,  # Recommended: 64 for SISO, 64/mimo_rank for MIMO
        dropout=0.0,  # Just to absorb the kwarg
        layer_idx=None,  # Absorb kwarg for general module
        n_layer=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        self.fuse_pregate_headwise_norm = bool(
            fuse_pregate_headwise_norm and self.is_mimo and self.is_outproj_norm
        )
        if not self.is_mimo:
            self.mimo_rank = 1

        # These are bare asserts upstream; only the diagnostics are added here,
        # the conditions themselves are unchanged.
        assert not self.is_mimo or self.mimo_rank >= 1, (
            f"mimo_rank must be >= 1, got {mimo_rank}"
        )
        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0, (
            f"d_inner ({self.d_inner} = expand {self.expand} * d_model {self.d_model}) "
            f"must be divisible by headdim ({self.headdim})"
        )
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        assert self.num_bc_heads in (1, self.nheads), (
            f"ngroups must be either 1 or nheads ({self.nheads}), got {ngroups}"
        )

        # RoPE flags
        assert rope_fraction in [0.5, 1.0], (
            f"rope_fraction must be either 0.5 or 1.0, got {rope_fraction}"
        )
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0, (
            f"d_state ({d_state}) * rope_fraction ({rope_fraction}) is too small "
            f"to form rotation pairs"
        )

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        # dt_bias parameterization
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True

        # B and C biases
        self.B_bias = nn.Parameter(
            1 + torch.zeros(
                (self.nheads, self.mimo_rank, self.d_state),
                dtype=torch.float32, device=device,
            ),
            requires_grad=True,
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros(
                (self.nheads, self.mimo_rank, self.d_state),
                dtype=torch.float32, device=device,
            ),
            requires_grad=True,
        )

        # RMS Norm for B and C
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)

        if self.is_mimo:
            # Initialize up/down MIMO projection (for x and z)
            mimo_x_init_weights = (
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
                / self.mimo_rank
            )
            mimo_z_init_weights = torch.ones(
                self.nheads, self.mimo_rank, self.headdim, device=device
            )
            mimo_o_init_weights = (
                torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device)
                / self.mimo_rank
            )

            self.mimo_x = nn.Parameter(mimo_x_init_weights, requires_grad=True)
            self.mimo_z = nn.Parameter(mimo_z_init_weights, requires_grad=True)
            self.mimo_o = nn.Parameter(mimo_o_init_weights, requires_grad=True)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner,
                eps=1e-5,
                norm_before_gate=True,
                group_size=self.headdim,
                **factory_kwargs,
            )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    # -- Internals ----------------------------------------------------------

    def _split_in_proj(self, u):
        """Split the fused projection into [z, x, B, C, dd_dt, dd_A, trap, angle]."""
        return torch.split(
            self.in_proj(u),
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

    # -- Forward ------------------------------------------------------------

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim)
        Returns: same shape as u

        When inference_params is given, states are updated in place following the
        official convention: seqlen_offset == 0 runs the chunked forward
        (prefill), while > 0 with seqlen == 1 runs single-step decoding. With
        seqlen_offset > 0 and seqlen > 1 it still runs the chunked forward but
        resumes from the existing state, which is how segmented training works:
        states go back to the cache via copy_, so gradients are truncated between
        segments (TBPTT).
        """
        if seq_idx is not None:
            raise NotImplementedError("seq_idx")
        if cu_seqlens is not None:
            raise NotImplementedError("variable-length sequences (cu_seqlens)")
        batch, seqlen, dim = u.shape

        angle_dt_state, ssm_state, k_state, v_state = None, None, None, None
        if inference_params is not None:
            angle_dt_state, ssm_state, k_state, v_state = self._get_states_from_cache(
                inference_params, batch
            )
            if inference_params.seqlen_offset > 0 and seqlen == 1:
                out, _, _, _, _ = self.step(
                    u.squeeze(1), angle_dt_state, ssm_state, k_state, v_state
                )
                return out.unsqueeze(1)

        # Apply in_proj
        z, x, B, C, dd_dt, dd_A, trap, angles = self._split_in_proj(u)
        z = z.unflatten(-1, (self.nheads, self.headdim))
        x = x.unflatten(-1, (self.nheads, self.headdim))
        B = B.unflatten(-1, (self.mimo_rank, self.num_bc_heads, self.d_state))
        C = C.unflatten(-1, (self.mimo_rank, self.num_bc_heads, self.d_state))
        trap = trap.transpose(1, 2)

        # Compute ADT, DT
        _A = -heavy_tail_activation(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = _A * DT
        DT = DT.transpose(1, 2)
        ADT = ADT.transpose(1, 2)

        # Compute angle
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

        # Apply RMS Norm on B and C
        B = self.B_norm(B)
        C = self.C_norm(C)

        # Only resume from existing state after prefill; seqlen_offset == 0 starts
        # from zero.
        input_states = None
        if inference_params is not None and inference_params.seqlen_offset > 0:
            input_states = (angle_dt_state, ssm_state, k_state, v_state)

        # Apply Mamba-3 kernel
        if self.is_mimo:
            y = mamba3_mimo_combined(
                Q=C,
                K=B,
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z,
                MIMO_Out=self.mimo_o
                if (self.fuse_pregate_headwise_norm or not self.is_outproj_norm) else None,
                Angles=angles,
                D=self.D,
                Z=z if (self.fuse_pregate_headwise_norm or not self.is_outproj_norm) else None,
                chunk_size=self.chunk_size,
                rotary_dim_divisor=self.rotary_dim_divisor,
                dtype=x.dtype,
                return_state=ssm_state is not None,
                cu_seqlens=cu_seqlens,
                fuse_pregate_headwise_rms_norm=self.fuse_pregate_headwise_norm,
                outproj_norm_weight=self.norm.weight if self.fuse_pregate_headwise_norm else None,
                outproj_norm_eps=self.norm.eps if self.fuse_pregate_headwise_norm else 1e-5,
                Input_States=input_states,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k, last_v, *rest = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_state.copy_(last_k)
                v_state.copy_(last_v)
            if self.is_outproj_norm and not self.fuse_pregate_headwise_norm:
                z = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
                y = self.norm(y.flatten(-2).float(), z.flatten(-2))
                y = y.unflatten(-1, (self.nheads, self.headdim))
                y = torch.einsum("blrhp,hrp->blhp", y, self.mimo_o)
            y = y.flatten(-2)
        else:
            y = mamba3_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=self.C_bias.squeeze(1),
                K_bias=self.B_bias.squeeze(1),
                Angles=angles,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                chunk_size=self.chunk_size,
                Input_States=input_states,
                return_final_states=ssm_state is not None,
                cu_seqlens=cu_seqlens,
            )
            if ssm_state is not None:
                y, last_angle, last_state, last_k, last_v, *rest = y
                angle_dt_state.copy_(last_angle)
                ssm_state.copy_(last_state)
                k_state.copy_(last_k.unsqueeze(1))
                v_state.copy_(last_v)
            y = y.flatten(-2)
            if self.is_outproj_norm:
                y = self.norm(y, z.flatten(-2))

        out = self.out_proj(y.to(x.dtype))
        return out

    def _preprocess(self, A_proj, dd_dt, B, C, x, z, trap_proj, angle_proj):
        _A = -heavy_tail_activation(A_proj.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_proj)

        B = B.unflatten(-1, (self.mimo_rank, self.num_bc_heads, self.d_state))
        C = C.unflatten(-1, (self.mimo_rank, self.num_bc_heads, self.d_state))

        B = self.B_norm(B)
        C = self.C_norm(C)

        B = B.expand(-1, -1, self.nheads, -1)  # (B, R, H, N)
        C = C.expand(-1, -1, self.nheads, -1)  # (B, R, H, N)

        x = x.unflatten(-1, (self.nheads, self.headdim))
        z = z.unflatten(-1, (self.nheads, self.headdim))

        angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1)

        return DT, B, C, x, z, trap, _A, angles

    def step(self, u, angle_state, ssm_state, k_state, v_state, **kwargs):
        """
        Decode function. Also modify the state vars in-place for the next step.

        Args:
            u: (batch, d_model)
            angle_state: (batch, nheads, num_rope_angles)
            ssm_state: (batch, nheads, headdim, d_state)
            k_state: (batch, R, nheads, d_state), where R = mimo_rank (R=1 if not MIMO)
            v_state: (batch, nheads, headdim)
            **kwargs: ignored
        Returns:
            out: (batch, d_model)
            nxt_angle_state: (batch, nheads, num_rope_angles)
            state_out: (batch, nheads, headdim, d_state)
            nxt_k_state: (batch, R, nheads, d_state), where R = mimo_rank (R=1 if not MIMO)
            nxt_v_state: (batch, nheads, headdim)
        """
        z, x, B, C, dd_dt, dd_A, trap, angles = self._split_in_proj(u)
        DT, B, C, x, z, trap, A, angles = self._preprocess(
            dd_A, dd_dt, B, C, x, z, trap, angles
        )

        # _mamba3_step is the length-one specialization of the scan used by
        # forward, bit-identical to it and taking _preprocess's output shapes
        # directly. Bias and rotation are applied inside.
        common = dict(
            V=x,
            ADT=A * DT,
            DT=DT,
            Trap=torch.logit(trap),
            Angles=angles,
            D=self.D,
            Q_bias=self.C_bias,
            K_bias=self.B_bias,
            Input_States=(angle_state, ssm_state, k_state, v_state),
        )

        if self.is_mimo:
            gated_in_scan = self.fuse_pregate_headwise_norm or not self.is_outproj_norm
            y, nxt_angle_state, state_out, nxt_k_state, nxt_v_state = _mamba3_step(
                Q=C,
                K=B,
                Z=z if gated_in_scan else None,
                rotate_pairwise=False,
                MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z,
                MIMO_Out=self.mimo_o if gated_in_scan else None,
                outproj_norm_weight=(
                    self.norm.weight if self.fuse_pregate_headwise_norm else None
                ),
                outproj_norm_eps=(
                    self.norm.eps if self.fuse_pregate_headwise_norm else 1e-5
                ),
                **common,
            )
            if self.is_outproj_norm and not self.fuse_pregate_headwise_norm:
                y = self._postprocess(
                    y,
                    self.mimo_o.permute(1, 0, 2),
                    z,
                    self.mimo_z.permute(1, 0, 2),
                    self.headdim,
                )
        else:
            y, nxt_angle_state, state_out, nxt_k_state, nxt_v_state = _mamba3_step(
                Q=C,
                K=B,
                Z=z if not self.is_outproj_norm else None,
                rotate_pairwise=True,
                MIMO_V=None,
                MIMO_Z=None,
                MIMO_Out=None,
                outproj_norm_weight=None,
                outproj_norm_eps=1e-5,
                **common,
            )
            if self.is_outproj_norm:
                y = self.norm(y.flatten(-2), z.flatten(-2)).unflatten(
                    -1, (self.nheads, self.headdim)
                )

        # out_proj
        out = self.out_proj(y.flatten(-2).to(x.dtype))

        angle_state.copy_(nxt_angle_state)
        ssm_state.copy_(state_out)
        k_state.copy_(nxt_k_state)
        v_state.copy_(nxt_v_state)

        return out, nxt_angle_state, ssm_state, nxt_k_state, nxt_v_state

    def _postprocess(self, y, outpj, z, zpj, headdim):
        # y: (batch, R, H, D) — apply mimo_z to z, then norm, then mimo_o
        z_r = torch.einsum("bhp,rhp->brhp", z.float(), zpj)  # (batch, R, H, D)
        y = self.norm(y.flatten(-2).float(), z_r.flatten(-2))
        y = y.unflatten(-1, (-1, headdim))
        y = torch.einsum("brhp,rhp->bhp", y, outpj)  # (batch, H, D)
        return y

    def allocate_inference_cache(
        self, batch_size, max_seqlen, device=None, dtype=None, inplace_state=None, **kwargs
    ):
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype

        # RoPE State
        angle_dt_state = torch.zeros(
            (batch_size, self.nheads, self.num_rope_angles),
            device=device,
            dtype=torch.float32,
        )

        # Mamba-3 Combined Kernel States
        # SSM State
        ssm_state = torch.zeros(
            (batch_size, self.nheads, self.headdim, self.d_state),
            device=device,
            dtype=torch.float32,
        )

        # K (=B) State
        k_state = torch.zeros(
            (batch_size, self.mimo_rank, self.nheads, self.d_state),
            device=device,
            dtype=dtype,
        )

        # V (=x) State
        v_state = torch.zeros(
            (batch_size, self.nheads, self.headdim),
            device=device,
            dtype=dtype,
        )

        return (angle_dt_state, ssm_state, k_state, v_state)

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = (
                self.allocate_inference_cache(batch_size, inference_params.max_seqlen)
            )
        else:
            if initialize_states:
                for state in inference_params.key_value_memory_dict[self.layer_idx]:
                    state.zero_()
        return inference_params.key_value_memory_dict[self.layer_idx]

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, d_inner={self.d_inner}, "
            f"nheads={self.nheads}, headdim={self.headdim}, "
            f"num_bc_heads={self.num_bc_heads}, is_mimo={self.is_mimo}, "
            f"mimo_rank={self.mimo_rank}, num_rope_angles={self.num_rope_angles}, "
            f"is_outproj_norm={self.is_outproj_norm}, chunk_size={self.chunk_size}"
        )


# =============================================================================
# 4. GatedMLP / Block / create_block
# =============================================================================


class GatedMLP(nn.Module):
    """Counterpart of mamba_ssm.modules.mlp.GatedMLP."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


class Block(nn.Module):
    """Counterpart of mamba_ssm.modules.block.Block.

    Slightly different from a conventional pre-norm Transformer block: the order
    here is Add -> LN -> Mixer, and both the mixer output and the residual are
    returned, so every block except the first needs the previous block's residual
    passed in.

    Upstream, fused_add_norm=True routes through Triton's layer_norm_fn. There is
    no Triton here, so that branch uses a mathematically equivalent PyTorch
    implementation: same result, just without the operator fusion.
    """

    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm,
        fused_add_norm=False, residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None

    def _add_norm(self, hidden_states, residual, norm):
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = norm(residual.to(dtype=norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        return hidden_states, residual

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None,
        inference_params=None, **mixer_kwargs,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        hidden_states, residual = self._add_norm(hidden_states, residual, self.norm)
        hidden_states = self.mixer(
            hidden_states, inference_params=inference_params, **mixer_kwargs
        )

        if self.mlp is not None:
            hidden_states, residual = self._add_norm(hidden_states, residual, self.norm2)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """Counterpart of mamba_ssm.models.mixer_seq_simple.create_block.

    Upstream can also insert MHA layers according to attn_layer_idx; only Mamba3
    exists here, so that branch is dropped.
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    ssm_cfg = dict(ssm_cfg)
    ssm_layer = ssm_cfg.pop("layer", "Mamba3")
    if ssm_layer != "Mamba3":
        raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba3")
    mixer_cls = partial(Mamba3, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    """Counterpart of mamba_ssm.models.mixer_seq_simple._init_weights (the GPT-2
    residual rescaling).

    The deeper the model, the more branches accumulate on the residual stream and
    the variance grows linearly with depth; scaling each residual branch's output
    projection by 1/sqrt(N) pulls it back to a constant. Every Linear in this file
    has bias=False, so the zeros_ branch never fires; dt_bias / D / B_bias /
    C_bias are nn.Parameter rather than Linear.bias, so their own initialization
    is untouched.
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Re-init with kaiming then scale, rather than an in-place *=,
                # which would compound on repeated apply() calls.
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    """A stack of Blocks, counterpart of
    mamba_ssm.models.mixer_seq_simple.MixerModel.

    The only difference from upstream is the dropped embedding: that layer turns
    input_ids into hidden_states via a lookup, which only makes sense for language
    models, whereas this one takes (batch, seqlen, d_model) continuous features
    directly. Everything else matches: the residual handoff between Blocks, the
    trailing norm_f, and the {layer_idx: states} layout of
    allocate_inference_cache.

    This is exactly the contract capture_graph / update_graph_cache expect: a
    forward that accepts inference_params, plus an allocate_inference_cache
    method.
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int = 0,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.layers = nn.ModuleList([
            create_block(
                d_model,
                d_intermediate=d_intermediate,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(n_layer)
        ])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        residual = (hidden_states + residual) if residual is not None else hidden_states
        return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))


# =============================================================================
# 5. Inference parameters and state
# =============================================================================


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


def initialize_states(inference_params: InferenceParams, batch_mask: Optional[Tensor] = None):
    """Zero the cached states, matching the official
    _get_states_from_cache(initialize_states=True).

    batch_mask: (batch,) bool tensor, True marks a sample to be zeroed; None
                zeroes the whole batch. Per-sample zeroing is useful at clip
                boundaries and for state dropout.

    Writes are always in place, so an already captured graph stays valid.
    seqlen_offset is reset as well, sending the next forward back down the
    prefill path.
    """
    for states in inference_params.key_value_memory_dict.values():
        for state in states:
            if batch_mask is None:
                state.zero_()
            else:
                state[batch_mask] = 0
    if batch_mask is None:
        inference_params.seqlen_offset = 0


# =============================================================================
# 6. Accelerator graph decoding
# =============================================================================


def _new_graph(pool=None):
    """Create an accelerator graph, preferring the device-agnostic API.

    torch.accelerator.Graph is the backend-neutral interface. As of PyTorch 2.13
    only XPU registers an implementation, so constructing one on CUDA raises
    "Graph is not supported on device type: cuda"; torch.cuda.CUDAGraph covers
    that case with the same capture_begin / capture_end / replay / pool surface.
    Once CUDA registers its implementation (pytorch#171313) the neutral path is
    taken here with no other change.

    Returns (graph, begin), because the two spell the capture options
    differently: the neutral API takes them at construction, CUDA at
    capture_begin.
    """
    try:
        graph = torch.accelerator.Graph(pool=pool)
        return graph, graph.capture_begin
    except (AttributeError, RuntimeError):
        graph = torch.cuda.CUDAGraph()
        return graph, partial(
            graph.capture_begin, pool=pool, capture_error_mode="global"
        )


@dataclass
class DecodingCGCache:
    """Counterpart of mamba_ssm.utils.generation.DecodingCGCache."""

    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    # Captures sharing a pool must also share the stream they were captured on,
    # or the pool cannot reuse memory across them. torch.cuda.graph hid this by
    # using one class-level capture stream; with the neutral API the stream is
    # ours to keep.
    stream = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None


def _infer_d_model(model) -> int:
    for module in model.modules():
        if isinstance(module, Mamba3):
            return module.d_model
    raise ValueError("could not infer d_model from model; pass it explicitly")


def capture_graph(
    model, inference_params, batch_size, max_seqlen, decoding_seqlen=1,
    mempool=None, n_warmups=2, d_model=None, dtype=None, stream=None,
):
    """Record single-step decoding into an accelerator graph, matching the
    official generation.capture_graph.

    Graph capture appears exactly once in the official repo, and that one place
    is tied entirely to language models: the static buffers are input_ids /
    position_ids, forward is read for .logits, and the only public entry point is
    GenerationMixin.generate(cg=True). There is no non-LM version to follow, so
    the static buffer here becomes hidden_states of shape
    (batch, decoding_seqlen, d_model), and d_model / dtype are accepted as extra
    arguments (token ids carry no feature dimension, so upstream never needs
    them). Everything else follows the official code line by line, except that
    the runtime calls are the device-agnostic torch.accelerator ones rather than
    torch.cuda; see _new_graph for the one place a backend is still named.

    stream is the other addition, and it exists because of that same move.
    Captures that share a mempool only reuse memory if they also share the
    stream they were captured on, which torch.cuda.graph got for free from a
    class-level capture stream. Passing the same stream alongside the same
    mempool restores that; leaving it None captures on a fresh stream, which is
    what a lone graph wants anyway.

    States are updated in place with copy_ inside forward, per the official
    convention, so a replay reads and writes the very addresses fixed at capture
    time and nothing needs copying back. The warmup before capture runs on a side
    stream so that lazy initialization in cuBLAS and friends completes first;
    otherwise those one-off allocations would be recorded into the graph.

    As upstream, both capture and replay must happen under
    torch.inference_mode() (the official code marks the whole decode() that way).
    Outside it, the static buffers count as inference tensors and refuse in-place
    writes.
    """
    param_example = next(iter(model.parameters()))
    device = param_example.device
    dtype = param_example.dtype if dtype is None else dtype
    d_model = _infer_d_model(model) if d_model is None else d_model

    hidden_states = torch.zeros(
        batch_size, decoding_seqlen, d_model, device=device, dtype=dtype
    )
    seqlen_offset_og = inference_params.seqlen_offset
    inference_params.seqlen_offset = max_seqlen - decoding_seqlen
    if inference_params.lengths_per_sample is not None:
        inference_params.lengths_per_sample[:] = inference_params.seqlen_offset

    # Warmup before capture
    stream = torch.Stream(device=device) if stream is None else stream
    stream.wait_stream(torch.accelerator.current_stream(device))
    with stream:
        for _ in range(n_warmups):
            out = model(hidden_states, inference_params=inference_params)
        stream.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.accelerator.current_stream(device).wait_stream(stream)

    # Captures the graph. Free what we can first, as both backends' own capture
    # context managers do. The side stream is explicit because the neutral API
    # records whatever stream is current, and a graph cannot be captured on the
    # default one.
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    torch.accelerator.empty_host_cache()
    graph, begin = _new_graph(mempool)
    with stream:
        begin()
        out = model(hidden_states, inference_params=inference_params)
        graph.capture_end()

    def run(new_hidden_states, seqlen=None):
        if seqlen is not None and inference_params.lengths_per_sample is not None:
            inference_params.lengths_per_sample[:] = seqlen
        hidden_states.copy_(new_hidden_states)
        graph.replay()
        # The static output buffer is overwritten by the next replay, so hand
        # back a copy.
        return out.clone()

    # Exposed so update_graph_cache can share this graph's memory pool with the
    # next capture: the neutral API has no standalone pool handle to allocate up
    # front, only Graph.pool() once a graph exists.
    run.graph = graph

    inference_params.seqlen_offset = seqlen_offset_og
    return run


@torch.inference_mode()
def update_graph_cache(
    model, cache, batch_size, seqlen_og, max_seqlen,
    decoding_seqlens=(1,), dtype=None, n_warmups=2,
):
    """Build or reuse the accelerator graph cache, matching the official
    generation.update_graph_cache.

    Capture warms up on zero inputs, which dirties the states; they are zeroed
    before returning so the caller starts clean. From then on
    initialize_states(cache.inference_params) can reset them at any clip
    boundary.

    Usage (both cache.run and initialize_states must be called inside
    inference_mode)::

        with torch.inference_mode():
            cache = update_graph_cache(model, None, batch_size, 0, max_seqlen)
            for clip in stream:
                initialize_states(cache.inference_params)
                for frame in clip:            # frame: (batch, 1, d_model)
                    y = cache.run(frame)

    The batch handed to cache.run must match the captured one exactly; anything
    larger or smaller raises KeyError, because what the graph recorded is a fixed
    set of addresses and shapes. This matches upstream, whose dispatch also looks
    up (batch_size, decoding_seqlen) directly. For varying batch sizes there are
    two options: capture at the largest batch and pad the input up to it
    (discarding the surplus rows), or capture one graph per batch size actually
    used, trading memory for flexibility.
    """
    if cache is None:
        cache = DecodingCGCache()
    param_example = next(iter(model.parameters()))
    device = param_example.device
    if dtype is None:
        dtype = param_example.dtype
    if (
        (device, dtype) != (cache.device, cache.dtype)
        or batch_size > cache.max_batch_size
        or max_seqlen > cache.max_seqlen
    ):  # Invalidate the cache
        cache.callables = {}
        cache.mempool = None
        cache.stream = None
        cache.inference_params = None
        gc.collect()
        cache.device, cache.dtype = device, dtype
        cache.max_batch_size, cache.max_seqlen = batch_size, max_seqlen
        assert hasattr(model, "allocate_inference_cache"), (
            "Graph decoding requires that the model has a method allocate_inference_cache"
        )
        inf_cache = model.allocate_inference_cache(batch_size, max_seqlen, dtype)
        lengths_per_sample = torch.full(
            (batch_size,), seqlen_og, dtype=torch.int32, device=device
        )
        cache.inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            seqlen_offset=seqlen_og,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
    if cache.stream is None:
        cache.stream = torch.Stream(device=device)
    for decoding_seqlen in decoding_seqlens:
        if (batch_size, decoding_seqlen) not in cache.callables:
            run = capture_graph(
                model,
                cache.inference_params,
                batch_size,
                max_seqlen,
                decoding_seqlen=decoding_seqlen,
                mempool=cache.mempool,
                n_warmups=n_warmups,
                stream=cache.stream,
            )
            cache.callables[batch_size, decoding_seqlen] = run
            # The first graph opens the pool the rest capture into. Held on the
            # cache together with the stream, since sharing one without the
            # other defeats the reuse; see capture_graph.
            if cache.mempool is None:
                cache.mempool = run.graph.pool()

    def dispatch(hidden_states, seqlen=None):
        batch_size, decoding_seqlen = hidden_states.shape[:2]
        return cache.callables[batch_size, decoding_seqlen](hidden_states, seqlen)

    cache.run = dispatch
    initialize_states(cache.inference_params)
    cache.inference_params.seqlen_offset = 0  # Reset so it's not confusing
    return cache
