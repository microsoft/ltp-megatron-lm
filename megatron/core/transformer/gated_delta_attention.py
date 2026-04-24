# Copyright (c) 2025. All rights reserved.
# Naive chunk GDR implementation adapted from FLA (MIT License):
#   https://github.com/fla-org/flash-linear-attention
#   Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""Gated Delta Rule (GDR) linear attention for block loop pass 2+.

Drop-in replacement for DotProductAttention: receives Q, K, V from
existing SelfAttention projections, returns context in the same format.

Uses a pure-PyTorch chunked delta rule implementation to avoid triton
compatibility issues on ROCm clusters. If FLA's triton kernel is
available, it is used for better performance.

FLOPs: O(n * d^2) per head vs O(n^2 * d) for softmax attention.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

# Try FLA triton kernel first (fastest); fall back to inlined pure-PyTorch.
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.modules.l2norm import l2norm

    HAVE_FLA_TRITON = True
except (ImportError, ValueError, Exception):
    HAVE_FLA_TRITON = False
    chunk_gated_delta_rule = None


def _l2norm(x: Tensor) -> Tensor:
    """L2-normalize along the last dimension."""
    return F.normalize(x, p=2, dim=-1)


def _chunk_loop_body(
    q_i, k_i, v_i, k_cumdecay_i, L_mask_i, decay_i, S, mask2,
):
    """One chunk iteration, factored out for gradient checkpointing."""
    attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask_i).masked_fill_(mask2, 0)
    v_prime = k_cumdecay_i @ S
    v_new = v_i - v_prime
    o_inter = (q_i * decay_i[:, :, :, None].exp()) @ S
    o_i = o_inter + attn_i @ v_new
    S_new = S * decay_i[:, :, -1, None, None].exp() + (
        k_i * (decay_i[:, :, -1, None] - decay_i).exp()[..., None]
    ).transpose(-1, -2) @ v_new
    return o_i, S_new


def _naive_chunk_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: Tensor = None,
    output_final_state: bool = False,
    use_checkpoint: bool = True,
):
    """Pure-PyTorch chunked gated delta rule (adapted from FLA, MIT license).

    Memory-optimized: stays in input dtype (bf16), only uses float32 for
    decay cumsum. Chunk loop optionally uses gradient checkpointing.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H]
        beta: [B, T, H]
        chunk_size: int
        scale: float, optional

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    BT = chunk_size
    dtype = q.dtype
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous(),
        [q, k, v, beta, g],
    )

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))

    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % BT == 0

    mask = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0
    )
    q, k, v, k_beta = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT),
        [q, k, v, k_beta],
    )
    # decay cumsum in float32 for numerical stability, then cast back
    decay = rearrange(g.unsqueeze(-1), 'b h (n c) d -> b h n c d', c=BT)
    decay = decay.squeeze(-1).float().cumsum(-1).to(dtype)
    decay_exp = decay.exp()[..., None]
    L_mask = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()
    ).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, BT):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(BT, dtype=dtype, device=q.device)
    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    v = k_cumsum

    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state.to(dtype)

    o = torch.zeros_like(v)
    mask2 = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1
    )
    for i in range(0, l // BT):
        if use_checkpoint:
            o_i, S = torch.utils.checkpoint.checkpoint(
                _chunk_loop_body,
                q[:, :, i], k[:, :, i], v[:, :, i],
                k_cumdecay[:, :, i], L_mask[:, :, i], decay[:, :, i],
                S, mask2,
                use_reentrant=False,
            )
        else:
            o_i, S = _chunk_loop_body(
                q[:, :, i], k[:, :, i], v[:, :, i],
                k_cumdecay[:, :, i], L_mask[:, :, i], decay[:, :, i],
                S, mask2,
            )
        o[:, :, i] = o_i

    if not output_final_state:
        S = None

    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


class GatedDeltaRuleAttention(MegatronModule):
    """Drop-in replacement for DotProductAttention using Gated Delta Rule.

    Receives Q, K, V from existing SelfAttention projections (with RoPE already
    applied). Uses FLA's chunk_gated_delta_rule kernel instead of softmax.

    Input:  query, key, value in [sq, b, np, hn] format
    Output: context in [sq, b, np*hn] format
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        A_init_range: tuple = (1, 16),
        **kwargs,
    ):
        super().__init__(config=config)

        self.use_checkpoint = getattr(config, 'block_loop_linear_checkpoint', True)

        self.layer_number = layer_number
        self.head_dim = config.kv_channels
        tp = max(1, getattr(config, 'tensor_model_parallel_size', 1))
        self.num_heads = config.num_attention_heads // tp
        # GQA: K/V may have fewer heads than Q
        nqg = getattr(config, 'num_query_groups', None) or config.num_attention_heads
        self.num_kv_heads = nqg // tp
        self.kv_repeat = self.num_heads // self.num_kv_heads

        # Learnable decay parameter A_log: per head
        A = torch.empty(self.num_heads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))

        # dt_bias: per head, controls gate strength
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # Beta projection: project from value to per-head write strength
        self.beta_proj = nn.Linear(self.head_dim, 1, bias=True)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 1.0)  # sigmoid(1) ≈ 0.73

        # Output group norm (per-head RMSNorm)
        self.out_norm = nn.RMSNorm(self.head_dim, eps=config.layernorm_epsilon)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        attn_mask_type=None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
    ) -> Tensor:
        """Forward pass using Gated Delta Rule linear attention.

        Args:
            query: [sq, b, np, hn]
            key:   [sq, b, np, hn]
            value: [sq, b, np, hn]

        Returns:
            context: [sq, b, np*hn]
        """
        # Transpose: [sq, b, np, hn] -> [b, sq, np, hn] for GDR
        q = query.permute(1, 0, 2, 3).contiguous()
        k = key.permute(1, 0, 2, 3).contiguous()
        v = value.permute(1, 0, 2, 3).contiguous()

        b, sq, np_q, hn = q.shape

        # GQA: expand K/V heads to match Q heads
        if self.kv_repeat > 1:
            # [b, sq, np_kv, hn] -> [b, sq, np_kv, repeat, hn] -> [b, sq, np_q, hn]
            k = k.unsqueeze(3).expand(-1, -1, -1, self.kv_repeat, -1).reshape(b, sq, np_q, hn)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.kv_repeat, -1).reshape(b, sq, np_q, hn)

        # L2 normalize Q, K (standard for delta rule to stabilize)
        if HAVE_FLA_TRITON:
            q = l2norm(q)
            k = l2norm(k)
        else:
            q = _l2norm(q)
            k = _l2norm(k)

        # Compute gate g: per-head decay
        g = -self.A_log.float().exp() * F.softplus(self.dt_bias.float())
        g = g.view(1, 1, np_q).expand(b, sq, np_q)

        # Compute beta: per-token, per-head write strength from value
        beta = self.beta_proj(v).squeeze(-1).sigmoid()

        # Core GDR computation: triton kernel if available, else pure-PyTorch
        if HAVE_FLA_TRITON:
            output, _ = chunk_gated_delta_rule(
                q.to(v.dtype),
                k.to(v.dtype),
                v,
                g=g.to(v.dtype),
                beta=beta.to(v.dtype),
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            output, _ = _naive_chunk_gated_delta_rule(
                q.to(v.dtype),
                k.to(v.dtype),
                v,
                g=g.to(v.dtype),
                beta=beta.to(v.dtype),
                chunk_size=64,
                use_checkpoint=self.use_checkpoint,
            )

        # Cast back to input dtype (naive impl computes in float32)
        input_dtype = query.dtype
        output = output.to(input_dtype)

        # Per-head RMSNorm on output
        output = self.out_norm(output)

        # Transpose back: [b, sq, np_q, hn] -> [sq, b, np_q*hn]
        output = output.permute(1, 0, 2, 3).contiguous()
        output = output.view(sq, b, np_q * hn)

        return output
