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
):
    """Pure-PyTorch chunked gated delta rule (adapted from FLA, MIT license).

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
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32),
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

    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % BT == 0

    mask = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0
    )
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    decay_exp = decay.exp()[..., None]
    L_mask = (
        (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()
    ).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, BT):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(BT, dtype=torch.float, device=q.device)
    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    v = k_cumsum

    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    mask2 = torch.triu(
        torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1
    )
    for i in range(0, l // BT):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(
            mask2, 0
        )
        v_prime = k_cumdecay[:, :, i] @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn_i @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (
            k_i
            * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
        ).transpose(-1, -2) @ v_new

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

        self.layer_number = layer_number
        self.head_dim = config.kv_channels
        self.num_heads = (
            config.num_attention_heads
            // max(1, getattr(config, 'tensor_model_parallel_size', 1))
        )

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

        b, sq, np, hn = q.shape

        # L2 normalize Q, K (standard for delta rule to stabilize)
        if HAVE_FLA_TRITON:
            q = l2norm(q)
            k = l2norm(k)
        else:
            q = _l2norm(q)
            k = _l2norm(k)

        # Compute gate g: per-head decay
        g = -self.A_log.float().exp() * F.softplus(self.dt_bias.float())
        g = g.view(1, 1, np).expand(b, sq, np)

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
                initial_state=None,
                output_final_state=False,
            )

        # Per-head RMSNorm on output
        output = self.out_norm(output)

        # Transpose back: [b, sq, np, hn] -> [sq, b, np*hn]
        output = output.permute(1, 0, 2, 3).contiguous()
        output = output.view(sq, b, np * hn)

        return output
