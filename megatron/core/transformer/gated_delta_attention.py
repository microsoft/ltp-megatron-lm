# Copyright (c) 2025. All rights reserved.

"""Gated Delta Rule (GDR) linear attention for block loop pass 2.

Minimal wrapper around FLA library's chunk_gated_delta_rule kernel.
Drop-in replacement for DotProductAttention: receives Q, K, V from
existing SelfAttention projections, returns context in the same format.

FLOPs: O(n * d^2) per head vs O(n^2 * d) for softmax attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.modules.l2norm import l2norm

    HAVE_FLA = True
    HAVE_FLA_TRITON = True
except (ImportError, ValueError):
    # ValueError: triton autotune incompatibility on ROCm/triton 3.1.0
    HAVE_FLA_TRITON = False
    try:
        from fla.ops.gated_delta_rule.naive import naive_chunk_gated_delta_rule

        chunk_gated_delta_rule = None
        HAVE_FLA = True
    except ImportError:
        chunk_gated_delta_rule = None
        naive_chunk_gated_delta_rule = None
        HAVE_FLA = False

# Pure-PyTorch l2norm fallback (avoid triton dependency)
try:
    from fla.modules.l2norm import l2norm
except (ImportError, ValueError):
    def l2norm(x):
        return F.normalize(x, p=2, dim=-1)


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

        assert HAVE_FLA, (
            "flash-linear-attention is required for GatedDeltaRuleAttention. "
            "Install with: pip install flash-linear-attention"
        )

        self.layer_number = layer_number
        self.head_dim = config.kv_channels
        self.num_heads = (
            config.num_attention_heads
            // max(1, getattr(config, 'tensor_model_parallel_size', 1))
        )

        # Learnable decay parameter A_log: per head
        # Initialized from log(uniform(A_init_range))
        A = torch.empty(self.num_heads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))

        # dt_bias: per head, controls gate strength
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # Beta projection: project from value to per-head write strength
        # Input: value [b, sq, np, hn] -> per-head scalar [b, sq, np]
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
        """
        Forward pass using Gated Delta Rule linear attention.

        Args:
            query: [sq, b, np, hn]
            key:   [sq, b, np, hn]
            value: [sq, b, np, hn]
            attention_mask: ignored (causal masking is built into GDR kernel)
            Others: ignored for compatibility

        Returns:
            context: [sq, b, np*hn]
        """
        # Transpose: [sq, b, np, hn] -> [b, sq, np, hn] for FLA
        q = query.permute(1, 0, 2, 3).contiguous()
        k = key.permute(1, 0, 2, 3).contiguous()
        v = value.permute(1, 0, 2, 3).contiguous()

        b, sq, np, hn = q.shape

        # L2 normalize Q, K (standard for delta rule to stabilize)
        q = l2norm(q)
        k = l2norm(k)

        # Compute gate g: per-head decay
        # g = -exp(A_log) * softplus(dt_bias), shape [np] -> [b, sq, np]
        g = -self.A_log.float().exp() * F.softplus(self.dt_bias.float())
        g = g.view(1, 1, np).expand(b, sq, np)

        # Compute beta: per-token, per-head write strength from value
        # [b, sq, np, hn] -> [b, sq, np, 1] -> [b, sq, np]
        beta = self.beta_proj(v).squeeze(-1).sigmoid()

        # Core GDR computation: triton kernel if available, else naive chunk
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
            output, _ = naive_chunk_gated_delta_rule(
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
