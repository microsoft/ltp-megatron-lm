import pytest
import torch

from types import SimpleNamespace
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.global_vars import set_args
from tests.numerical_tests.modules.test_module import TestModule


class TestTEDotProductAttention(TestModule):
    """
    Test TEDotProductAttention
    """

    @pytest.mark.parametrize('seq', [4096])
    @pytest.mark.parametrize('mbs', [1])
    @pytest.mark.parametrize('hidden_size', [5120])
    @pytest.mark.parametrize('num_heads', [64])
    @pytest.mark.parametrize('num_query_groups', [8])
    @pytest.mark.parametrize('kv_channels', [128])
    @pytest.mark.parametrize('dtype_str', ['bf16'])
    @pytest.mark.parametrize('lr', [1e-3])
    @pytest.mark.parametrize('steps', [10])
    def test_te_dot_product_attention(
        self, seq, mbs, hidden_size, num_heads, num_query_groups, kv_channels,
        dtype_str, lr, steps, request):
        if dtype_str == 'bf16':
            dtype = torch.bfloat16
        else:
            raise ValueError(f'Unsupported dtype {dtype_str}')

        args = SimpleNamespace()
        args.bf16 = (dtype == torch.bfloat16)
        set_args(args)
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_query_groups=num_query_groups,
            kv_channels=kv_channels,
            attention_dropout=0.0,
            num_layers=1,
            bf16=args.bf16,
        )

        model = TEDotProductAttention(
            config=config,
            layer_number=0,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
        )
        self.setup_model_and_optimizer(config, model, lr)

        inputs = (
            torch.randn((seq, mbs, num_heads, kv_channels), dtype=dtype).cuda(),
            torch.randn((seq, mbs, num_query_groups, kv_channels), dtype=dtype).cuda(),
            torch.randn((seq, mbs, num_query_groups, kv_channels), dtype=dtype).cuda(),
            torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0).cuda(),
            AttnMaskType.causal,
        )
        output = self.model(*inputs)

        self.save_output(
            [output],
            [],
            [],
            request
        )
