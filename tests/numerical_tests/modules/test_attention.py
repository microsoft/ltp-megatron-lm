import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from tests.numerical_tests.modules.test_module import TestModule


class TestSelfAttention(TestModule):
    """
    Test SelfAttention
    """

    @pytest.mark.parametrize('inputs_kv', [
        {
            'seq_length': 4096,
            'micro_batch_size': 1,
        },
    ])
    @pytest.mark.parametrize('config_kv', [
        {
            'bf16': True,
            'num_layers': 1,
            'hidden_size': 5120,
            'num_attention_heads': 64,
            'num_query_groups': 8,
            'kv_channels': 128,
            'qk_layernorm': True,
        },
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_self_attention(self, inputs_kv, config_kv, steps, request):
        config = TransformerConfig(**config_kv)

        if config.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        module_spec = ModuleSpec(
            module=SelfAttention,
            params={'attn_mask_type': AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TELayerNormColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=TENorm if config.qk_layernorm else IdentityOp,
                k_layernorm=TENorm if config.qk_layernorm else IdentityOp,
            ),
        )
        model = build_module(
            module_spec,
            config=config,
            layer_number=0,
        )
        model, optimizer = self.setup_model_and_optimizer(config, model)

        for step in range(steps):
            inputs = (
                torch.randn(
                    (
                        inputs_kv['seq_length'],
                        inputs_kv['micro_batch_size'],
                        config.hidden_size
                    ),
                    dtype=dtype
                ).cuda(),
                torch.triu(
                    torch.ones(
                        (
                            inputs_kv['seq_length'],
                            inputs_kv['seq_length']
                        ),
                        dtype=torch.bool
                    ), diagonal=1
                ).unsqueeze(0).unsqueeze(0).cuda(),
            )
            output, bias = model(*inputs)
            merged_output = output + bias
            loss = merged_output.mean()
            loss.backward()
            optimizer.step()

            self.save_output(
                [*inputs],
                [merged_output],
                optimizer.get_parameters(),
                optimizer.get_main_grads_for_grad_norm(),
                step,
                request,
            )
