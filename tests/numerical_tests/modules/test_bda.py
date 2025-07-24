import pytest
import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import build_module
from tests.numerical_tests.modules.test_module import TestModule

class TestBDA(TestModule):
    """
    Test BDA (Bias-Dropout-Add)
    """

    @pytest.mark.parametrize('inputs_kv', [
        {
            'seq_length': 4096,
            'micro_batch_size': 1,
        },
    ])
    @pytest.mark.parametrize('config_kv', [
        {
            'use_cpu_initialization': True,
            'bf16': True,
            'num_layers': 1,
            'num_attention_heads': 64,
            'kv_channels': 128,
            'hidden_dropout': 0.0,
            'attention_dropout': 0.0,
        },
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_bda(self, inputs_kv, config_kv, steps, request):
        config = TransformerConfig(**config_kv)

        if config.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        model = build_module(get_bias_dropout_add)

        for step in range(steps):
            mlp_input = torch.randn(
                (
                    inputs_kv['seq_length'],
                    inputs_kv['micro_batch_size'],
                    config.num_attention_heads,
                    config.kv_channels,
                ),
                dtype=dtype,
                requires_grad=True,
            )
            mlp_bias = torch.randn(
                (
                    inputs_kv['seq_length'],
                    inputs_kv['micro_batch_size'],
                    config.num_attention_heads,
                    config.kv_channels,
                ),
                dtype=dtype,
                requires_grad=True,
            )
            residual = torch.randn(
                (
                    inputs_kv['seq_length'],
                    inputs_kv['micro_batch_size'],
                    config.num_attention_heads,
                    config.kv_channels,
                ),
                dtype=dtype,
                requires_grad=True,
            )
            output = model(training=True, fused=config.bias_dropout_fusion)(
                (mlp_input.cuda(), mlp_bias.cuda()), residual.cuda(), config.hidden_dropout
            )
            loss = output.mean()
            loss.backward()

            self.save_output(
                [mlp_input, mlp_bias, residual],
                [output],
                [],
                [mlp_input.grad, mlp_bias.grad, residual.grad],
                step,
                request,
            )
