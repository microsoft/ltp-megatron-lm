import pytest
import torch

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from tests.numerical_tests.modules.test_module import TestModule


class TestLayerNorm(TestModule):
    """
    Test LayerNorm
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
            'hidden_size': 5120,
            'num_attention_heads': 1,
            'normalization': 'RMSNorm',
        },
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_layer_norm(self, inputs_kv, config_kv, steps, request):
        config = TransformerConfig(**config_kv)

        if config.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        module_spec = ModuleSpec(module=TENorm)

        model = build_module(
            module_spec,
            config=config,
            hidden_size=config_kv['hidden_size'],
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
            )
            output = model(*inputs)
            loss = output.mean()
            loss.backward()

            self.save_output(
                [*inputs],
                [output],
                optimizer.get_parameters(),
                optimizer.get_main_grads_for_grad_norm(),
                step,
                request,
            )

            optimizer.step()
