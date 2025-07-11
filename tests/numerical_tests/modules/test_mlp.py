import pytest
import torch

from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from tests.numerical_tests.modules.test_module import TestModule


class TestMLP(TestModule):
    """
    Test MLP
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
            'num_attention_heads': 1,
            'add_bias_linear': False,
            'gated_linear_unit': torch.nn.functional.silu,
        },
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_mlp(self, inputs_kv, config_kv, steps, request):
        config = TransformerConfig(**config_kv)

        if config.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        module_spec = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        )

        model = build_module(
            module_spec,
            config=config,
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
            output, _ = model(*inputs)
            loss = output.mean()
            loss.backward()
            optimizer.step()

            self.save_output(
                [*inputs],
                [output],
                optimizer.get_parameters(),
                optimizer.get_main_grads_for_grad_norm(),
                step,
                request,
            )
