import pytest
import torch

from types import SimpleNamespace

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer import TransformerConfig
from megatron.training.global_vars import set_args
from tests.numerical_tests.modules.test_module import TestModule


class TestTELinear(TestModule):
    """
    Test TELinear
    """

    @pytest.mark.parametrize('seq', [4096])
    @pytest.mark.parametrize('mbs', [1])
    @pytest.mark.parametrize('input_output_size', [[5120, 2160]])
    @pytest.mark.parametrize('parallel_mode', ['duplicated'])
    @pytest.mark.parametrize('dtype_str', ['bf16'])
    @pytest.mark.parametrize('lr', [1e-3])
    @pytest.mark.parametrize('steps', [10])
    def test_te_linear(self, seq, mbs, input_output_size, parallel_mode, dtype_str, lr, steps, request):
        if dtype_str == 'bf16':
            dtype = torch.bfloat16
        else:
            raise ValueError(f'Unsupported dtype {dtype_str}')

        args = SimpleNamespace()
        args.bf16 = (dtype == torch.bfloat16)
        set_args(args)
        config = TransformerConfig(
            num_attention_heads=1,
            num_layers=1,
            bf16=args.bf16,
        )

        input_size, output_size = input_output_size
        model = TELinear(
            input_size=input_size,
            output_size=output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
        )
        self.setup_model_and_optimizer(config, model, lr)

        for step in range(steps):
            inputs = (
                torch.randn((seq, mbs, input_size), dtype=dtype).cuda(),
            )
            output, _ = self.model(*inputs)
            loss = output.mean()
            loss.backward()
            self.optimizer.step()

        self.save_output(
            [output],
            self.optimizer.get_parameters(),
            self.optimizer.get_main_grads_for_grad_norm(),
            request
        )
