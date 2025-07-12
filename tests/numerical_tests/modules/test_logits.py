import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.transformer import TransformerConfig
from tests.numerical_tests.modules.test_module import TestModule


class TestLogits(TestModule):
    """
    Test Logits
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
        },
    ])
    @pytest.mark.parametrize('logits_kv', [
        {
            'vocab_size': 200064,
        }
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_logits(self, inputs_kv, config_kv, logits_kv, steps, request):
        config = TransformerConfig(**config_kv)

        if config.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        model = tensor_parallel.ColumnParallelLinear(
            config_kv['hidden_size'],
            logits_kv['vocab_size'],
            config=config,
            init_method=config.init_method,
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
