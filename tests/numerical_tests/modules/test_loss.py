import pytest
import torch

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer import TransformerConfig
from tests.numerical_tests.modules.test_module import TestModule

class TestLanguageModuleComputeLanguageModelLoss(TestModule):
    """
    Test LanguageModule.compute_language_model_loss
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
            'num_attention_heads': 1,
        },
    ])
    @pytest.mark.parametrize('loss_kv', [
        {
            'vocab_size': 200064,
        }
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_vocab_parallel_cross_entropy(self, inputs_kv, config_kv, loss_kv, steps, request):
        config = TransformerConfig(**config_kv)

        if config.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        model = LanguageModule(config)

        for step in range(steps):
            logits = torch.randn(
                (
                    inputs_kv['seq_length'],
                    inputs_kv['micro_batch_size'],
                    loss_kv['vocab_size'],
                ),
                dtype=dtype,
                requires_grad=True,
            )
            labels = torch.randint(
                low=0,
                high=loss_kv['vocab_size'],
                size=(inputs_kv['micro_batch_size'], inputs_kv['seq_length']),
                dtype=torch.int64,
            )
            output = model.compute_language_model_loss(labels.cuda(), logits.cuda())
            loss = output.mean()
            loss.backward()

            self.save_output(
                [logits, labels],
                [output],
                [],
                [logits.grad],
                step,
                request,
            )
