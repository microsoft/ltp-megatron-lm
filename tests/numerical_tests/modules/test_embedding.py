import pytest
import torch

from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from tests.numerical_tests.modules.test_module import TestModule


class TestLanguageModelEmbedding(TestModule):
    """
    Test LanguageModelEmbedding
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
        },
    ])
    @pytest.mark.parametrize('embedding_kv', [
        {
            'vocab_size': 200064,
            'position_embedding_type': 'rope',
        }
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_language_model_embedding(self, inputs_kv, config_kv, embedding_kv, steps, request):
        config = TransformerConfig(**config_kv)

        module_spec = ModuleSpec(
            module=LanguageModelEmbedding,
        )
        model = build_module(
            module_spec,
            config=config,
            vocab_size=embedding_kv['vocab_size'],
            max_sequence_length=inputs_kv['seq_length'],
            position_embedding_type=embedding_kv['position_embedding_type'],
        )
        model, optimizer = self.setup_model_and_optimizer(config, model)

        for step in range(steps):
            inputs = (
                torch.randint(
                    low=0,
                    high=embedding_kv['vocab_size'],
                    size=(inputs_kv['seq_length'], inputs_kv['micro_batch_size']),
                    dtype=torch.int64
                ).cuda(),
                None,
            )
            output = model(*inputs)
            loss = output.mean()
            loss.backward()

            self.save_output(
                [inputs[0]],
                [output],
                optimizer.get_parameters(),
                optimizer.get_main_grads_for_grad_norm(),
                step,
                request,
            )

            optimizer.step()
