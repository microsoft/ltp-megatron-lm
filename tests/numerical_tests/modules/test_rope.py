import pytest
import torch

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from tests.numerical_tests.modules.test_module import TestModule

class TestRoPE(TestModule):
    """
    Test RoPE
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
            'num_attention_heads': 64,
            'kv_channels': 128,
        },
    ])
    @pytest.mark.parametrize('rope_kv', [
        {
            'rotary_percent': 1.0,
        },
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_rope(self, inputs_kv, config_kv, rope_kv, steps, request):
        config = TransformerConfig(**config_kv)

        if config.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        model = RotaryEmbedding(
            kv_channels=config.kv_channels,
            rotary_percent=rope_kv['rotary_percent'],
        )
        model = model.cuda()
        if config.bf16:
            model = Float16Module(config, model)

        for step in range(steps):
            query = torch.randn(
                (
                    inputs_kv['seq_length'],
                    inputs_kv['micro_batch_size'],
                    config.num_attention_heads,
                    config.kv_channels,
                ),
                dtype=dtype,
                requires_grad=True,
            )
            key = torch.randn(
                (
                    inputs_kv['seq_length'],
                    inputs_kv['micro_batch_size'],
                    config.num_attention_heads,
                    config.kv_channels,
                ),
                dtype=dtype,
                requires_grad=True,
            )
            rotary_pos_emb = model(inputs_kv['seq_length'])
            rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_output = apply_rotary_pos_emb(query.cuda(), q_pos_emb, config=config)
            key_output = apply_rotary_pos_emb(key.cuda(), k_pos_emb, config=config)
            loss = query_output.mean() + key_output.mean()
            loss.backward()

            self.save_output(
                [query, key],
                [query_output, key_output],
                [q_pos_emb, k_pos_emb],
                [query.grad, key.grad],
                step,
                request,
            )
