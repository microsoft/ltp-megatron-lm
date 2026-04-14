# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Block Loop (Loop Full Transformer Block)."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _make_config(
    block_loop_iterations=1,
    block_loop_norm=True,
    block_loop_scaling="none",
    block_loop_embedding=False,
    num_experts=4,
    topk=2,
    hidden_size=12,
    moe_num_iterations=1,
):
    """Helper to create a TransformerConfig for block loop tests."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        use_cpu_initialization=True,
        moe_token_dispatcher_type="allgather",
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=topk,
        moe_aux_loss_coeff=0.01,
        moe_grouped_gemm=False,
        add_bias_linear=False,
        # Block loop settings
        block_loop_iterations=block_loop_iterations,
        block_loop_norm=block_loop_norm,
        block_loop_scaling=block_loop_scaling,
        block_loop_embedding=block_loop_embedding,
        # MoE iteration (should be 1 when block loop is active)
        moe_num_iterations=moe_num_iterations,
    )


def _build_layer(config):
    """Build a TransformerLayer from config."""
    spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    layer = TransformerLayer(config, spec.submodules)
    return layer


def _make_inputs(config, seq_length=8, micro_batch_size=2):
    """Create test inputs for a transformer layer."""
    hidden_states = torch.randn(
        seq_length, micro_batch_size, config.hidden_size, device="cuda"
    )
    attention_mask = torch.ones(
        1, 1, seq_length, seq_length, dtype=bool, device="cuda"
    )
    return hidden_states, attention_mask


# ============================================================
# Config Validation Tests (no GPU)
# ============================================================
class TestBlockLoopConfig:
    def test_invalid_iterations_zero(self):
        with pytest.raises(ValueError, match="block_loop_iterations must be >= 1"):
            _make_config(block_loop_iterations=0)

    def test_invalid_scaling(self):
        with pytest.raises(ValueError, match="block_loop_scaling"):
            _make_config(block_loop_iterations=2, block_loop_scaling="invalid")

    def test_mutual_exclusivity(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _make_config(block_loop_iterations=2, moe_num_iterations=2)

    def test_valid_config_n1(self):
        config = _make_config(block_loop_iterations=1)
        assert config.block_loop_iterations == 1

    def test_valid_config_n2(self):
        config = _make_config(block_loop_iterations=2)
        assert config.block_loop_iterations == 2


# ============================================================
# Constructor Tests (GPU)
# ============================================================
class TestBlockLoopInit:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_n1_no_block_loop_modules(self):
        config = _make_config(block_loop_iterations=1)
        layer = _build_layer(config)
        assert layer.block_loop_iterations == 1
        assert not hasattr(layer, 'block_loop_norm')
        assert not hasattr(layer, 'block_loop_embedding')
        assert not hasattr(layer, 'block_loop_gate')

    def test_n2_norm_created(self):
        config = _make_config(block_loop_iterations=2, block_loop_norm=True)
        layer = _build_layer(config)
        assert layer.block_loop_norm is not None

    def test_n2_no_norm(self):
        config = _make_config(block_loop_iterations=2, block_loop_norm=False)
        layer = _build_layer(config)
        assert layer.block_loop_norm is None

    def test_n2_embedding_shape(self):
        config = _make_config(block_loop_iterations=2, block_loop_embedding=True)
        layer = _build_layer(config)
        assert layer.block_loop_embedding is not None
        assert layer.block_loop_embedding.shape == (2, config.hidden_size)

    def test_n2_learned_gate_shape(self):
        config = _make_config(block_loop_iterations=2, block_loop_scaling="learned_gate")
        layer = _build_layer(config)
        assert layer.block_loop_gate is not None
        assert layer.block_loop_gate.shape == (2,)
        # Check initialization: 1/N
        torch.testing.assert_close(
            layer.block_loop_gate.data,
            torch.full((2,), 0.5),
        )

    def test_n3_embedding_shape(self):
        config = _make_config(block_loop_iterations=3, block_loop_embedding=True)
        layer = _build_layer(config)
        assert layer.block_loop_embedding.shape == (3, config.hidden_size)


# ============================================================
# Forward Tests (GPU)
# ============================================================
class TestBlockLoopForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_n1_output_shape(self):
        config = _make_config(block_loop_iterations=1)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output, context = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert output.shape == hidden_states.shape

    def test_n2_output_shape(self):
        config = _make_config(block_loop_iterations=2)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output, context = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert output.shape == hidden_states.shape

    def test_n2_different_from_n1(self):
        """N=2 should produce different output than N=1."""
        config_n1 = _make_config(block_loop_iterations=1)
        config_n2 = _make_config(block_loop_iterations=2)
        layer_n1 = _build_layer(config_n1).cuda()
        layer_n2 = _build_layer(config_n2).cuda()
        # Copy weights from n1 to n2 (shared attention + MoE weights)
        layer_n2.load_state_dict(layer_n1.state_dict(), strict=False)

        hidden_states, attention_mask = _make_inputs(config_n1)
        out_n1, _ = layer_n1(
            hidden_states=hidden_states.clone(), attention_mask=attention_mask
        )
        out_n2, _ = layer_n2(
            hidden_states=hidden_states.clone(), attention_mask=attention_mask
        )
        # Outputs should differ because n2 runs the layer twice
        assert not torch.allclose(out_n1, out_n2, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow to attention and MoE parameters."""
        config = _make_config(block_loop_iterations=2)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        hidden_states.requires_grad_(True)

        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        loss = output.sum()
        loss.backward()

        # Check gradient flow to input
        assert hidden_states.grad is not None
        assert hidden_states.grad.abs().sum() > 0

        # Check gradient flow to attention weights
        attn_params = list(layer.self_attention.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn_params)

        # Check gradient flow to MoE weights
        mlp_params = list(layer.mlp.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mlp_params)

    def test_gradient_flow_to_embedding(self):
        """Block loop embedding should receive gradients."""
        config = _make_config(block_loop_iterations=2, block_loop_embedding=True)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)

        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        loss = output.sum()
        loss.backward()

        assert layer.block_loop_embedding.grad is not None
        assert layer.block_loop_embedding.grad.abs().sum() > 0

    def test_gradient_flow_to_gate(self):
        """Learned gate should receive gradients."""
        config = _make_config(block_loop_iterations=2, block_loop_scaling="learned_gate")
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)

        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        loss = output.sum()
        loss.backward()

        assert layer.block_loop_gate.grad is not None
        assert layer.block_loop_gate.grad.abs().sum() > 0

    def test_n3_output_shape(self):
        config = _make_config(block_loop_iterations=3)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output, context = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert output.shape == hidden_states.shape

    @pytest.mark.parametrize("scaling", ["none", "uniform", "learned_gate"])
    def test_all_scaling_modes(self, scaling):
        """All scaling modes should produce valid output."""
        config = _make_config(block_loop_iterations=2, block_loop_scaling=scaling)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_with_embedding_and_norm(self):
        """Full config: loop + norm + embedding + learned_gate."""
        config = _make_config(
            block_loop_iterations=2,
            block_loop_norm=True,
            block_loop_scaling="learned_gate",
            block_loop_embedding=True,
        )
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        hidden_states.requires_grad_(True)

        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        loss = output.sum()
        loss.backward()

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert hidden_states.grad is not None
        assert layer.block_loop_embedding.grad is not None
        assert layer.block_loop_gate.grad is not None


# ============================================================
# Scaling Tests (GPU)
# ============================================================
class TestBlockLoopScaling:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_learned_gate_initial_values(self):
        config = _make_config(block_loop_iterations=3, block_loop_scaling="learned_gate")
        layer = _build_layer(config)
        expected = torch.full((3,), 1.0 / 3.0)
        torch.testing.assert_close(layer.block_loop_gate.data, expected)
