# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Block Loop (Full-Model Loop of Transformer Block)."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _make_config(
    num_layers=2,
    block_loop_iterations=1,
    block_loop_norm=True,
    block_loop_scaling="none",
    block_loop_embedding="none",
    num_experts=4,
    topk=2,
    hidden_size=12,
    moe_num_iterations=1,
):
    """Helper to create a TransformerConfig for block loop tests."""
    return TransformerConfig(
        num_layers=num_layers,
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


def _build_block(config):
    """Build a TransformerBlock from config."""
    spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    block = TransformerBlock(config, spec)
    return block


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

    def test_invalid_embedding(self):
        with pytest.raises(ValueError, match="block_loop_embedding"):
            _make_config(block_loop_iterations=2, block_loop_embedding="invalid")

    def test_mutual_exclusivity(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _make_config(block_loop_iterations=2, moe_num_iterations=2)

    def test_valid_config_n1(self):
        config = _make_config(block_loop_iterations=1)
        assert config.block_loop_iterations == 1

    def test_valid_config_n2(self):
        config = _make_config(block_loop_iterations=2)
        assert config.block_loop_iterations == 2

    @pytest.mark.parametrize("embedding", ["none", "per_layer", "global"])
    def test_valid_embedding_choices(self, embedding):
        config = _make_config(block_loop_iterations=2, block_loop_embedding=embedding)
        assert config.block_loop_embedding == embedding


# ============================================================
# Constructor Tests (GPU) - TransformerLayer
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

    def test_n2_per_layer_embedding_shape(self):
        config = _make_config(block_loop_iterations=2, block_loop_embedding="per_layer")
        layer = _build_layer(config)
        assert layer.block_loop_embedding is not None
        assert layer.block_loop_embedding.shape == (2, config.hidden_size)

    def test_n2_global_embedding_no_layer_param(self):
        config = _make_config(block_loop_iterations=2, block_loop_embedding="global")
        layer = _build_layer(config)
        assert layer.block_loop_embedding is None

    def test_n2_no_embedding(self):
        config = _make_config(block_loop_iterations=2, block_loop_embedding="none")
        layer = _build_layer(config)
        assert layer.block_loop_embedding is None

    def test_n2_learned_gate_shape(self):
        config = _make_config(block_loop_iterations=2, block_loop_scaling="learned_gate")
        layer = _build_layer(config)
        assert layer.block_loop_gate is not None
        assert layer.block_loop_gate.shape == (2,)
        torch.testing.assert_close(
            layer.block_loop_gate.data,
            torch.full((2,), 0.5),
        )

    def test_n3_per_layer_embedding_shape(self):
        config = _make_config(block_loop_iterations=3, block_loop_embedding="per_layer")
        layer = _build_layer(config)
        assert layer.block_loop_embedding.shape == (3, config.hidden_size)


# ============================================================
# Constructor Tests (GPU) - TransformerBlock
# ============================================================
class TestBlockLoopBlockInit:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_global_embedding_created(self):
        config = _make_config(block_loop_iterations=2, block_loop_embedding="global")
        block = _build_block(config)
        assert block.block_loop_global_embedding is not None
        assert block.block_loop_global_embedding.shape == (2, config.hidden_size)

    def test_no_global_embedding_when_per_layer(self):
        config = _make_config(block_loop_iterations=2, block_loop_embedding="per_layer")
        block = _build_block(config)
        assert block.block_loop_global_embedding is None

    def test_no_global_embedding_when_none(self):
        config = _make_config(block_loop_iterations=2, block_loop_embedding="none")
        block = _build_block(config)
        assert block.block_loop_global_embedding is None

    def test_no_global_embedding_when_n1(self):
        config = _make_config(block_loop_iterations=1, block_loop_embedding="global")
        block = _build_block(config)
        assert block.block_loop_global_embedding is None


# ============================================================
# Forward Tests - TransformerLayer (GPU)
# ============================================================
class TestBlockLoopLayerForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_standard_path_without_block_loop_iteration(self):
        """When block_loop_iteration is None, standard path is used."""
        config = _make_config(num_layers=1, block_loop_iterations=2)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        # Call without block_loop_iteration — should use standard path
        output, context = layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert output.shape == hidden_states.shape

    def test_with_block_loop_iteration_0(self):
        config = _make_config(num_layers=1, block_loop_iterations=2)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output, context = layer(
            hidden_states=hidden_states, attention_mask=attention_mask,
            block_loop_iteration=0,
        )
        assert output.shape == hidden_states.shape

    def test_iteration_0_vs_1_differ(self):
        """Same layer with different iteration indices should produce different outputs
        when per_layer embedding is enabled."""
        config = _make_config(
            num_layers=1, block_loop_iterations=2, block_loop_embedding="per_layer"
        )
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)

        out0, _ = layer(
            hidden_states=hidden_states.clone(), attention_mask=attention_mask,
            block_loop_iteration=0,
        )
        out1, _ = layer(
            hidden_states=hidden_states.clone(), attention_mask=attention_mask,
            block_loop_iteration=1,
        )
        assert not torch.allclose(out0, out1, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow to attention and MoE parameters."""
        config = _make_config(num_layers=1, block_loop_iterations=2)
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        hidden_states.requires_grad_(True)

        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask,
            block_loop_iteration=0,
        )
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.abs().sum() > 0
        attn_params = list(layer.self_attention.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn_params)
        mlp_params = list(layer.mlp.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mlp_params)

    def test_gradient_flow_to_per_layer_embedding(self):
        config = _make_config(
            num_layers=1, block_loop_iterations=2, block_loop_embedding="per_layer"
        )
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)

        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask,
            block_loop_iteration=0,
        )
        loss = output.sum()
        loss.backward()

        assert layer.block_loop_embedding.grad is not None
        assert layer.block_loop_embedding.grad.abs().sum() > 0

    def test_gradient_flow_to_gate(self):
        config = _make_config(
            num_layers=1, block_loop_iterations=2, block_loop_scaling="learned_gate"
        )
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)

        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask,
            block_loop_iteration=0,
        )
        loss = output.sum()
        loss.backward()

        assert layer.block_loop_gate.grad is not None
        assert layer.block_loop_gate.grad.abs().sum() > 0

    @pytest.mark.parametrize("scaling", ["none", "uniform", "learned_gate"])
    def test_all_scaling_modes(self, scaling):
        config = _make_config(
            num_layers=1, block_loop_iterations=2, block_loop_scaling=scaling
        )
        layer = _build_layer(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output, _ = layer(
            hidden_states=hidden_states, attention_mask=attention_mask,
            block_loop_iteration=0,
        )
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ============================================================
# Forward Tests - TransformerBlock Full-Model Loop (GPU)
# ============================================================
class TestBlockLoopBlockForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_n1_output_shape(self):
        config = _make_config(num_layers=2, block_loop_iterations=1)
        block = _build_block(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape

    def test_n2_output_shape(self):
        config = _make_config(num_layers=2, block_loop_iterations=2)
        block = _build_block(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape

    def test_n2_different_from_n1(self):
        """Full-model loop N=2 should differ from N=1."""
        config_n1 = _make_config(num_layers=2, block_loop_iterations=1)
        config_n2 = _make_config(num_layers=2, block_loop_iterations=2)
        block_n1 = _build_block(config_n1).cuda()
        block_n2 = _build_block(config_n2).cuda()
        # Copy weights
        block_n2.load_state_dict(block_n1.state_dict(), strict=False)

        hidden_states, attention_mask = _make_inputs(config_n1)
        out_n1 = block_n1(hidden_states=hidden_states.clone(), attention_mask=attention_mask)
        out_n2 = block_n2(hidden_states=hidden_states.clone(), attention_mask=attention_mask)
        assert not torch.allclose(out_n1, out_n2, atol=1e-5)

    def test_gradient_flow_through_block(self):
        config = _make_config(num_layers=2, block_loop_iterations=2)
        block = _build_block(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        hidden_states.requires_grad_(True)

        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.abs().sum() > 0
        # Check all layers get gradients
        for layer in block.layers:
            attn_params = list(layer.self_attention.parameters())
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in attn_params)

    def test_global_embedding_gradient(self):
        config = _make_config(
            num_layers=2, block_loop_iterations=2, block_loop_embedding="global"
        )
        block = _build_block(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)

        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        loss = output.sum()
        loss.backward()

        assert block.block_loop_global_embedding.grad is not None
        assert block.block_loop_global_embedding.grad.abs().sum() > 0

    def test_per_layer_embedding_gradient(self):
        config = _make_config(
            num_layers=2, block_loop_iterations=2, block_loop_embedding="per_layer"
        )
        block = _build_block(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)

        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        loss = output.sum()
        loss.backward()

        for layer in block.layers:
            assert layer.block_loop_embedding.grad is not None
            assert layer.block_loop_embedding.grad.abs().sum() > 0

    def test_learned_gate_gradient_through_block(self):
        config = _make_config(
            num_layers=2, block_loop_iterations=2, block_loop_scaling="learned_gate"
        )
        block = _build_block(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)

        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        loss = output.sum()
        loss.backward()

        for layer in block.layers:
            assert layer.block_loop_gate.grad is not None
            assert layer.block_loop_gate.grad.abs().sum() > 0

    def test_n2_no_nan(self):
        config = _make_config(
            num_layers=2, block_loop_iterations=2,
            block_loop_embedding="per_layer", block_loop_scaling="learned_gate",
        )
        block = _build_block(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_n3_output_shape(self):
        config = _make_config(num_layers=2, block_loop_iterations=3)
        block = _build_block(config).cuda()
        hidden_states, attention_mask = _make_inputs(config)
        output = block(hidden_states=hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape


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
