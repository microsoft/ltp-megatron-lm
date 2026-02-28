# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Recursive MoE (Loop Expert) layer."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def _make_config(
    num_iterations=1,
    routing_strategy="reroute",
    residual="add",
    aux_loss_scale=1.0,
    num_experts=4,
    topk=2,
    hidden_size=12,
    dispatcher_type="allgather",
):
    """Helper to create a TransformerConfig for recursive MoE tests."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        num_moe_experts=num_experts,
        use_cpu_initialization=True,
        moe_token_dispatcher_type=dispatcher_type,
        moe_router_load_balancing_type="aux_loss",
        moe_router_topk=topk,
        moe_aux_loss_coeff=0.01,
        moe_grouped_gemm=False,
        add_bias_linear=False,
        # Recursive MoE settings
        moe_num_iterations=num_iterations,
        moe_iteration_residual=residual,
        moe_iteration_routing_strategy=routing_strategy,
        moe_iteration_aux_loss_scale=aux_loss_scale,
    )


def _build_moe_layer(config):
    """Build a MoELayer from config."""
    spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    layer = MoELayer(config, spec.submodules.mlp.submodules)
    layer.set_layer_number(1)
    return layer


class TestRecursiveMoELayerInit:
    """Test that RecursiveMoE layers initialize correctly."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_default_config_is_standard_moe(self):
        """When moe_num_iterations=1 (default), should behave as standard MoE."""
        config = _make_config(num_iterations=1)
        layer = _build_moe_layer(config)
        assert layer.num_iterations == 1
        assert layer.extra_routers is None

    @pytest.mark.parametrize("num_iterations", [2, 3])
    def test_reroute_init(self, num_iterations):
        """Reroute strategy uses a single router."""
        config = _make_config(num_iterations=num_iterations, routing_strategy="reroute")
        layer = _build_moe_layer(config)
        assert layer.num_iterations == num_iterations
        assert layer.extra_routers is None  # reroute reuses the same router

    @pytest.mark.parametrize("num_iterations", [2, 3])
    def test_multi_router_init(self, num_iterations):
        """Multi-router strategy creates N-1 extra routers."""
        config = _make_config(num_iterations=num_iterations, routing_strategy="multi_router")
        layer = _build_moe_layer(config)
        assert layer.num_iterations == num_iterations
        assert layer.extra_routers is not None
        assert len(layer.extra_routers) == num_iterations - 1

    def test_dedup_init(self):
        config = _make_config(num_iterations=2, routing_strategy="dedup")
        layer = _build_moe_layer(config)
        assert layer.num_iterations == 2
        assert layer.extra_routers is None

    def test_fixed_init(self):
        config = _make_config(num_iterations=2, routing_strategy="fixed")
        layer = _build_moe_layer(config)
        assert layer.num_iterations == 2
        assert layer.extra_routers is None


class TestRecursiveMoEForward:
    """Test forward pass of RecursiveMoE layers."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _run_forward(self, config, seq_len=8, batch_size=2):
        """Run a forward pass and return (output, input)."""
        layer = _build_moe_layer(config)
        layer.cuda()
        hidden = torch.randn(
            seq_len, batch_size, config.hidden_size,
            device=torch.cuda.current_device()
        )
        layer.train()
        output, bias = layer(hidden)
        return output, bias, hidden

    def test_standard_moe_output_shape(self):
        config = _make_config(num_iterations=1)
        output, bias, hidden = self._run_forward(config)
        assert output.shape == hidden.shape

    @pytest.mark.parametrize("num_iterations", [2, 3])
    @pytest.mark.parametrize("routing_strategy", ["reroute", "multi_router", "dedup", "fixed"])
    def test_recursive_output_shape(self, num_iterations, routing_strategy):
        """Output shape must match input shape for all strategies."""
        config = _make_config(
            num_iterations=num_iterations,
            routing_strategy=routing_strategy,
        )
        output, bias, hidden = self._run_forward(config)
        assert output.shape == hidden.shape

    @pytest.mark.parametrize("residual", ["add", "replace"])
    def test_residual_modes(self, residual):
        config = _make_config(num_iterations=2, residual=residual)
        output, bias, hidden = self._run_forward(config)
        assert output.shape == hidden.shape

    def test_iteration1_matches_standard(self):
        """With num_iterations=1, output should match standard MoE exactly."""
        model_parallel_cuda_manual_seed(42)
        config_std = _make_config(num_iterations=1)
        config_rec = _make_config(num_iterations=1)

        layer_std = _build_moe_layer(config_std).cuda()
        layer_rec = _build_moe_layer(config_rec).cuda()

        # Copy weights from standard to recursive
        layer_rec.load_state_dict(layer_std.state_dict())

        hidden = torch.randn(
            8, 2, config_std.hidden_size,
            device=torch.cuda.current_device()
        )
        layer_std.eval()
        layer_rec.eval()

        with torch.no_grad():
            out_std, _ = layer_std(hidden.clone())
            out_rec, _ = layer_rec(hidden.clone())

        torch.testing.assert_close(out_std, out_rec, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        """Verify that gradients flow through the recursive MoE layer."""
        config = _make_config(num_iterations=2, routing_strategy="reroute")
        layer = _build_moe_layer(config).cuda()
        layer.train()

        hidden = torch.randn(
            8, 2, config.hidden_size,
            device=torch.cuda.current_device(),
            requires_grad=True,
        )
        output, _ = layer(hidden)
        loss = output.sum()
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.abs().sum() > 0

        # Check that expert weights have gradients
        for name, param in layer.experts.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_multi_router_gradient_flows(self):
        """Verify gradients flow for multi_router strategy."""
        config = _make_config(num_iterations=2, routing_strategy="multi_router")
        layer = _build_moe_layer(config).cuda()
        layer.train()

        hidden = torch.randn(
            8, 2, config.hidden_size,
            device=torch.cuda.current_device(),
            requires_grad=True,
        )
        output, _ = layer(hidden)
        loss = output.sum()
        loss.backward()

        # Check main router
        assert layer.router.weight.grad is not None
        # Check extra routers
        for i, router in enumerate(layer.extra_routers):
            assert router.weight.grad is not None, f"No gradient for extra_router[{i}]"

    @pytest.mark.parametrize("dispatcher_type", ["allgather", "alltoall"])
    def test_dispatcher_types(self, dispatcher_type):
        """Test recursive MoE with different dispatcher types."""
        config = _make_config(
            num_iterations=2,
            routing_strategy="reroute",
            dispatcher_type=dispatcher_type,
        )
        output, bias, hidden = self._run_forward(config)
        assert output.shape == hidden.shape

    def test_more_iterations_changes_output(self):
        """More iterations should produce different output than fewer."""
        model_parallel_cuda_manual_seed(42)
        config_1 = _make_config(num_iterations=1)
        config_2 = _make_config(num_iterations=2)

        layer_1 = _build_moe_layer(config_1).cuda()
        layer_2 = _build_moe_layer(config_2).cuda()

        # Share weights
        layer_2.load_state_dict(layer_1.state_dict(), strict=False)

        hidden = torch.randn(
            8, 2, config_1.hidden_size,
            device=torch.cuda.current_device(),
        )
        layer_1.eval()
        layer_2.eval()

        with torch.no_grad():
            out_1, _ = layer_1(hidden.clone())
            out_2, _ = layer_2(hidden.clone())

        # Outputs should differ (2 iterations vs 1)
        assert not torch.allclose(out_1, out_2, atol=1e-5)


class TestRecursiveMoEConfigValidation:
    """Test that invalid configs raise appropriate errors."""

    def test_invalid_num_iterations(self):
        with pytest.raises(ValueError, match="moe_num_iterations must be >= 1"):
            _make_config(num_iterations=0)

    def test_invalid_residual(self):
        with pytest.raises(ValueError, match="moe_iteration_residual"):
            _make_config(residual="invalid")

    def test_invalid_routing_strategy(self):
        with pytest.raises(ValueError, match="moe_iteration_routing_strategy"):
            _make_config(routing_strategy="invalid")
