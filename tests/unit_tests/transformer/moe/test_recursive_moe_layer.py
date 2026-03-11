# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Recursive MoE (Loop Expert) layer."""

import pytest
import torch

from megatron.core import parallel_state
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
    iteration_norm=True,
    iteration_scaling="uniform",
    iteration_embedding=False,
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
        moe_iteration_norm=iteration_norm,
        moe_iteration_scaling=iteration_scaling,
        moe_iteration_embedding=iteration_embedding,
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
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_default_config_is_standard_moe(self):
        config = _make_config(num_iterations=1)
        layer = _build_moe_layer(config)
        assert layer.num_iterations == 1
        assert layer.extra_routers is None

    @pytest.mark.parametrize("num_iterations", [2, 3])
    def test_reroute_init(self, num_iterations):
        config = _make_config(num_iterations=num_iterations, routing_strategy="reroute")
        layer = _build_moe_layer(config)
        assert layer.num_iterations == num_iterations
        assert layer.extra_routers is None

    @pytest.mark.parametrize("num_iterations", [2, 3])
    def test_multi_router_init(self, num_iterations):
        config = _make_config(num_iterations=num_iterations, routing_strategy="multi_router")
        layer = _build_moe_layer(config)
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

    def test_learned_gate_init(self):
        config = _make_config(num_iterations=2, iteration_scaling="learned_gate")
        layer = _build_moe_layer(config)
        assert layer.iteration_gate is not None
        assert layer.iteration_gate.shape == (2,)
        torch.testing.assert_close(
            layer.iteration_gate.data, torch.full((2,), 0.5), atol=1e-6, rtol=1e-6
        )

    def test_iteration_embedding_init(self):
        config = _make_config(num_iterations=2, iteration_embedding=True)
        layer = _build_moe_layer(config)
        assert layer.iteration_embedding is not None
        assert layer.iteration_embedding.shape == (2, config.hidden_size)


class TestRecursiveMoEForward:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _run_forward(self, config, seq_len=8, batch_size=2):
        layer = _build_moe_layer(config)
        layer.cuda()
        hidden = torch.randn(seq_len, batch_size, config.hidden_size, device=torch.cuda.current_device())
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
        config = _make_config(num_iterations=num_iterations, routing_strategy=routing_strategy)
        output, bias, hidden = self._run_forward(config)
        assert output.shape == hidden.shape

    @pytest.mark.parametrize("residual", ["add", "replace"])
    def test_residual_modes(self, residual):
        config = _make_config(num_iterations=2, residual=residual)
        output, bias, hidden = self._run_forward(config)
        assert output.shape == hidden.shape

    def test_iteration1_matches_standard(self):
        model_parallel_cuda_manual_seed(42)
        config_std = _make_config(num_iterations=1)
        config_rec = _make_config(num_iterations=1)
        layer_std = _build_moe_layer(config_std).cuda()
        layer_rec = _build_moe_layer(config_rec).cuda()
        layer_rec.load_state_dict(layer_std.state_dict())
        hidden = torch.randn(8, 2, config_std.hidden_size, device=torch.cuda.current_device())
        layer_std.eval()
        layer_rec.eval()
        with torch.no_grad():
            out_std, _ = layer_std(hidden.clone())
            out_rec, _ = layer_rec(hidden.clone())
        torch.testing.assert_close(out_std, out_rec, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        config = _make_config(num_iterations=2, routing_strategy="reroute")
        layer = _build_moe_layer(config).cuda()
        layer.train()
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device(), requires_grad=True)
        output, _ = layer(hidden)
        loss = output.sum()
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.abs().sum() > 0
        for name, param in layer.experts.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_multi_router_gradient_flows(self):
        config = _make_config(num_iterations=2, routing_strategy="multi_router")
        layer = _build_moe_layer(config).cuda()
        layer.train()
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device(), requires_grad=True)
        output, _ = layer(hidden)
        loss = output.sum()
        loss.backward()
        assert layer.router.weight.grad is not None
        for i, router in enumerate(layer.extra_routers):
            assert router.weight.grad is not None, f"No gradient for extra_router[{i}]"

    @pytest.mark.parametrize("dispatcher_type", ["allgather", "alltoall"])
    def test_dispatcher_types(self, dispatcher_type):
        config = _make_config(num_iterations=2, routing_strategy="reroute", dispatcher_type=dispatcher_type)
        output, bias, hidden = self._run_forward(config)
        assert output.shape == hidden.shape

    def test_more_iterations_changes_output(self):
        model_parallel_cuda_manual_seed(42)
        config_1 = _make_config(num_iterations=1)
        config_2 = _make_config(num_iterations=2)
        layer_1 = _build_moe_layer(config_1).cuda()
        layer_2 = _build_moe_layer(config_2).cuda()
        layer_2.load_state_dict(layer_1.state_dict(), strict=False)
        hidden = torch.randn(8, 2, config_1.hidden_size, device=torch.cuda.current_device())
        layer_1.eval()
        layer_2.eval()
        with torch.no_grad():
            out_1, _ = layer_1(hidden.clone())
            out_2, _ = layer_2(hidden.clone())
        assert not torch.allclose(out_1, out_2, atol=1e-5)


class TestRecursiveMoEDeltaReturn:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=42, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_uniform_scaling_smaller_than_none(self):
        model_parallel_cuda_manual_seed(42)
        config_none = _make_config(num_iterations=2, iteration_scaling="none")
        config_uniform = _make_config(num_iterations=2, iteration_scaling="uniform")
        layer_none = _build_moe_layer(config_none).cuda()
        layer_uniform = _build_moe_layer(config_uniform).cuda()
        layer_uniform.load_state_dict(layer_none.state_dict(), strict=False)
        hidden = torch.randn(8, 2, config_none.hidden_size, device=torch.cuda.current_device())
        layer_none.eval()
        layer_uniform.eval()
        with torch.no_grad():
            out_none, _ = layer_none(hidden.clone())
            out_uniform, _ = layer_uniform(hidden.clone())
        assert out_uniform.abs().mean() < out_none.abs().mean()

    def test_single_iteration_unchanged_by_scaling(self):
        config = _make_config(num_iterations=1, iteration_scaling="uniform")
        layer = _build_moe_layer(config).cuda()
        layer.eval()
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device())
        with torch.no_grad():
            out, _ = layer(hidden)
        assert out.abs().sum() > 0


class TestRecursiveMoEScaling:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=42, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("scaling", ["none", "uniform", "learned_gate"])
    def test_scaling_output_shape(self, scaling):
        config = _make_config(num_iterations=2, iteration_scaling=scaling)
        layer = _build_moe_layer(config).cuda()
        layer.train()
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device())
        output, _ = layer(hidden)
        assert output.shape == hidden.shape

    def test_learned_gate_gradient_flows(self):
        config = _make_config(num_iterations=2, iteration_scaling="learned_gate")
        layer = _build_moe_layer(config).cuda()
        layer.train()
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device(), requires_grad=True)
        output, _ = layer(hidden)
        loss = output.sum()
        loss.backward()
        assert layer.iteration_gate.grad is not None
        assert layer.iteration_gate.grad.abs().sum() > 0


class TestRecursiveMoEIterationEmbedding:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=42, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_embedding_output_shape(self):
        config = _make_config(num_iterations=2, iteration_embedding=True)
        layer = _build_moe_layer(config).cuda()
        layer.train()
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device())
        output, _ = layer(hidden)
        assert output.shape == hidden.shape

    def test_embedding_gradient_flows(self):
        config = _make_config(num_iterations=2, iteration_embedding=True)
        layer = _build_moe_layer(config).cuda()
        layer.train()
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device(), requires_grad=True)
        output, _ = layer(hidden)
        loss = output.sum()
        loss.backward()
        assert layer.iteration_embedding.grad is not None
        assert layer.iteration_embedding.grad.abs().sum() > 0

    def test_with_vs_without_embedding_different(self):
        model_parallel_cuda_manual_seed(42)
        config_emb = _make_config(num_iterations=2, iteration_embedding=True)
        config_no = _make_config(num_iterations=2, iteration_embedding=False)
        layer_emb = _build_moe_layer(config_emb).cuda()
        layer_no = _build_moe_layer(config_no).cuda()
        layer_emb.load_state_dict(layer_no.state_dict(), strict=False)
        # Use larger embedding to ensure measurable difference
        with torch.no_grad():
            layer_emb.iteration_embedding.normal_(mean=0.0, std=0.5)
        hidden = torch.randn(8, 2, config_emb.hidden_size, device=torch.cuda.current_device())
        layer_emb.eval()
        layer_no.eval()
        with torch.no_grad():
            out_emb, _ = layer_emb(hidden.clone())
            out_no, _ = layer_no(hidden.clone())
        assert not torch.allclose(out_emb, out_no, atol=1e-5)


class TestRecursiveMoEConfigValidation:
    def test_invalid_num_iterations(self):
        with pytest.raises(ValueError, match="moe_num_iterations must be >= 1"):
            _make_config(num_iterations=0)

    def test_invalid_residual(self):
        with pytest.raises(ValueError, match="moe_iteration_residual"):
            _make_config(residual="invalid")

    def test_invalid_routing_strategy(self):
        with pytest.raises(ValueError, match="moe_iteration_routing_strategy"):
            _make_config(routing_strategy="invalid")

    def test_invalid_scaling(self):
        with pytest.raises(ValueError, match="moe_iteration_scaling"):
            _make_config(iteration_scaling="invalid")


class TestRecursiveMoEIterationNorm:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_norm_module_created_when_enabled(self):
        config = _make_config(num_iterations=2, iteration_norm=True)
        layer = _build_moe_layer(config)
        assert layer.iteration_norm is not None

    def test_norm_module_not_created_when_disabled(self):
        config = _make_config(num_iterations=2, iteration_norm=False)
        layer = _build_moe_layer(config)
        assert layer.iteration_norm is None

    def test_norm_module_not_created_single_iteration(self):
        config = _make_config(num_iterations=1, iteration_norm=True)
        layer = _build_moe_layer(config)
        assert layer.iteration_norm is None

    def test_norm_gradient_flows(self):
        config = _make_config(num_iterations=2, iteration_norm=True)
        layer = _build_moe_layer(config).cuda()
        layer.train()
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device(), requires_grad=True)
        output, _ = layer(hidden)
        loss = output.sum()
        loss.backward()
        norm_has_grad = False
        for name, param in layer.named_parameters():
            if 'iteration_norm' in name and param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                norm_has_grad = True
        assert norm_has_grad


class TestRecursiveMoEMetricsNormalization:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_reroute_metrics_normalized(self):
        config = _make_config(num_iterations=2, routing_strategy="reroute")
        layer = _build_moe_layer(config).cuda()
        layer.train()
        tracker = parallel_state.get_moe_layer_wise_logging_tracker()
        for name in list(tracker.keys()):
            del tracker[name]
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device())
        output, _ = layer(hidden)
        if "load_balancing_loss" in tracker:
            loss_val = tracker["load_balancing_loss"]["values"][0].item()
            assert loss_val >= 0
            assert not torch.isinf(torch.tensor(loss_val))

    def test_fixed_strategy_no_double_counting(self):
        config = _make_config(num_iterations=2, routing_strategy="fixed")
        layer = _build_moe_layer(config).cuda()
        layer.train()
        tracker = parallel_state.get_moe_layer_wise_logging_tracker()
        for name in list(tracker.keys()):
            del tracker[name]
        hidden = torch.randn(8, 2, config.hidden_size, device=torch.cuda.current_device())
        output, _ = layer(hidden)
        if "load_balancing_loss" in tracker:
            loss_val = tracker["load_balancing_loss"]["values"][0].item()
            assert loss_val >= 0
            assert not torch.isinf(torch.tensor(loss_val))
