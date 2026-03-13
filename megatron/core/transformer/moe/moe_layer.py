# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.moe_utils import (
    save_to_aux_losses_tracker,
    save_to_tokens_per_expert_tracker,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import TENorm

    HAVE_TE_NORM = True
except ImportError:
    HAVE_TE_NORM = False


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: Optional[int] = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )

        # Recursive MoE config
        self.num_iterations = config.moe_num_iterations
        self.iteration_residual = config.moe_iteration_residual
        self.iteration_routing_strategy = config.moe_iteration_routing_strategy
        self.iteration_aux_loss_scale = config.moe_iteration_aux_loss_scale
        self.iteration_norm_enabled = config.moe_iteration_norm and self.num_iterations > 1
        self.iteration_scaling = config.moe_iteration_scaling

        # Initialize iteration normalization (applied between expert iterations)
        if self.iteration_norm_enabled:
            if HAVE_TE_NORM:
                self.iteration_norm = TENorm(
                    config, config.hidden_size, eps=config.layernorm_epsilon
                )
            else:
                # Fallback to PyTorch native RMSNorm or LayerNorm
                if config.normalization == "RMSNorm":
                    self.iteration_norm = torch.nn.RMSNorm(
                        config.hidden_size, eps=config.layernorm_epsilon
                    )
                else:
                    self.iteration_norm = torch.nn.LayerNorm(
                        config.hidden_size, eps=config.layernorm_epsilon
                    )
        else:
            self.iteration_norm = None

        # Initialize iteration embedding (iteration-aware expert computation)
        if config.moe_iteration_embedding and self.num_iterations > 1:
            self.iteration_embedding = torch.nn.Parameter(
                torch.zeros(self.num_iterations, config.hidden_size)
            )
            # Small initialization so it doesn't dominate at start
            torch.nn.init.normal_(self.iteration_embedding, mean=0.0, std=0.02)
        else:
            self.iteration_embedding = None

        # Initialize learnable output gate (per-iteration scaling)
        if self.iteration_scaling == "learned_gate" and self.num_iterations > 1:
            # Initialize all gates to 1/N so total contribution starts at ~1.0
            self.iteration_gate = torch.nn.Parameter(
                torch.full((self.num_iterations,), 1.0 / self.num_iterations)
            )
        else:
            self.iteration_gate = None

        # Initialize router(s)
        if self.iteration_routing_strategy == "multi_router" and self.num_iterations > 1:
            # Independent router per iteration
            self.router = TopKRouter(config=self.config)  # iteration 0
            self.extra_routers = torch.nn.ModuleList(
                [TopKRouter(config=self.config) for _ in range(self.num_iterations - 1)]
            )
        else:
            self.router = TopKRouter(config=self.config)
            self.extra_routers = None

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def _set_moe_layer_recompute(self):
        if self.moe_layer_recompute:
            if isinstance(self.config.moe_layer_recompute_freq, int):
                self.moe_layer_recompute = (
                    ((self.layer_number - 1) % self.config.moe_layer_recompute_freq) == 0
                )
            else:
                self.moe_layer_recompute = (
                    bool(self.config.moe_layer_recompute_freq[self.layer_number - 1])
                )

    def set_layer_number(self, layer_number: int):
        super(MoELayer, self).set_layer_number(layer_number)
        self._set_moe_layer_recompute()
        # Also set layer number for extra routers if using multi_router strategy
        if self.extra_routers is not None:
            for router in self.extra_routers:
                router.set_layer_number(layer_number)

    def _get_router_for_iteration(self, iteration: int):
        """Get the appropriate router for a given iteration index."""
        if self.iteration_routing_strategy == "multi_router" and self.extra_routers is not None:
            if iteration == 0:
                return self.router
            else:
                return self.extra_routers[iteration - 1]
        else:
            return self.router

    def _apply_dedup_mask(self, probs, routing_map, accumulated_routing_map):
        """For 'dedup' strategy: mask out (token, expert) pairs already selected
        in prior iterations and renormalize the routing probabilities.

        Args:
            probs: Routing probabilities [num_tokens, num_experts].
            routing_map: Current routing map [num_tokens, num_experts] (bool).
            accumulated_routing_map: Union of all prior routing maps [num_tokens, num_experts] (bool).

        Returns:
            Tuple of (modified probs, modified routing_map).
        """
        # Mask out experts already selected in prior iterations
        dedup_mask = ~accumulated_routing_map  # True for NOT-yet-selected
        routing_map = routing_map & dedup_mask

        # Zero out masked positions in probs
        probs = probs * routing_map

        # Renormalize probs so probability mass is preserved per token.
        # This prevents weakened expert contributions when some top-k slots are masked.
        prob_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / (prob_sum + 1e-20)

        return probs, routing_map

    def _get_output_scale(self, iteration: int):
        """Get the output scaling factor for a given iteration."""
        if self.iteration_scaling == "uniform":
            return 1.0 / self.num_iterations
        elif self.iteration_scaling == "learned_gate":
            # Return the learned gate value (used as tensor for gradient flow)
            return self.iteration_gate[iteration]
        else:
            # "none" — no scaling
            return 1.0

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            if self.num_iterations <= 1:
                # Standard MoE path — no loop overhead
                probs, routing_map = self.router(hidden_states)
                (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                    hidden_states, probs, routing_map
                )
                expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
                output, mlp_bias = self.token_dispatcher.token_unpermutation(
                    expert_output, mlp_bias
                )
                if self.use_shared_expert and not self.shared_expert_overlap:
                    output = output + self.shared_experts(hidden_states)
                return output, mlp_bias

            # ============ Recursive MoE path ============
            # KEY DESIGN: Separate "routing_input" (evolves for routing decisions)
            # from "accumulated_delta" (pure expert output to return).
            #
            # TransformerLayer does: output = residual + MoE(norm(residual))
            # So MoE must return only the expert delta, NOT the input itself.
            # routing_input evolves so later iterations can make informed routing
            # decisions based on prior expert outputs.
            original_hidden_states = hidden_states  # For shared expert
            routing_input = hidden_states  # Evolves across iterations for routing
            accumulated_delta = torch.zeros_like(hidden_states)  # Pure expert output

            accumulated_routing_map = None  # For dedup strategy
            fixed_probs = None  # For fixed strategy
            fixed_routing_map = None
            final_mlp_bias = None
            num_router_calls = 0

            # --- Iteration diagnostics state ---
            diagnostics_enabled = (
                self.config.moe_iteration_diagnostics
                and self.num_iterations > 1
                and self.training
                and torch.is_grad_enabled()
                and self.layer_number is not None
            )
            prev_routing_map = None
            prev_logits = None
            diag_overlap_sum = 0.0
            diag_kl_sum = 0.0
            diag_pair_count = 0

            for iteration in range(self.num_iterations):
                # --- Apply iteration normalization (iteration > 0) ---
                if iteration > 0 and self.iteration_norm is not None:
                    routing_input = self.iteration_norm(routing_input)

                # --- Apply iteration embedding (iteration-aware) ---
                if self.iteration_embedding is not None:
                    current_routing_input = routing_input + self.iteration_embedding[iteration]
                else:
                    current_routing_input = routing_input

                # --- Set aux loss scale for this iteration ---
                aux_scale = 1.0 if iteration == 0 else self.iteration_aux_loss_scale

                # --- Routing ---
                if self.iteration_routing_strategy == "fixed":
                    if iteration == 0:
                        fixed_probs, fixed_routing_map = self.router(current_routing_input)
                        num_router_calls += 1
                    probs, routing_map = fixed_probs, fixed_routing_map
                elif self.iteration_routing_strategy == "multi_router":
                    router = self._get_router_for_iteration(iteration)
                    router._aux_loss_scale = aux_scale
                    probs, routing_map = router(current_routing_input)
                    num_router_calls += 1
                elif self.iteration_routing_strategy == "dedup":
                    self.router._aux_loss_scale = aux_scale
                    probs, routing_map = self.router(current_routing_input)
                    num_router_calls += 1
                    if accumulated_routing_map is not None:
                        probs, routing_map = self._apply_dedup_mask(
                            probs, routing_map, accumulated_routing_map
                        )
                    if accumulated_routing_map is None:
                        accumulated_routing_map = routing_map.clone()
                    else:
                        accumulated_routing_map = accumulated_routing_map | routing_map
                else:
                    # "reroute"
                    self.router._aux_loss_scale = aux_scale
                    probs, routing_map = self.router(current_routing_input)
                    num_router_calls += 1

                # --- Iteration diagnostics (after routing, before expert computation) ---
                if diagnostics_enabled:
                    router_for_iter = self._get_router_for_iteration(iteration)
                    curr_logits = router_for_iter._last_logits

                    with torch.no_grad():
                        # Metric: per-iteration routing entropy
                        if curr_logits is not None:
                            full_probs = torch.softmax(curr_logits.float(), dim=-1)
                            entropy = -(full_probs * torch.log(full_probs + 1e-12)).sum(dim=-1).mean()
                            save_to_aux_losses_tracker(
                                f"iter_diag_entropy_iter_{iteration}",
                                entropy,
                                self.layer_number,
                                self.config.num_layers,
                            )

                        # Metric: expert overlap rate (iteration > 0)
                        if prev_routing_map is not None:
                            overlap = (
                                (prev_routing_map & routing_map).float().sum(dim=-1).mean()
                                / self.router.topk
                            )
                            diag_overlap_sum += overlap.item()

                        # Metric: KL divergence between consecutive iterations
                        if prev_logits is not None and curr_logits is not None:
                            p = torch.softmax(prev_logits.float(), dim=-1)
                            q = full_probs  # already computed above
                            kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1).mean()
                            diag_kl_sum += kl.item()
                            diag_pair_count += 1

                        # Store for next iteration comparison
                        prev_routing_map = routing_map.detach().clone()
                        prev_logits = curr_logits

                # --- Expert computation (permute → experts → unpermute) ---
                # Expert sees routing_input (not the one with iteration embedding)
                (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                    routing_input, probs, routing_map
                )

                # Metric: per-iteration tokens per expert
                if diagnostics_enabled:
                    tpe_device = tokens_per_expert
                    if not tpe_device.is_cuda:
                        tpe_device = tpe_device.to(device=hidden_states.device)
                    save_to_tokens_per_expert_tracker(
                        f"iter_{iteration}_tokens_per_expert",
                        tpe_device,
                        self.layer_number,
                        self.config.num_layers,
                        reduce_group=parallel_state.get_data_parallel_group(),
                    )

                expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
                output, mlp_bias = self.token_dispatcher.token_unpermutation(
                    expert_output, mlp_bias
                )

                # --- Output scaling ---
                scale = self._get_output_scale(iteration)
                scaled_output = output * scale

                # --- Accumulate delta (always add to delta) ---
                accumulated_delta = accumulated_delta + scaled_output

                # --- Update routing_input for next iteration ---
                if self.iteration_residual == "add":
                    routing_input = routing_input + output
                else:
                    # "replace"
                    routing_input = output

                if mlp_bias is not None:
                    final_mlp_bias = mlp_bias

            # Apply shared expert once at the end (not per iteration)
            if self.use_shared_expert and not self.shared_expert_overlap:
                accumulated_delta = accumulated_delta + self.shared_experts(original_hidden_states)

            # --- Write aggregated diagnostic metrics ---
            if diagnostics_enabled and diag_pair_count > 0:
                save_to_aux_losses_tracker(
                    "iter_diag_expert_overlap",
                    torch.tensor(diag_overlap_sum / diag_pair_count, device=hidden_states.device),
                    self.layer_number,
                    self.config.num_layers,
                )
                save_to_aux_losses_tracker(
                    "iter_diag_kl_div",
                    torch.tensor(diag_kl_sum / diag_pair_count, device=hidden_states.device),
                    self.layer_number,
                    self.config.num_layers,
                )

            # NOTE: Tracker normalization for load_balancing_loss, z_loss, and
            # tokens_per_expert is NOT done here. It is done once per training step
            # in track_moe_metrics() to avoid compounding division across microbatches.

            return accumulated_delta, final_mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias
