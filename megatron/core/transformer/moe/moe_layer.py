# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


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

    def _apply_dedup_mask(self, routing_map, accumulated_routing_map):
        """For 'dedup' strategy: mask out (token, expert) pairs already selected in prior iterations.

        Args:
            routing_map: Current routing map [num_tokens, num_experts] (bool).
            accumulated_routing_map: Union of all prior routing maps [num_tokens, num_experts] (bool).

        Returns:
            Modified routing_map with previously-selected pairs masked out.
        """
        # Mask out experts already selected in prior iterations
        dedup_mask = ~accumulated_routing_map  # True for NOT-yet-selected
        routing_map = routing_map & dedup_mask
        return routing_map

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
            # Save original input for shared expert (applied once at the end)
            original_hidden_states = hidden_states
            accumulated_routing_map = None  # For dedup strategy
            fixed_probs = None  # For fixed strategy
            fixed_routing_map = None
            final_mlp_bias = None

            for iteration in range(self.num_iterations):
                # --- Set aux loss scale for this iteration ---
                aux_scale = 1.0 if iteration == 0 else self.iteration_aux_loss_scale
                
                # --- Routing ---
                if self.iteration_routing_strategy == "fixed":
                    # Route only on the first iteration, reuse for subsequent ones
                    if iteration == 0:
                        fixed_probs, fixed_routing_map = self.router(hidden_states)
                    probs, routing_map = fixed_probs, fixed_routing_map
                elif self.iteration_routing_strategy == "multi_router":
                    router = self._get_router_for_iteration(iteration)
                    router._aux_loss_scale = aux_scale
                    probs, routing_map = router(hidden_states)
                elif self.iteration_routing_strategy == "dedup":
                    self.router._aux_loss_scale = aux_scale
                    probs, routing_map = self.router(hidden_states)
                    if accumulated_routing_map is not None:
                        routing_map = self._apply_dedup_mask(routing_map, accumulated_routing_map)
                        # Recompute probs: zero out masked positions
                        probs = probs * routing_map
                    # Track which (token, expert) pairs have been used
                    if accumulated_routing_map is None:
                        accumulated_routing_map = routing_map.clone()
                    else:
                        accumulated_routing_map = accumulated_routing_map | routing_map
                else:
                    # "reroute": simply re-invoke the same router with updated hidden_states
                    self.router._aux_loss_scale = aux_scale
                    probs, routing_map = self.router(hidden_states)

                # --- Expert computation (permute → experts → unpermute) ---
                (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                    hidden_states, probs, routing_map
                )
                expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
                output, mlp_bias = self.token_dispatcher.token_unpermutation(
                    expert_output, mlp_bias
                )

                # --- Residual connection ---
                if self.iteration_residual == "add":
                    hidden_states = hidden_states + output
                else:
                    # "replace"
                    hidden_states = output

                if mlp_bias is not None:
                    final_mlp_bias = mlp_bias  # keep the last one

            # Apply shared expert once at the end (not per iteration)
            if self.use_shared_expert and not self.shared_expert_overlap:
                hidden_states = hidden_states + self.shared_experts(original_hidden_states)

            return hidden_states, final_mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias
