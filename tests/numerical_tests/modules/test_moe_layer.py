import pytest
import torch

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from tests.numerical_tests.modules.test_module import TestModule
from tests.numerical_tests.modules.test_utilities import Utils


class TestMoELayer(TestModule):
    """
    Test MoELayer
    """

    def setup_parallelism(self):
        expert_model_parallel_size = 8
        assert Utils.world_size == expert_model_parallel_size
        Utils.initialize_model_parallel(**{
            'expert_model_parallel_size': expert_model_parallel_size
        })

    def setup_random_seed(self):
        seed = 42 + 10 * torch.distributed.get_rank()
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)

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
            'add_bias_linear': False,
            'gated_linear_unit': torch.nn.functional.silu,
            'expert_model_parallel_size': 8,
            'num_moe_experts': 64,
            'moe_ffn_hidden_size': 2160,
            'moe_router_load_balancing_type': 'seq_aux_loss',
            'moe_router_score_function': 'sigmoid',
            'moe_aux_loss_score_function': 'softmax',
            'moe_router_topk': 8,
            'moe_aux_loss_coeff': 1e-3,
            'moe_shared_expert_overlap': True,
            'moe_grouped_gemm': True,
            'moe_use_legacy_grouped_gemm': True,
            'moe_token_dispatcher_type': 'alltoall',
            'moe_router_dtype': 'fp32',
            'moe_permute_fusion': True,
        },
    ])
    @pytest.mark.parametrize('steps', [10])
    def test_moe_layer(self, inputs_kv, config_kv, steps, request):
        config = TransformerConfig(**config_kv)

        if config.bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        module_spec = ModuleSpec(
            module=MoELayer,
            submodules=MoESubmodules(
                experts=ModuleSpec(
                    module=GroupedMLP,
                    submodules=None
                ),
                shared_experts=ModuleSpec(
                    module=SharedExpertMLP,
                    params={"gate": False},
                    submodules=MLPSubmodules(
                        linear_fc1=TEColumnParallelLinear,
                        linear_fc2=TERowParallelLinear,
                    )
                )
            )
        )

        model = build_module(
            module_spec,
            config=config,
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

            assert hasattr(optimizer, 'chained_optimizers')
            parameters = []
            grads = []
            for chained_optimizer in optimizer.chained_optimizers:
                parameters += chained_optimizer.get_parameters()
                grads += chained_optimizer.get_main_grads_for_grad_norm()

            self.save_output(
                [*inputs],
                [output],
                parameters,
                grads,
                step,
                request,
            )

            optimizer.step()
