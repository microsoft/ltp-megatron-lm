# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""
import json
import os
import sys
from datetime import datetime
from typing import List, Dict

import torch
from megatron.training import get_args

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:

        import warnings
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.training import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import DistributedDataParallel as DDP
from megatron.core.distributed.custom_fsdp import FullyShardedDataParallel as custom_FSDP
from megatron.core import mpu
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_data_parallel_group_if_dtensor,
    to_local_if_dtensor,
)
from megatron.core.transformer.module import Float16Module
from megatron.legacy.model.module import param_is_not_shared

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, torch_FSDP, custom_FSDP, Float16Module)
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, custom_FSDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model, force_create_fp32_copy=False):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Seperate moe and dense params
    params_data = []
    moe_params_data = []
    sharded_params_data = []
    data_parallel_group = None

    custom_fsdp_all_param_is_shared = False
    for model_chunk in model:
        for param in model_chunk.parameters():
            data_parallel_group = get_data_parallel_group_if_dtensor(param, data_parallel_group)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if not is_not_tp_duplicate:
                continue
            assert is_not_tp_duplicate
            if hasattr(param, "fully_shard_param_local_shard"):
                param = param.fully_shard_param_local_shard
                assert [getattr(p, "fully_shard_param_local_shard", None) is not None for p in model_chunk.parameters()]
                custom_fsdp_all_param_is_shared = True
                if param.numel() == 0:
                    continue
            if not getattr(param, 'allreduce', True):
                # TODO: Implement memory optimization for MoE parameters.
                assert param_is_not_shared(param)
                param = to_local_if_dtensor(param)
                moe_params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                if param_is_not_shared(param):
                    param = to_local_if_dtensor(param)
                    if args.bf16:
                        if not force_create_fp32_copy and hasattr(param, 'main_param'):
                            if getattr(param, 'main_param_sharded', False):
                                if param.main_param is not None:
                                    sharded_params_data.append(param.main_param)
                            else:
                                params_data.append(param.main_param)
                        else:
                            # Fallback to original logic of making a fp32 copy of the
                            # parameter if `.main_param` attribute is not available.
                            params_data.append(param.data.float())
                    else:
                        params_data.append(param.data)

    # Calculate norm.
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    if len(params_data) > 0:
        norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [params_data],
            False # no per-parameter norm.
        )
        norm_2 = norm * norm
    else:
        norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')

    if data_parallel_group is not None:
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=data_parallel_group)

    # Add norm contribution from params with sharded main_params. These norms need to be
    # accumulated across the DP group since the main parameters are sharded because
    # of distributed optimizer.
    if len(sharded_params_data) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        sharded_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [sharded_params_data],
            False # no per-parameter norm.
        )
        sharded_norm_2 = sharded_norm * sharded_norm
        # Sum over all DP groups.
        torch.distributed.all_reduce(
            sharded_norm_2,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_data_parallel_group()
        )
        norm_2 += sharded_norm_2

    if custom_fsdp_all_param_is_shared:
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_data_parallel_group())

    # Add norm contribution from expert layers in MoEs.
    if len(moe_params_data) > 0:
        moe_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [moe_params_data],
            False # no per-parameter norm.
        )
        moe_norm_2 = moe_norm * moe_norm

        if custom_fsdp_all_param_is_shared:
            torch.distributed.all_reduce(moe_norm_2,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=mpu.get_expert_data_parallel_group())
    # Account for MoE norm even if current rank doesn't have any expert params to prevent
    # hang in models with un-even numbers of MoE layers.
    # See details in https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/409
    else:
        moe_norm_2 = torch.zeros_like(norm_2)

    # Reduce norm across model parallel groups (dense and expert).
    # Dense params should sum across all model-parallel GPUs (tensor + pipeline).
    dense_reduce_group = mpu.get_model_parallel_group()
    ranks_in_dense_reduce_group = torch.distributed.get_process_group_ranks(dense_reduce_group)
    # Expert params should sum across all model-parallel GPUs (expert + tensor + pipeline).
    expert_reduce_group = mpu.get_expert_tensor_model_pipeline_parallel_group()
    ranks_in_expert_reduce_group = torch.distributed.get_process_group_ranks(expert_reduce_group)

    # If dense and expert reduce groups are the same, sum then reduce.
    if ranks_in_dense_reduce_group == ranks_in_expert_reduce_group:
        norm_2 += moe_norm_2
        torch.distributed.all_reduce(
            norm_2,
            op=torch.distributed.ReduceOp.SUM,
            group=dense_reduce_group
        )
    # If dense and expert reduce groups are different, reduce then sum.
    else:
        torch.distributed.all_reduce(
            norm_2,
            op=torch.distributed.ReduceOp.SUM,
            group=dense_reduce_group
        )
        torch.distributed.all_reduce(
            moe_norm_2,
            op=torch.distributed.ReduceOp.SUM,
            group=expert_reduce_group
        )
        norm_2 += moe_norm_2

    return norm_2.item() ** 0.5


class NormalizationCalculator:
    def __init__(self, model, global_layer_offset=None):
        if not isinstance(model, list):
            print("Warning: calc_params_l2_norm_per_param only support list of models")
            model = [model]

        self.sigma_model_structure = {
            "embedding": ["word_embeddings.weight", ],
            "output_layer": ["weight"],
            "final_layernorm": ["weight"],
            "decoder": ["self_attention.linear_proj.weight", 
                      "self_attention.linear_qkv.layer_norm_weight",
                      "self_attention.linear_qkv.weight",
                      "mlp.linear_fc2.weight",
                      "mlp.linear_fc1.layer_norm_weight",
                      "mlp.linear_fc1.weight",
                      "input_layernorm.weight",
                      "self_attention.linear_q_proj.weight",
                      "self_attention.linear_kv_down_proj.weight",
                      "self_attention.linear_kv_up_proj.layer_norm_weight",
                      "self_attention.linear_kv_up_proj.weight",
                      "pre_mlp_layernorm.weight",
                      "mlp.router.weight",
                      "mlp.shared_experts.linear_fc2.weight",
                      "mlp.shared_experts.linear_fc1.weight",
                    ]
        }

        self.dense_param_names = []
        self.moe_param_names = []
        self.moe_layer_names = []

        # Cache parameter shapes
        self.param_shapes = {}

        self.data_parallel_group = None
        for model_chunk in model:
            for i, (name, param) in enumerate(model_chunk.named_parameters()):
                self.data_parallel_group = get_data_parallel_group_if_dtensor(param, self.data_parallel_group)
                is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
                if not (param.requires_grad and is_not_tp_duplicate):
                    continue
                assert is_not_tp_duplicate
                if not getattr(param, 'allreduce', True):
                    assert param_is_not_shared(param)
                    param = to_local_if_dtensor(param)
                    self.moe_param_names.append(name)
                else:
                    if param_is_not_shared(param):
                        param = to_local_if_dtensor(param)
                        self.dense_param_names.append(name)

                self.param_shapes[name] = param.shape

        # Find MOE params
        for name in self.moe_param_names:
            moe_layer_name = name.rsplit(".", 1)[0]
            if moe_layer_name not in self.moe_layer_names:
                self.moe_layer_names.append(moe_layer_name)

        self.global_layer_offset = global_layer_offset

    def parse_local_layer_id(self, name):
        """get local layer id from name."""
        if not "layers" in name:
            return -1
        segs = name.split(".")
        for idx in range(len(segs)):
            if segs[idx] == "layers":
                break
        return int(segs[idx+1])

    def map_param_index(self, name):
        """Map parameter name to a tuple of (global_layer_id, weight_index)."""
        ele = name.split('.')
        if len(ele) > 2:
            component = ele[2]
            if component == "embedding":
                return (0, 0)
            elif component == "output_layer":
                return (1, 0)
            elif component == "final_layernorm":
                return (2, 0)
            elif component == "decoder":
                local_layer_id = self.parse_local_layer_id(name)
                # Work in 64 layers with 8 pipeline parallel size
                global_layer_id = local_layer_id + self.global_layer_offset

                weight_index = None
                for i, suffix in enumerate(self.sigma_model_structure["decoder"]):
                    if name.endswith(suffix):
                        weight_index = i
                        break
                return (global_layer_id + 3, weight_index)
            else:
                print(f"Warning: {name} is not handled by map_param_index")
                return None

        else:
            print(f"Warning: {name} is not handled by map_param_index")
            return None

    def map_moe_param_index(self, name):
        local_layer_id = self.parse_local_layer_id(name)
        global_layer_id = local_layer_id + self.global_layer_offset

        return global_layer_id

    def reverse_param_index(self, param_index):
        param_types = ["embedding", "output_layer", "final_layernorm", "decoder"]
        if param_index[0] < 3:
            param_type = param_types[param_index[0]]
        else:
            param_type = "decoder"

        if param_type != "decoder":
            if param_index[1] < len(self.sigma_model_structure[param_type]):
                return f"{param_type}.{self.sigma_model_structure[param_type][param_index[1]]}"
            else:
                return None
        else:
            if param_index[1] < len(self.sigma_model_structure["decoder"]):
                return f"decoder.layers.{param_index[0]-3}.{self.sigma_model_structure['decoder'][param_index[1]]}"
            else:
                print(f"Warning: {param_index} is not handled by reverse_param_index")
                return None

    def reverse_moe_param_index(self, param_index):
        return f"decoder.layers.{param_index[0]}.experts"

    def fill_none_tensor(self, tensor_dict_list):
        filled_tensor_dict_list = [{}]
        all_params = {}
        for tensor_dict in tensor_dict_list:
            for i, (name, param) in enumerate(tensor_dict.items()):
                if name not in all_params:
                    all_params[name] = param
                else:
                    print(f"Warning: {name} already exists in all_params, skipping")
                    pass

        for name, param_shape in self.param_shapes.items():
            if name not in all_params:
                filled_tensor_dict_list[0][name] = torch.zeros(param_shape, dtype=torch.float32, device='cuda')
            else:
                filled_tensor_dict_list[0][name] = all_params[name]
        return filled_tensor_dict_list

    def build_norm_tensor(self, tensor_dict):
        args = get_args()
        data = torch.zeros(size=(len(args.log_grad_norm_per_layer_extra_patterns) + args.num_layers, len(self.sigma_model_structure['decoder'])), dtype=torch.float32, device='cuda')

        for i, (name, norm_2) in enumerate(tensor_dict.items()):
            if name in self.dense_param_names:
                index = self.map_param_index(name)
                layer_index, weight_index = index
                data[layer_index, weight_index] += norm_2[0]
        return data

    def reverse_norm_tensor(self, tensor):
        data = {}
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                if tensor[i, j] > 0:
                    name = self.reverse_param_index((i, j))
                    if name is not None:
                        data[name] = tensor[i, j].item()
        return data

    def build_moe_norm_tensor(self, tensor_dict):
        args = get_args()
        data = torch.zeros(size=(args.num_layers, ), dtype=torch.float32, device='cuda')

        for i, (name, norm_2) in enumerate(tensor_dict.items()):
            if name in self.moe_param_names:
                layer_index = self.map_moe_param_index(name)
                data[layer_index] += norm_2[0]
        return data

    def reverse_moe_norm_tensor(self, tensor):
        data = {}
        for i in range(tensor.shape[0]):
            if tensor[i] > 0:
                name = self.reverse_moe_param_index((i, 0))
                if name is not None:
                    data[name] = tensor[i].item()
        return data

    def build_expert_norm_tensor(self, moe_params):
        args = get_args()
        num_experts = args.num_experts
        data = torch.zeros(size=(args.num_layers, num_experts), dtype=torch.float32, device='cuda')
        # get local expert index
        num_local_experts = num_experts // mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        # calc expert range
        start_expert = ep_rank * num_local_experts
        end_expert = (ep_rank + 1) * num_local_experts
        for i, (name, param) in enumerate(moe_params):
            if name in self.moe_param_names:
                layer_index = self.map_moe_param_index(name)
                experts_param = param.view(num_local_experts, -1)
                experts_param_data = experts_param.data.float() if args.bf16 else experts_param.data
                moe_norm_2 = torch.square(experts_param_data).sum(dim=1)
                data[layer_index, start_expert:end_expert] += moe_norm_2
        return data

    def reverse_expert_norm_tensor(self, tensor):
        data = {}
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                data[f"layer{i}_expert{j}"] = tensor[i, j].item()
        return data

    def calc_l2_norm_list(self, tensor_dict_list: List[Dict[str, torch.Tensor]], model_parallel_group=None, reduce_dp=False):
        """Calculate l2 norm of parameters for each param """
        # Separate moe and dense params
        dense_params = []
        moe_params = []

        if model_parallel_group is None:
            model_parallel_group = mpu.get_model_parallel_group()

        for tensor_dict in tensor_dict_list:
            for i, (name, param) in enumerate(tensor_dict.items()):
                if name in self.dense_param_names:
                    dense_params.append((name, param))
                elif name in self.moe_param_names:
                    moe_params.append((name, param))

        results = self.calc_norm_per_param(dense_params, moe_params, model_parallel_group, reduce_dp)
        expert_results = self.calc_norm_per_expert(moe_params, reduce_dp=reduce_dp)
        return results, expert_results

    def calc_norm_per_param(self, dense_params, moe_params, model_parallel_group, reduce_dp):
        args = get_args()
        # Dense Params
        dense_params_data = {}
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        for name, param in dense_params:
            # Calculate dense param norm
            norm, _ = multi_tensor_applier(
                multi_tensor_l2norm,
                dummy_overflow_buf,
                [[param.data.float() if args.bf16 else param.data,],],
                False # no per-parameter norm
            )
            norm_2 = norm * norm

            if self.data_parallel_group is not None:
                print(f"rank {torch.distributed.get_rank()} reduce norm {name} data parallel")
                torch.distributed.all_reduce(norm_2,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=self.data_parallel_group)
            dense_params_data[name] = norm_2

        dense_params_tensor = self.build_norm_tensor(dense_params_data)
        # Sum across all model-parallel GPUs(tensor + pipeline).
        torch.distributed.all_reduce(
            dense_params_tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=model_parallel_group
        )

        if reduce_dp:
            torch.distributed.all_reduce(
                dense_params_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_data_parallel_group()
            )

        reduced_dense_params_data = self.reverse_norm_tensor(dense_params_tensor)

        # MOE Params
        moe_params_data = {}
        for name, param in moe_params:
            moe_norm, _ = multi_tensor_applier(
                multi_tensor_l2norm,
                dummy_overflow_buf,
                [[param.data.float() if args.bf16 else param.data,],],
                False # no per-parameter norm
            )
            moe_norm_2 = moe_norm * moe_norm
            moe_params_data[name] = moe_norm_2

        moe_params_tensor = self.build_moe_norm_tensor(moe_params_data)
        # Sum across expert tensor, model and pipeline parallel GPUs.
        torch.distributed.all_reduce(
            moe_params_tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_expert_tensor_model_pipeline_parallel_group(),
        )

        if reduce_dp:
            torch.distributed.all_reduce(
                moe_params_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_data_parallel_group()
            )

        reduced_moe_params_data = self.reverse_moe_norm_tensor(moe_params_tensor)

        results = {}

        for name, norm_2 in reduced_dense_params_data.items():
            results[name] = norm_2 ** 0.5

        for name, norm_2 in reduced_moe_params_data.items():
            results[name] = norm_2 ** 0.5
        return results

    def calc_norm_per_expert(self, moe_params, reduce_dp=False):
        expert_results = {}
        moe_data = self.build_expert_norm_tensor(moe_params)

        torch.distributed.all_reduce(
            moe_data,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_expert_tensor_model_pipeline_parallel_group())

        if reduce_dp:
            torch.distributed.all_reduce(
                moe_data,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_data_parallel_group())

        expert_results = self.reverse_expert_norm_tensor(moe_data)

        for name, norm_2 in expert_results.items():
            expert_results[name] = norm_2 ** 0.5
        return expert_results

    def calc_param_l2_norm_per_param(self, model):
        """Calculate l2 norm of parameters for each param """
        if not isinstance(model, list):
            print("Warning: calc_params_l2_norm_per_param only support list of models")
            model = [model]
        tensor_dict_list = [dict(model_chunck.named_parameters()) for model_chunck in model]
        return self.calc_l2_norm_list(tensor_dict_list)
    

def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        stat, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        return None
    else:
        return stat.item()


def logical_and_across_model_parallel_group(input: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group
    """
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        input, op=torch.distributed.ReduceOp.MIN, group=mpu.get_model_parallel_group()
    )
    return bool(input.item())


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.training.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_rank0():
    """Returns true if called in the rank0, false otherwise"""
    return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)

def get_device_arch_version():
    """Returns GPU arch version (8: Ampere, 9: Hopper, 10: Blackwell, ...)"""
    return torch.cuda.get_device_properties(torch.device("cuda:0")).major

def append_to_progress_log(string, barrier=True):
    """Append given string to progress log."""
    args = get_args()
    if args.save is None:
        return
    progress_log_filename = os.path.join(args.save, "progress.txt")
    if barrier:
        torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(progress_log_filename, 'a') as f:
            job_id = os.getenv('SLURM_JOB_ID', '')
            num_gpus = args.world_size
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t"
                    f"# GPUs: {num_gpus}\t{string}\n")


def get_blend_and_blend_per_split(args):
    """Get blend and blend_per_split from passed-in arguments."""
    use_data_path = args.data_path is not None or \
        args.data_args_path is not None
    use_per_split_data_path = any(
        elt is not None
        for elt in [args.train_data_path,
                    args.valid_data_path,
                    args.test_data_path]) or \
        args.per_split_data_args_path is not None

    blend = None
    blend_per_split = None
    if use_data_path:
        if args.data_args_path is not None:
            assert args.data_path is None
            with open(args.data_args_path, 'r') as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert args.data_path is not None
            blend = get_blend_from_list(args.data_path)
    elif use_per_split_data_path:
        if args.per_split_data_args_path is not None:
            with open(args.per_split_data_args_path, 'r') as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"])
                ]
        else:
            blend_per_split = [
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path)
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
       if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

       if data_iterator is not None:
           data = next(data_iterator)
       else:
           data = None

       batch = {
           'tokens': data["tokens"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True)
       }

       if args.pipeline_model_parallel_size == 1:
           _broadcast(batch['tokens'])
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       elif mpu.is_pipeline_first_stage():
           _broadcast(batch['tokens'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       elif mpu.is_pipeline_last_stage():
           # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
           # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
           # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
           if args.mtp_num_layers is not None:
                _broadcast(batch['tokens'])
                _broadcast(batch['position_ids'])
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])

    else:

       tokens=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       labels=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       loss_mask=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.float32 , device = torch.cuda.current_device())
       if args.create_attention_mask_in_dataloader:
           attention_mask=torch.empty(
                (args.micro_batch_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
            )
       else:
           attention_mask=None
       position_ids=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())

       if args.pipeline_model_parallel_size == 1:
           _broadcast(tokens)
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
           _broadcast(position_ids)

       elif mpu.is_pipeline_first_stage():
           labels=None
           loss_mask=None

           _broadcast(tokens)
           _broadcast(attention_mask)
           _broadcast(position_ids)

       elif mpu.is_pipeline_last_stage():
           # Multi-Token Prediction (MTP) layers need tokens and position_ids to calculate embedding.
           # Currently the Multi-Token Prediction (MTP) layers is fixed on the last stage, so we need
           # to broadcast tokens and position_ids to all of the tensor parallel ranks on the last stage.
           if args.mtp_num_layers is not None:
                _broadcast(tokens)
                _broadcast(position_ids)
           else:
               tokens=None
               position_ids=None

           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)

       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids
       }

    return batch


def update_use_dist_ckpt(args):
    args.use_dist_ckpt = args.ckpt_format != "torch"
