import os
import logging
from collections import OrderedDict

from utils import (
    RetainLogLevel,
    log_and_exit,
    get_folder_name,
    get_vpp_source_position,
    MODEL_OPTIM_RNG_FILENAME,
    CKPTContext,
    get_num_layers_for_this_vstage,
)

import torch

logger = logging.getLogger(__name__)

def _convert_state_dict_args(args, target_pp_rank, target_ep_rank, state_dict, ckpt_ctx):
    state_dict_args = state_dict["args"]
    if ckpt_ctx.uneven_mode:
        state_dict_args.decoder_first_pipeline_num_layers_split = ckpt_ctx.first_vpp_layer_split
        state_dict_args.decoder_last_pipeline_num_layers_split = ckpt_ctx.last_vpp_layer_split
 
    state_dict_args.num_virtual_stages_per_pipeline_rank = ckpt_ctx.vpp_size
    state_dict_args.virtual_pipeline_model_parallel_size = ckpt_ctx.vpp_size
    state_dict_args.overlap_p2p_comm = True
    state_dict_args.align_param_gather = True
    if hasattr(state_dict_args, "local_rank"):
        delattr(state_dict_args, "local_rank")
    if hasattr(state_dict_args, "rank"):
        delattr(state_dict_args, "rank")

def _convert_state_dict_optimizer(args, target_pp_rank, target_ep_rank, state_dict, ckpt_ctx):
    # optimizer state dict is equal
    state_optimizer = state_dict["optimizer"]
    optimizer_states = [state_optimizer] if not isinstance(state_optimizer, list) \
        else state_optimizer
    try:
        current_step = -1
        param_group_cadidates = []
        for opt_state in optimizer_states:
            for param_group in opt_state["optimizer"]["param_groups"]:
                if "step" in param_group:
                    current_step = param_group["step"]
                else:
                    param_group_cadidates.append(param_group)
        if current_step != -1:
            for param_group in param_group_cadidates:
                param_group["step"] = current_step
                logger.info(f"add step={current_step} in optimizer state")
    except Exception:
        logger.warning("add step to optimizer state failed")

def _convert_state_dict_rng(args, target_pp_rank, target_ep_rank, state_dict, ckpt_ctx):
    # rng state is equal
    pass

def _fetch_model_state_dict(args, src_pp_rank, src_ep_rank, src_start_layer_idx, num_layers_for_this_virtual_stage, base_layer_idx):
    src_folder_path = os.path.join(args.load_iteration_dir, get_folder_name(args, src_pp_rank, src_ep_rank))
    src_file_path = os.path.join(src_folder_path, MODEL_OPTIM_RNG_FILENAME)

    #logger.debug(f"loading {src_file_path} to fetch source tensors in virtual stage...")
    with RetainLogLevel():
        state_dict = torch.load(src_file_path, map_location="cpu", weights_only=False)
    state_dict_model = state_dict["model"]

    layer_idx_added = set()
    outputs = OrderedDict()
    for k, v in state_dict_model.items():
        if not ".layers." in k:
            continue
        layer_idx = int(k.split(".layers.")[1].split(".")[0])
        if src_start_layer_idx <= layer_idx < src_start_layer_idx+num_layers_for_this_virtual_stage:
            new_key = k.replace(f".layers.{layer_idx}", f".layers.{layer_idx - src_start_layer_idx + base_layer_idx}")
            outputs[new_key] = v.clone().detach() if torch.is_tensor(v) else v
            layer_idx_added.add(layer_idx)

    num_layers_remain = num_layers_for_this_virtual_stage - len(layer_idx_added)
    if num_layers_remain > 0:
        assert base_layer_idx == 0
        next_level_outputs = _fetch_model_state_dict(args,
            src_pp_rank+1,
            src_ep_rank,
            0,
            num_layers_remain,
            len(layer_idx_added))
        outputs.update(next_level_outputs)
    return outputs

def _convert_state_dict_model(args, target_pp_rank, target_ep_rank, state_dict, ckpt_ctx):
    state_dict_model = state_dict["model"]
    vmodels = [OrderedDict() for i in range(ckpt_ctx.vpp_size)]

    for (vidx, vmodel) in enumerate(vmodels):
        if target_pp_rank == 0 and vidx == 0:
            for (k, v) in state_dict_model.items():
                if k.startswith("embedding."):
                    vmodel[k] = v.clone().detach() if torch.is_tensor(v) else v
        
        src_pp_rank, src_start_layer_idx = get_vpp_source_position(
            target_pp_rank,
            vidx,
            ckpt_ctx)
        
        num_layers_for_this_virtual_stage = get_num_layers_for_this_vstage(target_pp_rank, vidx, ckpt_ctx)
        src_model_state_dict = _fetch_model_state_dict(
            args,
            src_pp_rank,
            target_ep_rank,
            src_start_layer_idx,
            num_layers_for_this_virtual_stage,
            0)
        vmodel.update(src_model_state_dict)

        if target_pp_rank == ckpt_ctx.pp_size-1 and vidx == ckpt_ctx.vpp_size-1:
            for (k, v) in state_dict_model.items():
                if "final_layernorm." in k or k.startswith("output_layer."):
                    vmodel[k] = v.clone().detach() if torch.is_tensor(v) else v
    
    for i in range(ckpt_ctx.vpp_size):
        state_dict[f"model{i}"] = vmodels[i]

    del state_dict["model"]

def convert_model_optim_rng(args, target_pp_rank, target_ep_rank):
    src_folder_path = os.path.join(args.load_iteration_dir, get_folder_name(args, target_pp_rank, target_ep_rank))
    src_file_path = os.path.join(src_folder_path, MODEL_OPTIM_RNG_FILENAME)

    logger.info(f"loading model_optim_rng from {src_file_path} ...")
    with RetainLogLevel():
        target_state_dict = torch.load(src_file_path, map_location="cpu", weights_only=False)

    if target_pp_rank==0 and target_ep_rank<=1:
        logger.info("[pp_rank=0][ep_rank={}] keys of model state dict : {}\n".format(target_ep_rank, target_state_dict.keys()))

    ckpt_ctx = CKPTContext()
    ckpt_ctx.check_args_and_fill(args, target_state_dict)

    _convert_state_dict_args(args, target_pp_rank, target_ep_rank, target_state_dict, ckpt_ctx)

    _convert_state_dict_optimizer(args, target_pp_rank, target_ep_rank, target_state_dict, ckpt_ctx)

    _convert_state_dict_rng(args, target_pp_rank, target_ep_rank, target_state_dict, ckpt_ctx)

    _convert_state_dict_model(args, target_pp_rank, target_ep_rank, target_state_dict, ckpt_ctx)

    # save
    target_folder_path = os.path.join(args.save_iteration_dir, get_folder_name(args, target_pp_rank, target_ep_rank))
    os.makedirs(target_folder_path, exist_ok = True)
    target_file_path = os.path.join(target_folder_path, MODEL_OPTIM_RNG_FILENAME)
    
    logger.info(f"saving model_optim_rng to {target_file_path} ...")
    torch.save(target_state_dict, target_file_path)

    return (target_state_dict, ckpt_ctx)