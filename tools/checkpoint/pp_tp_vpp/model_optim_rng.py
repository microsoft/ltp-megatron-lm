import os
import logging
from collections import OrderedDict

from utils import (
    log_and_exit,
    get_folder_name,
    get_vpp_source_position,
    MODEL_OPTIM_RNG_FILENAME,
)

import torch

logger = logging.getLogger(__name__)

def _convert_state_dict_args(args, target_pp_rank, target_ep_rank, state_dict):
    state_dict_args = state_dict["args"]
    if args.target_virtual_pipeline_model_parallel_size <= 1:
        log_and_exit(f"target_virtual_pipeline_model_parallel_size {args.target_virtual_pipeline_model_parallel_size} is smaller or equal to 1")

    if state_dict_args.tensor_model_parallel_size != 1:
        log_and_exit("currently only tensor_model_parallel_size=1 is supported, but found {} in checkpoint".format(
            state_dict_args.tensor_model_parallel_size))
    if args.expert_model_parallel_size != state_dict_args.expert_model_parallel_size:
        log_and_exit("expert_model_parallel_size in args does not match the one in checkpoint, {} vs {}".format(
            args.expert_model_parallel_size, state_dict_args.expert_model_parallel_size))
    if args.pipeline_model_parallel_size != state_dict_args.pipeline_model_parallel_size:
        log_and_exit("pipeline_model_parallel_size in args does not match the one in checkpoint, {} vs {}".format(
            args.pipeline_model_parallel_size, state_dict_args.pipeline_model_parallel_size))
        
    if state_dict_args.num_layers and \
        state_dict_args.num_layers % (args.pipeline_model_parallel_size * args.target_virtual_pipeline_model_parallel_size) != 0:
        log_and_exit("num_layers can not be evenly divided pipeline_model_parallel_size*target_virtual_pipeline_model_parallel_size, "
            "num_layers={}, pipeline_model_parallel_size={}, target_virtual_pipeline_model_parallel_size={}".format(
                state_dict_args.num_layers, args.pipeline_model_parallel_size, args.target_virtual_pipeline_model_parallel_size))
    
    # args
    state_dict_args.num_virtual_stages_per_pipeline_rank = args.target_virtual_pipeline_model_parallel_size
    state_dict_args.virtual_pipeline_model_parallel_size = args.target_virtual_pipeline_model_parallel_size
    state_dict_args.overlap_p2p_comm = True
    state_dict_args.align_param_gather = True

def _convert_state_dict_optimizer(args, target_pp_rank, target_ep_rank, state_dict):
    # TODO (optimizer state)
    # suggest to reinitialize optimizer and do not load state_dict from checkpoint
    #del state_dict["optimizer"]
    pass

def _convert_state_dict_rng(args, target_pp_rank, target_ep_rank, state_dict):
    # TODO
    # further check rng state is equal
    pass

def _fetch_model_state_dict(args, src_pp_rank, src_ep_rank, src_start_layer_idx, num_layers_per_virtual_stage):
    src_folder_path = os.path.join(args.load_iteration_dir, get_folder_name(args, src_pp_rank, src_ep_rank))
    src_file_path = os.path.join(src_folder_path, MODEL_OPTIM_RNG_FILENAME)

    #logger.debug(f"loading {src_file_path} to fetch source tensors in virtual stage...")
    state_dict = torch.load(src_file_path, map_location="cpu", weights_only=False)
    state_dict_model = state_dict["model"]

    layer_idx_added = set()
    outputs = OrderedDict()
    for k, v in state_dict_model.items():
        if not ".layers." in k:
            continue
        layer_idx = int(k.split(".layers.")[1].split(".")[0])
        if src_start_layer_idx <= layer_idx < src_start_layer_idx+num_layers_per_virtual_stage:
            new_key = k.replace(f".layers.{layer_idx}", f".layers.{layer_idx-src_start_layer_idx}")
            outputs[new_key] = v.clone().detach() if torch.is_tensor(v) else v
            layer_idx_added.add(layer_idx)

    assert len(layer_idx_added) == num_layers_per_virtual_stage, \
        "size of layer_idx_added does not equal to num_layers_per_virtual_stage, " \
        "{} vs {}".format(len(layer_idx_added), num_layers_per_virtual_stage)
    return outputs

def _convert_state_dict_model(args, target_pp_rank, target_ep_rank, state_dict):
    state_dict_model = state_dict["model"]

    num_virtual_stages = args.target_virtual_pipeline_model_parallel_size
    num_layers_per_virtual_stage = state_dict["args"].num_layers \
        // (args.pipeline_model_parallel_size * args.target_virtual_pipeline_model_parallel_size)

    vmodels = [OrderedDict() for i in range(num_virtual_stages)]

    for (vidx, vmodel) in enumerate(vmodels):
        if target_pp_rank == 0 and vidx == 0:
            for (k, v) in state_dict_model.items():
                if k.startswith("embedding."):
                    vmodel[k] = v.clone().detach() if torch.is_tensor(v) else v
        
        src_pp_rank, src_start_layer_idx = get_vpp_source_position(
            target_pp_rank,
            vidx,
            args.pipeline_model_parallel_size,
            num_virtual_stages,
            num_layers_per_virtual_stage)
        
        src_model_state_dict = _fetch_model_state_dict(args, src_pp_rank, target_ep_rank,
            src_start_layer_idx, num_layers_per_virtual_stage)
        vmodel.update(src_model_state_dict)

        if target_pp_rank == args.pipeline_model_parallel_size-1 and vidx == num_virtual_stages-1:
            for (k, v) in state_dict_model.items():
                if "final_layernorm." in k or k.startswith("output_layer."):
                    vmodel[k] = v.clone().detach() if torch.is_tensor(v) else v
    
    for i in range(num_virtual_stages):
        state_dict[f"model{i}"] = vmodels[i]

    del state_dict["model"]

def convert_model_optim_rng(args, target_pp_rank, target_ep_rank):
    src_folder_path = os.path.join(args.load_iteration_dir, get_folder_name(args, target_pp_rank, target_ep_rank))
    src_file_path = os.path.join(src_folder_path, MODEL_OPTIM_RNG_FILENAME)

    logger.info(f"loading model_optim_rng from {src_file_path} ...")
    target_state_dict = torch.load(src_file_path, map_location="cpu", weights_only=False)
    if target_pp_rank==0 and target_ep_rank<=1:
        logger.info("[pp_rank=0][ep_rank={}] keys of model state dict : {}\n".format(target_ep_rank, target_state_dict.keys()))

    _convert_state_dict_args(args, target_pp_rank, target_ep_rank, target_state_dict)

    _convert_state_dict_optimizer(args, target_pp_rank, target_ep_rank, target_state_dict)

    _convert_state_dict_rng(args, target_pp_rank, target_ep_rank, target_state_dict)

    _convert_state_dict_model(args, target_pp_rank, target_ep_rank, target_state_dict)

    # save
    target_folder_path = os.path.join(args.save_iteration_dir, get_folder_name(args, target_pp_rank, target_ep_rank))
    os.makedirs(target_folder_path, exist_ok = True)
    target_file_path = os.path.join(target_folder_path, MODEL_OPTIM_RNG_FILENAME)
    
    logger.info(f"saving model_optim_rng to {target_file_path} ...")
    torch.save(target_state_dict, target_file_path)

    return target_state_dict