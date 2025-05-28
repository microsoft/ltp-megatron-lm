import os
import sys
import logging

logger = logging.getLogger(__name__)

MODEL_OPTIM_RNG_FILENAME = "model_optim_rng.pt"
DISTRIB_OPTIM_FILENAME = "distrib_optim.pt"

def log_and_exit(message):
    logger.fatal(message)
    sys.exit(1)

def get_folder_name(args, target_pp_rank, target_ep_rank):
    folder_name = "mp_rank_00"
    if args.pipeline_model_parallel_size != 1:
        folder_name += f"_{target_pp_rank:03d}"
    if args.expert_model_parallel_size != 1:
        folder_name += f"_{target_ep_rank:03d}"
    return folder_name

def get_vpp_source_position(
    target_pp_rank,
    target_virtual_idx,
    pipeline_parallel_size,
    num_virtual_stages,
    num_layers_per_virtual_stage):

    num_layers = pipeline_parallel_size * num_layers_per_virtual_stage * num_virtual_stages
    assert num_layers % pipeline_parallel_size == 0, \
        f"num_layers({num_layers}) is not divisible by pipeline_parallel_size({pipeline_parallel_size})"
    num_layers_per_pipeline_stage = num_layers // pipeline_parallel_size

    target_global_layer_idx = (target_virtual_idx * pipeline_parallel_size + target_pp_rank) * num_layers_per_virtual_stage
    source_pp_rank = target_global_layer_idx // num_layers_per_pipeline_stage
    source_start_layer_idx = target_global_layer_idx % num_layers_per_pipeline_stage

    logger.debug(f"get_vpp_source_position, target_pp_rank={target_pp_rank}, target_virtual_idx={target_virtual_idx}; "
        f"source_pp_rank={source_pp_rank}, source_start_layer_idx={source_start_layer_idx}")
    return (source_pp_rank, source_start_layer_idx)