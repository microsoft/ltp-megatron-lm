import os
import logging
import itertools
import multiprocessing

from utils import log_and_exit
from model_optim_rng import convert_model_optim_rng
from distrib_optim import convert_distrib_optim

logger = logging.getLogger(__name__)

def _check_output_folder(args):
    output_folder = args.save_iteration_dir
    if os.path.exists(output_folder):
        if not os.path.isdir(output_folder):
            log_and_exit(f"output path {output_folder} exists but is not a directory")
        #if len(os.listdir(output_folder)) > 0:
        #    log_and_exit(f"output path {output_folder} exists but not empty")
    else:
        os.makedirs(output_folder)

def _convert_checkpoint_partial(args, target_pp_rank, target_ep_rank):
    logger.debug(f"start _convert_checkpoint_partial, pp_rank={target_pp_rank}, ep_rank={target_ep_rank}")
    target_model_state_dict = convert_model_optim_rng(args, target_pp_rank, target_ep_rank)
    convert_distrib_optim(args, target_pp_rank, target_ep_rank, target_model_state_dict)

def _func_arguments_wrapper(func_arguments):
    args, pp_rank, ep_rank = func_arguments
    _convert_checkpoint_partial(args, pp_rank, ep_rank)

def convert_checkpoint(args):
    _check_output_folder(args)

    if args.pipeline_ranks_to_process is None:
        args.pipeline_ranks_to_process = range(args.pipeline_model_parallel_size)

    pp_ranges = args.pipeline_ranks_to_process
    ep_ranges = range(args.expert_model_parallel_size)

    func_arguments_tuples = [(args, x, y) for x, y in itertools.product(pp_ranges, ep_ranges)]
    logger.info("pp_ranges : {}".format(pp_ranges))
    logger.info("ep_ranges : {}".format(ep_ranges))

    logger.info("start convert...")
    logger.debug(f"args.num_max_processing_processes={args.num_max_processing_processes}")
    with multiprocessing.Pool(processes = args.num_max_processing_processes) as pool:
        pool.map(_func_arguments_wrapper, func_arguments_tuples)
    logger.info("convert finished")