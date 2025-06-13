import os
import sys
import logging

logger = logging.getLogger(__name__)

MODEL_OPTIM_RNG_FILENAME = "model_optim_rng.pt"
DISTRIB_OPTIM_FILENAME = "distrib_optim.pt"

# torch.load sometimes change loglevel with weights_only=False
# so log message after torch.load does not show on terminal screen
class RetainLogLevel:
    def __enter__(self):
        self.origin_log_level = logging.getLogger().level
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        logging.getLogger().setLevel(self.origin_log_level)

def log_and_exit(message):
    logger.fatal(message)
    raise Exception("exit with fatal error")
    #sys.exit(1)

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
    ckpt_ctx):

    num_middle_stages = ckpt_ctx.pp_size - 2 # should be non-negative
    num_layers_per_middle_virtual_stage = (ckpt_ctx.num_middle_layers // (num_middle_stages * ckpt_ctx.vpp_size)) if num_middle_stages > 0 else 0

    target_global_layer_idx = target_virtual_idx * num_middle_stages * num_layers_per_middle_virtual_stage + \
        sum(ckpt_ctx.first_vpp_layer_split[:target_virtual_idx]) + sum(ckpt_ctx.last_vpp_layer_split[:target_virtual_idx]) \
        + (0 if target_pp_rank == 0 else ckpt_ctx.first_vpp_layer_split[target_virtual_idx] + (target_pp_rank-1) * num_layers_per_middle_virtual_stage)

    prefix_sum = 0
    for pp_stage in range(ckpt_ctx.pp_size):
        if pp_stage == 0:
            layers_in_current_stage = sum(ckpt_ctx.first_vpp_layer_split)
        elif pp_stage == ckpt_ctx.pp_size - 1:
            layers_in_current_stage = sum(ckpt_ctx.last_vpp_layer_split)
        else:
            layers_in_current_stage = num_layers_per_middle_virtual_stage * ckpt_ctx.vpp_size
        if prefix_sum <= target_global_layer_idx < (prefix_sum + layers_in_current_stage):
            source_start_layer_idx = target_global_layer_idx-prefix_sum
            logger.info(f"get_vpp_source_position, target_pp_rank={target_pp_rank}, target_virtual_idx={target_virtual_idx}; "
                f"source_pp_rank={pp_stage}, source_start_layer_idx={source_start_layer_idx}")
            return (pp_stage, source_start_layer_idx)
        prefix_sum += layers_in_current_stage
    log_and_exit("should never reach here")
    #

def get_num_layers_for_this_vstage(pp_rank, vpp_rank, ckpt_ctx):
    if not ckpt_ctx.uneven_mode:
        return ckpt_ctx.num_layers // (ckpt_ctx.pp_size * ckpt_ctx.vpp_size)
    if pp_rank == 0:
        return ckpt_ctx.first_vpp_layer_split[vpp_rank]
    if pp_rank == ckpt_ctx.pp_size - 1:
        return ckpt_ctx.last_vpp_layer_split[vpp_rank]
    
    num_middle_stages = ckpt_ctx.pp_size - 2
    return ckpt_ctx.num_middle_layers // (num_middle_stages * ckpt_ctx.vpp_size)


class CKPTContext:
    def check_args_and_fill(self, args, state_dict):
        self.vpp_size = args.target_virtual_pipeline_model_parallel_size
        self.pp_size = args.pipeline_model_parallel_size
        self.ep_size = args.expert_model_parallel_size
        self.first_vpp_layer_split = args.target_first_virtual_pipeline_num_layers_split
        self.last_vpp_layer_split = args.target_last_virtual_pipeline_num_layers_split

        state_dict_args = state_dict["args"]
        self.num_layers = state_dict_args.num_layers
        
        if state_dict_args.tensor_model_parallel_size != 1:
            log_and_exit("currently only tensor_model_parallel_size=1 is supported, but found {} in checkpoint".format(
                state_dict_args.tensor_model_parallel_size))
            
        if self.vpp_size <= 1:
            log_and_exit(f"target_virtual_pipeline_model_parallel_size {self.vpp_size} is smaller or equal to 1")

        if self.ep_size != state_dict_args.expert_model_parallel_size:
            log_and_exit("expert_model_parallel_size in args does not match the one in checkpoint, {} vs {}".format(
                self.ep_size, state_dict_args.expert_model_parallel_size))

        if self.pp_size != state_dict_args.pipeline_model_parallel_size:
            log_and_exit("pipeline_model_parallel_size in args does not match the one in checkpoint, {} vs {}".format(
                self.pp_size, state_dict_args.pipeline_model_parallel_size))

        if self.first_vpp_layer_split or self.last_vpp_layer_split:
            self.uneven_mode = True

            if not (self.first_vpp_layer_split and self.last_vpp_layer_split):
                log_and_exit("target_first_virtual_pipeline_num_layers_split and target_last_virtual_pipeline_num_layers_split "
                    "should be set at the same time for uneven pipeline mode")
            
            if state_dict_args.decoder_first_pipeline_num_layers != sum(self.first_vpp_layer_split) or \
                state_dict_args.decoder_last_pipeline_num_layers != sum(self.last_vpp_layer_split):
                log_and_exit("uneven layer number does not match arguments in state_dict :"
                    "decoder_first_pipeline_num_layers={}, decoder_last_pipeline_num_layers={}".format(
                    state_dict_args.decoder_first_pipeline_num_layers, state_dict_args.decoder_last_pipeline_num_layers))
                
            num_middle_pipeline_stages = self.pp_size - 2
            if num_middle_pipeline_stages < 0:
                log_and_exit("pipeline_model_parallel_size is too small for uneven mode, pipeline_model_parallel_size={}".format(
                    self.pp_size))
                
            if len(self.first_vpp_layer_split) != self.vpp_size or \
                len(self.first_vpp_layer_split) != self.vpp_size:
                log_and_exit("length of target_first_virtual_pipeline_num_layers_split and target_last_virtual_pipeline_num_layers_split should "
                    "equal to target_virtual_pipeline_model_parallel_size")
            
            self.num_middle_layers = self.num_layers - sum(self.first_vpp_layer_split) \
                - sum(self.last_vpp_layer_split)
            if num_middle_pipeline_stages > 0:
                if self.num_middle_layers <= 0:
                    log_and_exit("num_middle_layers can not be non-positve, "
                        "num_middle_layers={}, num_middle_pipeline_stages={}".format(self.num_middle_layers, num_middle_pipeline_stages))
                if self.num_middle_layers % (num_middle_pipeline_stages*self.vpp_size) != 0:
                    log_and_exit("num_middle_layers can not be evenly divided by "
                    "num_middle_pipeline_stages*target_virtual_pipeline_model_parallel_size, "
                    "num_middle_layers={}, num_middle_pipeline_stages={}".format(self.num_middle_layers, num_middle_pipeline_stages))
            elif self.num_middle_layers > 0:
                log_and_exit("insufficient num_middle_pipeline_stages, "
                    "num_middle_layers={}, num_middle_pipeline_stages={}".format(self.num_middle_layers, num_middle_pipeline_stages))
                
            state_dict_args.decoder_first_pipeline_num_layers_split = args.target_first_virtual_pipeline_num_layers_split
            state_dict_args.decoder_last_pipeline_num_layers_split = args.target_last_virtual_pipeline_num_layers_split

        else:
            self.uneven_mode = False
            if self.num_layers % (self.pp_size * self.vpp_size) != 0:
                log_and_exit("for even pipeline mode, num_layers can not be evenly divided "
                    "pipeline_model_parallel_size*target_virtual_pipeline_model_parallel_size, "
                    "num_layers={}, pipeline_model_parallel_size={}, target_virtual_pipeline_model_parallel_size={}".format(
                        self.num_layers, self.pp_size, self.vpp_size))
                
            num_layers_per_pp = self.num_layers // self.pp_size
            self.num_middle_layers = self.num_layers - num_layers_per_pp * 2

            num_layers_per_vpp_stage = self.num_layers // (self.pp_size * self.vpp_size)
            self.first_vpp_layer_split = [num_layers_per_vpp_stage for _ in range(self.vpp_size)]
            self.last_vpp_layer_split = [num_layers_per_vpp_stage for _ in range(self.vpp_size)]