import os
import logging

from utils import (
    RetainLogLevel,
    log_and_exit,
    get_folder_name,
    get_vpp_source_position,
    MODEL_OPTIM_RNG_FILENAME,
    DISTRIB_OPTIM_FILENAME,
    get_num_layers_for_this_vstage,
)

import torch

logger = logging.getLogger(__name__)

def _fetch_opt_parameters(
        args,
        src_pp_rank,
        src_ep_rank,
        src_start_layer_idx,
        num_layers_for_this_virtual_stage,
        count_dense : bool,
        count_experts : bool):
    src_folder_path = os.path.join(args.load_iteration_dir, get_folder_name(args, src_pp_rank, src_ep_rank))
    src_model_opt_rng_path = os.path.join(src_folder_path, MODEL_OPTIM_RNG_FILENAME)
    src_distrib_optim_path = os.path.join(src_folder_path, DISTRIB_OPTIM_FILENAME)
    
    with RetainLogLevel():
        state_dict_model = torch.load(src_model_opt_rng_path, map_location="cpu", weights_only=False)["model"]
    state_dict_disopts = torch.load(src_distrib_optim_path, map_location="cpu")
    if not isinstance(state_dict_disopts, list):
        state_dict_disopts = [state_dict_disopts]

    if count_dense and count_experts:
        assert len(state_dict_disopts)==1, "length of state_dict_disopts does not equal to 1"
        state_dict_disopt = state_dict_disopts[0]
    elif count_dense:
        state_dict_disopt = state_dict_disopts[0]
    else:
        assert count_experts
        assert len(state_dict_disopts)==2
        state_dict_disopt = state_dict_disopts[1]

    current_offset = 0
    start_offset, end_offset = -1, -1
    upper_bound_layer_idx = -1 # variable for double-check
    for (k, v) in state_dict_model.items():
        if not hasattr(v, "nelement"):
            continue
        if "router.expert_bias" in k:
            # in Megatron-LM
            # router.expert_bias is initialized by calling register_buffer
            continue
        if not ".layers." in k:
            if start_offset != -1:
                end_offset = current_offset
                upper_bound_layer_idx += 1
                break
            current_offset += v.nelement() if count_dense else 0
            continue
        layer_idx = int(k.split(".layers.")[1].split(".")[0])

        assert layer_idx >= upper_bound_layer_idx, \
            "double check failed,layer_idx stored unordered in checkpoint"
        upper_bound_layer_idx = layer_idx

        if layer_idx == src_start_layer_idx and start_offset == -1:
            start_offset = current_offset
        if layer_idx == src_start_layer_idx + num_layers_for_this_virtual_stage \
            and start_offset != -1:
            end_offset = current_offset
            break
        if ".mlp.experts." in k:
            current_offset += v.nelement() if count_experts else 0
        else:
            current_offset += v.nelement() if count_dense else 0
    assert start_offset != -1
    if end_offset == -1:
        end_offset = current_offset
        upper_bound_layer_idx += 1

    opt_parameters_dict = next(iter(state_dict_disopt[0].values()))
    (param_part, exp_avg_part, exp_avg_sq_part) = (
        opt_parameters_dict["param"][-end_offset:(-start_offset if start_offset!=0 else None)],
        opt_parameters_dict["exp_avg"][-end_offset:(-start_offset if start_offset!=0 else None)],
        opt_parameters_dict["exp_avg_sq"][-end_offset:(-start_offset if start_offset!=0 else None)],
    )

    num_layers_remain = num_layers_for_this_virtual_stage - upper_bound_layer_idx + src_start_layer_idx
    if num_layers_remain > 0:
        (next_level_param_part, next_level_exp_avg_part, next_level_exp_avg_sq_part) = _fetch_opt_parameters(
        args,
        src_pp_rank + 1,
        src_ep_rank,
        0,
        num_layers_remain,
        count_dense,
        count_experts)

        param_part =  torch.cat((next_level_param_part, param_part))
        exp_avg_part = torch.cat((next_level_exp_avg_part, exp_avg_part))
        exp_avg_sq_part = torch.cat((next_level_exp_avg_sq_part, exp_avg_sq_part))

    return (param_part, exp_avg_part, exp_avg_sq_part)

def convert_distrib_optim(args, target_pp_rank, target_ep_rank, target_model_state_dict, ckpt_ctx):
    src_folder_path = os.path.join(args.load_iteration_dir, get_folder_name(args, target_pp_rank, target_ep_rank))
    src_file_path = os.path.join(src_folder_path, DISTRIB_OPTIM_FILENAME)

    logger.info(f"loading distrib_optim state dict {src_file_path} ...")
    target_disopt_state_dicts = torch.load(src_file_path, map_location="cpu")
    
    if not isinstance(target_disopt_state_dicts, list):
        logger.info("target_disopt_state_dicts is not a list, pp_size={}, ep_size={}, pp_rank={}, ep_rank={}".format(
            ckpt_ctx.pp_size,
            ckpt_ctx.ep_size,
            target_pp_rank,
            target_ep_rank))
        target_disopt_state_dicts = [target_disopt_state_dicts]
    else:
        logger.info("length of target_disopt_state_dicts : {}".format(len(target_disopt_state_dicts)))

    """
    in distrib_optim.pt
    for ep=1, len(target_disopt_state_dicts) is always 1 containing all parameters
    for ep>1:
        if ep_rank=0
            target_disopt_state_dicts[0] contains non-experts parameters
            target_disopt_state_dicts[i>0] only contains experts parameters if experts exists
        if ep_rank>1
            target_disopt_state_dicts[0] is None
            target_disopt_state_dicts[i>0] only contains experts parameters if experts exists
    """

    ###
    for (i, target_disopt_state_dict) in enumerate(target_disopt_state_dicts):
        if target_disopt_state_dict is None:
            continue
        if 0 not in target_disopt_state_dict:
            log_and_exit("0 is not a key of target_disopt_state_dict, keys : {}".format(target_disopt_state_dict.keys()))
        src_disopt_state_dict = target_disopt_state_dict[0]
        if len(src_disopt_state_dict) != 1:
            log_and_exit("length of src_disopt_state_dict is not 1, keys : {}".format(src_disopt_state_dict.keys()))
        type_key = next(iter(src_disopt_state_dict))
        opt_parameters_dict = src_disopt_state_dict[type_key]
        if target_pp_rank==0 and target_ep_rank<=1:
            logger.info("[pp_rank=0][ep_rank={}] keys of opt_parameters_dict[{}] : {}".format(
                target_ep_rank, i, opt_parameters_dict.keys()))
            
        vdisopts = []
        for vidx in range(ckpt_ctx.vpp_size):
            """
            value in opt_parameters_dict are flatened tensors and were saved in reverse order compare to model state_dict
            for example:
                model_ordered_state_dict(key-tensor pair) : {
                    k1 : t1,
                    k2 : t2,
                    k3 : t3,
                    ...
                    kn : tn
                }
                opt_parameters_dict["param"] : [tn, ... t3, t2, t1]
                opt_parameters_dict["exp_avg"] : [tn, ... t3, t2, t1]
            """
            num_elements_in_vmodel = 0
            vmodel_dict = target_model_state_dict[f"model{vidx}"]
            new_parameters_dict = {
                "param" : torch.tensor([], dtype=opt_parameters_dict["param"].dtype, device="cpu"),
                "exp_avg" : torch.tensor([], dtype=opt_parameters_dict["exp_avg"].dtype, device="cpu"),
                "exp_avg_sq" : torch.tensor([], dtype=opt_parameters_dict["exp_avg_sq"].dtype, device="cpu"),
                "numel_unpadded" : 0,
            }

            # final_layernorm and output_layer
            if target_pp_rank == ckpt_ctx.pp_size - 1 \
                and vidx == ckpt_ctx.vpp_size - 1 \
                and i == 0:
                num_elements_tail = 0
                for (k, v) in reversed(vmodel_dict.items()):
                    if "final_layernorm." in k or "output_layer." in k:
                        if hasattr(v, "nelement"):
                            num_elements_tail += v.nelement()
                    else:
                        break
                logger.debug(f"[pp_rank={target_pp_rank}][ep_rank={target_ep_rank}] apply output_layer, "
                    f"num_elements_tail={num_elements_tail}")
                assert num_elements_tail != 0, f"[pp_rank={target_pp_rank}][ep_rank={target_ep_rank}] num_elements_tail is 0"
                new_parameters_dict["param"] = torch.cat((new_parameters_dict["param"],
                    opt_parameters_dict["param"][:num_elements_tail]))
                new_parameters_dict["exp_avg"] = torch.cat((new_parameters_dict["exp_avg"],
                    opt_parameters_dict["exp_avg"][:num_elements_tail]))
                new_parameters_dict["exp_avg_sq"] = torch.cat((new_parameters_dict["exp_avg_sq"],
                    opt_parameters_dict["exp_avg_sq"][:num_elements_tail]))
                num_elements_in_vmodel += num_elements_tail

            # middle layer
            src_pp_rank, src_start_layer_idx = get_vpp_source_position(
                target_pp_rank,
                vidx,
                ckpt_ctx)
            num_layers_for_this_virtual_stage = get_num_layers_for_this_vstage(target_pp_rank, vidx, ckpt_ctx)
            param_part, exp_avg_part, exp_avg_sq_part = _fetch_opt_parameters(
                args,
                src_pp_rank,
                target_ep_rank,
                src_start_layer_idx,
                num_layers_for_this_virtual_stage,
                i==0,
                i>0 or len(target_disopt_state_dicts)==1)
            new_parameters_dict["param"] = torch.cat((new_parameters_dict["param"], param_part))
            new_parameters_dict["exp_avg"] = torch.cat((new_parameters_dict["exp_avg"], exp_avg_part))
            new_parameters_dict["exp_avg_sq"] = torch.cat((new_parameters_dict["exp_avg_sq"], exp_avg_sq_part))
            num_elements_in_vmodel += param_part.nelement()

            # embedding layer
            if target_pp_rank == 0 and vidx == 0 and i == 0:
                num_elements_head = 0
                for (k, v) in vmodel_dict.items():
                    if k.startswith("embedding."):
                        if hasattr(v, "nelement"):
                            num_elements_head += v.nelement()
                    else:
                        break
                logger.debug(f"[pp_rank={target_pp_rank}][ep_rank={target_ep_rank}] apply embedding, "
                    f"num_elements_head={num_elements_head}")
                assert num_elements_head != 0, f"[pp_rank={target_pp_rank}][ep_rank={target_ep_rank}] num_elements_head is 0"

                new_parameters_dict["param"] = torch.cat((new_parameters_dict["param"],
                    opt_parameters_dict["param"][-num_elements_head:]))
                new_parameters_dict["exp_avg"] = torch.cat((new_parameters_dict["exp_avg"],
                    opt_parameters_dict["exp_avg"][-num_elements_head:]))
                new_parameters_dict["exp_avg_sq"] = torch.cat((new_parameters_dict["exp_avg_sq"],
                    opt_parameters_dict["exp_avg_sq"][-num_elements_head:]))
                num_elements_in_vmodel += num_elements_head

            new_parameters_dict["numel_unpadded"] = num_elements_in_vmodel

            for k, v in new_parameters_dict.items():
                if torch.is_tensor(v):
                    # .clone.detach() is needed for removing unneeded data
                    #   and decrease checkpoint file size
                    new_parameters_dict[k] = v.clone().detach()

            if num_elements_in_vmodel != 0:
                vdisopts.append({type_key : new_parameters_dict})

            logger.debug(f"i={i}, vidx={vidx}, num_elements_in_vmodel={num_elements_in_vmodel}")

        for vidx, vdisopt in enumerate(vdisopts):
            target_disopt_state_dict[vidx] = vdisopt
    
    # save
    target_folder_path = os.path.join(args.save_iteration_dir, get_folder_name(args, target_pp_rank, target_ep_rank))
    os.makedirs(target_folder_path, exist_ok = True)
    target_file_path = os.path.join(target_folder_path, DISTRIB_OPTIM_FILENAME)
    
    if len(target_disopt_state_dicts) == 1:
        target_disopt_state_dicts = target_disopt_state_dicts[0]
    logger.info(f"saving distrib_optim to {target_file_path} ...")
    torch.save(target_disopt_state_dicts, target_file_path)
