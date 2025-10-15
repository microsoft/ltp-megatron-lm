import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.dist_checkpointing import TempNamedDir

from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.training.global_vars import set_args
from megatron.training.utils import print_rank_0

CURDIR = os.path.dirname(os.path.abspath(__file__))

def create_args(
        num_layers,
        hidden_size,
        num_attn_heads,
        pipeline_parallel_size,
        ckpt_dir):
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.data_parallel_random_init = False
    args.no_save_optim = False
    args.no_save_rng = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.auto_detect_ckpt_format = False
    args.retro_add_retriever = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None
    args.use_distributed_optimizer = True
    args.use_dist_ckpt = False
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.add_position_embedding = False
    args.vocab_file = None
    args.tensor_model_parallel_size = 1
    args.ckpt_format = "torch"
    args.ckpt_isolated_save = True
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.ckpt_upload_blob_path = None
    args.perform_initialization = True
    args.num_virtual_stages_per_pipeline_rank = None
    args.num_layers = num_layers
    args.hidden_size = hidden_size
    args.num_attention_heads = num_attn_heads
    args.pipeline_model_parallel_size = pipeline_parallel_size
    args.normalization = "RMSNorm"
    args.transformer_impl = "transformer_engine"
    args.expert_model_parallel_size = 1
    args.save = ckpt_dir / "save_dir"
    args.load = ckpt_dir / "load_dir"

    return args

def get_checkpoint_content(args):

    def model_provider(args):
        transformer_config = TransformerConfig(
            add_bias_linear = False,
            params_dtype = torch.bfloat16,
            pipeline_dtype = torch.bfloat16,
            normalization = args.normalization,
            num_layers = args.num_layers,
            hidden_size = args.hidden_size,
            num_attention_heads = args.num_attention_heads,
            tensor_model_parallel_size = args.tensor_model_parallel_size,
            pipeline_model_parallel_size = args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank,
            perform_initialization = args.perform_initialization)
        
        def get_model():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                transformer_config,
                use_transformer_engine = args.transformer_impl == "transformer_engine")
            return GPTModel(
                config = transformer_config,
                transformer_layer_spec = transformer_layer_spec,
                position_embedding_type = "rope",
                vocab_size = 32,
                max_sequence_length = 32,
                pre_process = mpu.is_pipeline_first_stage(),
                post_process = mpu.is_pipeline_last_stage())
        
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
        model = []
        if args.num_virtual_stages_per_pipeline_rank \
            and args.num_virtual_stages_per_pipeline_rank > 1:
            model = []
            for i in range(args.num_virtual_stages_per_pipeline_rank):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                # Set pre_process and post_process only after virtual rank is set.
                this_model = DistributedDataParallel(transformer_config,
                    ddp_config, get_model())
                model.append(this_model)
        else:
            model.append(DistributedDataParallel(transformer_config,
                ddp_config, get_model()))
        return model
    
    model = model_provider(args)

    optimizer_config = OptimizerConfig(
        optimizer='adam',
        bf16=True,
        use_distributed_optimizer=True,
        params_dtype = torch.bfloat16,
        lr = 1e-6,
        min_lr = 1e-9)
    optimizer = get_megatron_optimizer(optimizer_config, model)
    optimizer.step()

    class MockState:
        def __init__(self, state_dict):
            self._state_dict = state_dict
            self.is_stub_optimizer = False

        def state_dict(self, is_loading=False):
            return self._state_dict

        def load_state_dict(self, state_dict):
            self._state_dict = state_dict

        def save_parameter_state(self, *args, **kwargs):
            pass

        def load_parameter_state(self, *args, **kwargs):
            pass
    
    opt_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})

    return (model, optimizer, opt_scheduler)

def reset_parallel_state(args):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size = args.tensor_model_parallel_size,
        pipeline_model_parallel_size = args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank)
    model_parallel_cuda_manual_seed(123)

def get_global_state(args, model, optimizer):

    def get_global_layer_index(num_layers,
        pp_size,
        vpp_size,
        current_pp_rank,
        current_vpp_rank,
        current_local_layer_index):
        num_layers_per_pipeline_rank = num_layers // pp_size
        if vpp_size is None or vpp_size == 1:
            return current_pp_rank * num_layers_per_pipeline_rank + current_local_layer_index
        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vpp_size
        total_virtual_chunks = num_layers // vpp_size
        return current_vpp_rank * total_virtual_chunks + (
            current_pp_rank * num_layers_per_virtual_rank)

    def to_cpu(x):
        if torch.is_tensor(x):
            return x.to("cpu")
        for k in x:
            x[k] = to_cpu(x[k])
        return x

    current_pp_rank = mpu.get_pipeline_model_parallel_rank()

    global_model_state = {}
    global_optimizer_state = {}
    for vpp_idx, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            key = name
            if ".layers." in key:
                layer_idx = int(key.split(".layers.")[1].split(".")[0])
                global_layer_idx = get_global_layer_index(
                    args.num_layers,
                    args.pipeline_model_parallel_size,
                    args.num_virtual_stages_per_pipeline_rank,
                    current_pp_rank,
                    vpp_idx,
                    layer_idx)
                key = key.replace(f".layers.{layer_idx}", f".layers.{global_layer_idx}")
            optimizer_param =  optimizer.chained_optimizers[0]._get_main_param_and_optimizer_states(param)
            global_model_state[key] = to_cpu(param)
            global_optimizer_state[key] = to_cpu(optimizer_param)
    
    return (global_model_state, global_optimizer_state)
                
def merge_state_dict_to_pipeline_rank0(local_dict):
    pipeline_group = mpu.get_pipeline_model_parallel_group()
    rank_in_group = dist.get_rank(group=pipeline_group)
    world_size = dist.get_world_size(group=pipeline_group)

    gathered_list = [None for _ in range(world_size)] if rank_in_group == 0 else None
    dist.gather_object(local_dict, gathered_list, dst=0, group=pipeline_group)

    if rank_in_group == 0:
        merged_dict = {}
        for d in gathered_list:
            merged_dict.update(d)
        return merged_dict
    return None

def is_state_dict_equal(x, y):
    if x.keys() != y.keys():
        return False
    for k in x:
        if isinstance(x[k], dict):
            assert isinstance(y[k], dict)
            if not is_state_dict_equal(x[k], y[k]):
                return False
        else:
            assert torch.is_tensor(x[k])
            assert torch.is_tensor(y[k])
            if not torch.equal(x[k], y[k]):
                return False
    return True
    
def _test_convert_pp_to_vpp_internal(ckpt_dir : Path):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        print_rank_0("current test_convert_pp_to_vpp only support world_size=8")
        return
    init_num_microbatches_calculator(rank, None, 1, 1, 1)

    args = create_args(
        num_layers=16,
        hidden_size=64,
        num_attn_heads=4,
        pipeline_parallel_size=8,
        ckpt_dir=ckpt_dir)

    set_args(args)

    reset_parallel_state(args)

    model, optimizer, opt_scheduler = get_checkpoint_content(args)
    
    # save model with virtual_pipeline_size=1
    iteration = 123
    flops = 456
    print_rank_0("saving checkpoint with virtual_pipeline_size=1...")
    save_checkpoint(iteration, model, optimizer, opt_scheduler, flops)
    

    global_model_state, global_optimizer_state = get_global_state(args, model, optimizer)
    global_model_state = merge_state_dict_to_pipeline_rank0(global_model_state)
    global_optimizer_state = merge_state_dict_to_pipeline_rank0(global_optimizer_state)

    if rank == 0:
        # convert model, increase virtual_pipeline_size to 2
        command = (
            "export PYTHONPATH={} ".format(os.path.join(CURDIR, "../..")) +
            "&& mkdir -p {}/iter_{:07d} ".format(args.load, iteration) +
            "&& echo {} > {}/latest_checkpointed_iteration.txt ".format(iteration, args.load) +
            "&& python {}/../../tools/checkpoint/pp_to_vpp/main.py ".format(CURDIR) +
                "--load-iteration-dir {}/iter_{:07d} ".format(args.save, iteration) +
                "--expert-model-parallel-size 1 " +
                "--pipeline-model-parallel-size 8 " +
                "--save-iteration-dir {}/iter_{:07d} ".format(args.load, iteration) +
                "--target-virtual-pipeline-model-parallel-size 2 " +
                "--num-max-processing-processes 2 "
        )
        print_rank_0("converting checkpoint from virtual_pipeline_size from 1 to 2")
        subprocess_result = subprocess.run(
            command,
            shell = True,
            text = True)
        print_rank_0(f"convert finished, exit code : {subprocess_result.returncode}")
        assert subprocess_result.returncode == 0

    dist.barrier()

    # change virtual_pipeline_size to 2 and load the model converted
    args.num_virtual_stages_per_pipeline_rank = 2
    args.perform_initialization = False
    reset_parallel_state(args)

    new_model, new_optimizer, new_opt_scheduler = get_checkpoint_content(args)
    
    print_rank_0("loading checkpoint with virtual_pipeline_size=2")
    loaded_iter, loaded_flops = load_checkpoint(
        new_model, new_optimizer, new_opt_scheduler, strict=True
    )
    
    # check iteration and flops are equal
    assert loaded_iter == iteration and loaded_flops == flops

    new_global_model_state, new_global_optimizer_state = get_global_state(args, new_model, new_optimizer)
    new_global_model_state = merge_state_dict_to_pipeline_rank0(new_global_model_state)
    new_global_optimizer_state = merge_state_dict_to_pipeline_rank0(new_global_optimizer_state)

    if mpu.get_pipeline_model_parallel_rank() == 0:
        # check model_state and optimizer parameter state are equal
        global_model_state_equal = is_state_dict_equal(global_model_state, new_global_model_state)
        assert global_model_state_equal
        global_optimizer_state_equal = is_state_dict_equal(global_optimizer_state, new_global_optimizer_state)
        assert global_optimizer_state_equal


"""
launch test with command:

torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 50326 \
   -m pytest -vx test_convert_checkpoint.py
"""
def test_convert_pp_to_vpp(tmp_path_dist_ckpt):
    Utils.initialize_distributed()
    with TempNamedDir(tmp_path_dist_ckpt / "test_convert_checkpoint", sync=True) as ckpt_dir:
        _test_convert_pp_to_vpp_internal(ckpt_dir)
