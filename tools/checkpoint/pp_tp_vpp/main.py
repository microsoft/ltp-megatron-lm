import ast
import logging
import argparse
from vpp_converter import convert_checkpoint

logger = logging.getLogger(__name__)

def _parse_list(s):
    if s is None:
        return None
    return ast.literal_eval(s)

def parse_arguments():
    parser = argparse.ArgumentParser(description="convert a non-virtual pipeline checkpoint to virtual pipeline checkpoint")
    parser.add_argument("--load-iteration-dir", type=str, required=True, help="iteration folder of source model checkpoint")
    parser.add_argument("--expert-model-parallel-size", type=int, required=True, help="ep_size of original model and the target model")
    parser.add_argument("--pipeline-model-parallel-size", type=int, required=True, help="physical pp_size of original model and the target model")
    
    # arguments of target model below
    parser.add_argument("--save-iteration-dir", type=str, required=True, help="iteration folder of target model checkpoint, need to be empty if existed")
    parser.add_argument("--target-virtual-pipeline-model-parallel-size", type=int, required=True, help="vpp_size of target model")

    parser.add_argument("--num-max-processing-processes", type=int, default=4,
      help="the maximum number of processing processes used by this script, " \
      "increasing this value can speed up model conversion(but the final bottleneck may be disk bandwidth), it will also consume more CPU memory.")
    parser.add_argument('--pipeline-ranks-to-process', type=_parse_list, default=None,
      help="pipeline rank list to process using this script, to accelerate converting \
        user can launch multiple tasks on different nodes, each one process part of pipeline ranks. \
        example : --pipeline-ranks-to-process [0,1,2,3] \
        default is None, means process all pipeline ranks")

    args = parser.parse_args()
    return args

"""
This tool can convert a checkpoint without virtual pipeline parallelism into one with virtual pipeline parallelism
  by increasing the virtual pipeline stage size.
Other model parallel parameters (tensor-parallel-size, pipeline-parallel-size, expert-parallel-size ...) remain unchanged.

Note that currently, all of the following configurations must be satisfied to be supported.
  current pipeline partition is even (num_layers for each pipeline stage is equal)
  tensor-model-parallel-size=1
  expert-tensor-parallel-size=1
  ckpt_type=CheckpointType.LEGACY
  ckpt_format=torch
so the checkpoint for each iteration folder should look like this:
.
├── mp_rank_00_000_000
│  ├── distrib_optim.pt
│  └── model_optim_rng.pt
├── mp_rank_00_000_001
│  ├── distrib_optim.pt
│  └── model_optim_rng.pt
├── mp_rank_00_000_002
│  ├── distrib_optim.pt
│  └── model_optim_rng.pt
...

"""
def main(args):
    logger.info(f"args : {args}\n")
    convert_checkpoint(args)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.DEBUG, 
        format = "[%(asctime)s][%(levelname)s] %(message)s",
        handlers = [logging.StreamHandler()])
    args = parse_arguments()
    main(args)
