import os
import ast
import sys
import logging
import argparse
from parallel_convert import convert_checkpoint

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="convert a non-virtual pipeline checkpoint to virtual pipeline checkpoint")
    parser.add_argument("--load-iteration-dir", type=str, required=True, help="iteration folder of source model checkpoint")
    parser.add_argument("--expert-model-parallel-size", type=int, required=True, help="ep_size of original model and the target model")
    parser.add_argument("--pipeline-model-parallel-size", type=int, required=True, help="physical pp_size of original model and the target model")
    
    # arguments of target model below
    parser.add_argument("--save-iteration-dir", type=str, required=True, help="iteration folder of target model checkpoint, need to be empty if existed")
    parser.add_argument("--target-virtual-pipeline-model-parallel-size", type=int, required=True, help="vpp_size of target model")
    parser.add_argument("--target-first-virtual-pipeline-num-layers-split", type=int, nargs="+", default=None,
      help="only used in uneven pipeline mode, virtual pipeline split of the first stage")
    parser.add_argument("--target-last-virtual-pipeline-num-layers-split", type=int, nargs="+", default=None,
      help="only used in uneven pipeline mode, virtual pipeline split of the last stage")

    # arguments of acceleration in parallel
    parser.add_argument("--num-max-processing-processes", type=int, default=8,
      help="the maximum number of processing processes used by this script, " \
      "increasing this value can speed up model conversion(but the final bottleneck may be disk bandwidth), it will also consume more CPU memory.")
    parser.add_argument('--pipeline-ranks-to-process', type=int, nargs="+", default=None,
      help="pipeline rank list to process using this script, to accelerate converting \
        user can launch multiple tasks on different nodes, each one process part of pipeline ranks. \
        example : --pipeline-ranks-to-process 0 1 2 3 \
        default is None, means process all pipeline ranks")

    args = parser.parse_args()
    return args

"""
This tool can convert a checkpoint without virtual pipeline parallelism into one with virtual pipeline parallelism
  by increasing the virtual pipeline stage size.

(2025-05-30)
It now supports uneven pipeline mode, as well as cases where the number of layers in a pipeline stage is not divisible by the virtual pipeline degree.
  see arguments:
    --target-first-virtual-pipeline-num-layers-split
    --target-last-virtual-pipeline-num-layers-split
The above two parameters must either both be provided(or both be omitted), indicating that uneven pipeline mode is enabled
  and specifying the virtual pipeline layer distribution for the first and last pipeline stages.
  (this distribution may be even, but it still needs to be explicitly provided.)
This feature was introduced based on the following Pull Request.
  https://github.com/microsoft/ltp-megatron-lm/pull/27
  The model after converted needs to be loaded using a Megatron-LM framework that has this Pull Request applied.

Other model parallel parameters (tensor-parallel-size, pipeline-parallel-size, expert-parallel-size ...) remain unchanged.

Note that currently, all of the following configurations must be satisfied to be supported.
  tensor_parallel_size=1
  ckpt_format=torch
So the checkpoint for each iteration folder should look like this:

iter_0000005
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
    logger.info("args : {}".format(args))
    convert_checkpoint(args)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.DEBUG, 
        format = "[%(asctime)s][%(levelname)s] %(message)s",
        handlers = [logging.StreamHandler()])
    args = parse_arguments()
    main(args)
