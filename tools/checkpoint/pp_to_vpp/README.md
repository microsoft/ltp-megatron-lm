# pp_to_vpp
## description
This tool can convert a language model checkpoint without virtual pipeline parallelism into one with virtual pipeline parallelism by increasing the virtual pipeline stage size.

Other model parallel parameters (tensor-parallel-size, pipeline-parallel-size, expert-parallel-size ...) remain unchanged.

---

**(2025-05-30)** It now supports uneven pipeline mode, as well as cases where the number of layers in a pipeline stage is not divisible by the virtual pipeline degree.

see arguments:
```
--target-first-virtual-pipeline-num-layers-split
--target-last-virtual-pipeline-num-layers-split
```
The above two parameters must either both be provided(or both be omitted), indicating that uneven pipeline mode is enabled
  and specifying the virtual pipeline layer distribution for the first and last pipeline stages(this distribution may be even, but it still needs to be explicitly provided).
  
This feature was introduced based on the following Pull Request.

  https://github.com/microsoft/ltp-megatron-lm/pull/27
  
  The model after converted needs to be loaded using a Megatron-LM framework that has this Pull Request applied.
  
---

**Currently, tests have been conducted on the DeepSeek(v2, v3) and Mixtral models.**

Note that currently, all of the following configurations must be satisfied to be supported.
  tensor_parallel_size=1
  ckpt_format=torch
so the checkpoint for each iteration folder should look like this:
```
iter_0000050
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
```


## how to use
you can modify run_convert_pp_to_vpp.sh and launch it as an example
```
usage: main.py [-h] --load-iteration-dir LOAD_ITERATION_DIR --expert-model-parallel-size EXPERT_MODEL_PARALLEL_SIZE --pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE
               --save-iteration-dir SAVE_ITERATION_DIR --target-virtual-pipeline-model-parallel-size TARGET_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE
               [--target-first-virtual-pipeline-num-layers-split TARGET_FIRST_VIRTUAL_PIPELINE_NUM_LAYERS_SPLIT [TARGET_FIRST_VIRTUAL_PIPELINE_NUM_LAYERS_SPLIT ...]]
               [--target-last-virtual-pipeline-num-layers-split TARGET_LAST_VIRTUAL_PIPELINE_NUM_LAYERS_SPLIT [TARGET_LAST_VIRTUAL_PIPELINE_NUM_LAYERS_SPLIT ...]]
               [--num-max-processing-processes NUM_MAX_PROCESSING_PROCESSES] [--pipeline-ranks-to-process PIPELINE_RANKS_TO_PROCESS]

convert a non-virtual pipeline checkpoint to virtual pipeline checkpoint

options:
  -h, --help            show this help message and exit
  --load-iteration-dir LOAD_ITERATION_DIR
                        iteration folder of source model checkpoint
  --expert-model-parallel-size EXPERT_MODEL_PARALLEL_SIZE
                        ep_size of original model and the target model
  --pipeline-model-parallel-size PIPELINE_MODEL_PARALLEL_SIZE
                        physical pp_size of original model and the target model
  --save-iteration-dir SAVE_ITERATION_DIR
                        iteration folder of target model checkpoint, need to be empty if existed
  --target-virtual-pipeline-model-parallel-size TARGET_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE
                        vpp_size of target model
  --target-first-virtual-pipeline-num-layers-split TARGET_FIRST_VIRTUAL_PIPELINE_NUM_LAYERS_SPLIT [TARGET_FIRST_VIRTUAL_PIPELINE_NUM_LAYERS_SPLIT ...]
                        only used in uneven pipeline mode, virtual pipeline split of the first stage
  --target-last-virtual-pipeline-num-layers-split TARGET_LAST_VIRTUAL_PIPELINE_NUM_LAYERS_SPLIT [TARGET_LAST_VIRTUAL_PIPELINE_NUM_LAYERS_SPLIT ...]
                        only used in uneven pipeline mode, virtual pipeline split of the last stage
  --num-max-processing-processes NUM_MAX_PROCESSING_PROCESSES
                        the maximum number of processing processes used by this script, increasing this value can speed up model conversion(but the final bottleneck may be disk
                        bandwidth), it will also consume more CPU memory.
  --pipeline-ranks-to-process PIPELINE_RANKS_TO_PROCESS
                        pipeline rank list to process using this script, to accelerate converting user can launch multiple tasks on different nodes, each one process part of pipeline
                        ranks. example : --pipeline-ranks-to-process [0,1,2,3] default is None, means process all pipeline ranks
```

## examples
1) The target model has virtual_pipeline_size=2, and uses 4 processes in parallel.
```
python main.py \
    --load-iteration-dir /path/to/src_checkpoints/iter_0000050 \
    --expert-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --save-iteration-dir /path/to/dst_checkpoints/iter_0000050 \
    --target-virtual-pipeline-model-parallel-size 2 \
    --num-max-processing-processes 4
```

2) Convert the checkpoints generated by pipeline ranks [0,1,2,3] on node 1, and convert the checkpoints generated by pipeline ranks [4,5,6,7] on node 2. (in cases where memory is limited on a single node)
```
# node1 :
python main.py \
    --load-iteration-dir /path/to/src_checkpoints/iter_0000050 \
    --expert-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --save-iteration-dir /path/to/dst_checkpoints/iter_0000050 \
    --target-virtual-pipeline-model-parallel-size 2 \
    --num-max-processing-processes 4 \
    --pipeline-ranks-to-process [0,1,2,3]

# node2:
python main.py \
    --load-iteration-dir /path/to/src_checkpoints/iter_0000050 \
    --expert-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --save-iteration-dir /path/to/dst_checkpoints/iter_0000050 \
    --target-virtual-pipeline-model-parallel-size 2 \
    --num-max-processing-processes 4 \
    --pipeline-ranks-to-process [4,5,6,7]
```

3) convert a model with uneven pipeline mode, which was saved by Megatron-LM with arguments
```
--decoder-first-pipeline-num-layers 8
--decoder-last-pipeline-num-layers 7
```

```
# suppose pipeline_parallel_size=4, the model contains 31 layers in total, the layers distribution for each pipeline stages is [8, 8, 8, 7]
# now we use this model to inscrease virtual pipeline size to 2,
#   the layer split in first pipeline stage is [4, 4] and the layer split in last pipeline stage is [4, 3]
#         vpp0          vpp1
# pp0  0, 1, 2, 3    16,17,18,19
# pp1  4, 5, 6, 7    20,21,22,23
# pp2  8, 9,10,11    24,25,26,27
# pp3 12,13,14,15    28,29,30

python main.py \
    --load-iteration-dir /path/to/src_checkpoints/iter_0000050 \
    --save-iteration-dir /path/to/dst_checkpoints/iter_0000050 \
    --expert-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --target-virtual-pipeline-model-parallel-size 2 \
    --target-first-virtual-pipeline-num-layers-split 4 4 \
    --target-last-virtual-pipeline-num-layers-split 4 3 \
    --num-max-processing-processes 8
```

Some training logs from the tests are available in the **logs** directory for review.

## NOTE
It's also possible to continue training by loading only the model weights without loading the optimizer state (add **--no-load-optim** argument when launch Megatron-LM, which will reset the optimizer), though performance may recover after training for a few more iterations.


