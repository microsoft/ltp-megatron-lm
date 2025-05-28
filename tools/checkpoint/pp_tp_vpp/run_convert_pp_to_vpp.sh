set -ex

CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR

python main.py \
    --load-iteration-dir /path/to/checkpoint_load/iter_0000050/ \
    --expert-model-parallel-size 8 \
    --pipeline-model-parallel-size 2 \
    --save-iteration-dir /path/to/checkpoint_save/iter_0000050/ \
    --target-virtual-pipeline-model-parallel-size 2 \
    --num-max-processing-processes 4
