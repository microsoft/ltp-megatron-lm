set -ex

CURDIR=$(cd $(dirname $0); pwd)
cd $CURDIR

python main.py \
    --load-iteration-dir /path/to/src_checkpoints/iter_0000050 \
    --expert-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --save-iteration-dir /path/to/dst_checkpoints/iter_0000050 \
    --target-virtual-pipeline-model-parallel-size 2 \
    --num-max-processing-processes 8