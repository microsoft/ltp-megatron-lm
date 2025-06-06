set -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

TORCHRUN_ARGS=(
  --nproc_per_node 8
  --nnodes 1
  --node_rank 0
  --master_addr localhost
  --master_port 50326
)

PYTEST_COV_ARGS=(
  --cov-branch
  --cov megatron
  --cov-append
  --no-cov-on-fail
)

torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vs \
  ${PYTEST_COV_ARGS[@]} \
  tests/unit_tests
