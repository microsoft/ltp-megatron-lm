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
  --cov-report term
  --cov-branch
  --cov megatron
  --no-cov-on-fail
)

clear_previous_runs() {
    ps axu | grep python | awk -F' ' '{print "kill -9 "$2}' | bash
    sleep 10
}

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vs \
  ${PYTEST_COV_ARGS[@]} \
  tests/unit_tests
