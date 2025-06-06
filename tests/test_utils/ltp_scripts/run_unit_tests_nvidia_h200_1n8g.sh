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

clear_previous_runs() {
    ps axu | grep python | awk -F' ' '{print "kill -9 "$2}' | bash
    sleep 10
}

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vs \
  ${PYTEST_COV_ARGS[@]} \
  --ignore tests/unit_tests/data \
  tests/unit_tests

clear_previous_runs
disable_pattern=""
disable_pattern+="not test_preprocess_data_bert"
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vs \
  ${PYTEST_COV_ARGS[@]} \
  -k "${disable_pattern}" \
  tests/unit_tests/data
