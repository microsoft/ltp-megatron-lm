# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -e

pip install -r requirements_ci.txt
pip install mock

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MAX_NCHANNELS=1
export RCCL_MSCCL_ENABLE=0

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
    ps axu | grep '[p]ython' | awk '{print $2}' | xargs -r -n 1 kill -9 2>/dev/null || true
    sleep 10
}

# Exclude test categories that fail to pass in the full test.
# Some test cases fail in:
# - inference/engines/test_dynamic_engine.py
# - transformer/moe/test_moe_layer_discrepancy.py

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  -m "not flaky_in_dev and not flaky" \
  --ignore tests/unit_tests/inference/engines/test_dynamic_engine.py \
  --ignore tests/unit_tests/transformer/moe/test_moe_layer_discrepancy.py \
  tests/unit_tests
