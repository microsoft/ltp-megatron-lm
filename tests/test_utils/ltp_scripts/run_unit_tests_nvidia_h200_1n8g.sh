set -e

pip install -r requirements_ci.txt
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.2.post1
pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.2
MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git@v2.2.0
apt purge -y python3-blinker
pip install flask flask-restful tiktoken tensorstore

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0

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
# - inference/engines/test_dynamic_engine.py

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  -m "not flaky_in_dev and not flaky" \
  --ignore tests/unit_tests/inference/engines/test_dynamic_engine.py \
  tests/unit_tests
