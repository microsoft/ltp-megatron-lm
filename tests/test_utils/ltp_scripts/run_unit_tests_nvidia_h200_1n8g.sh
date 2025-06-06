set -e

pip install -r requirements_ci.txt
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install git+https://github.com/Dao-AILab/causal-conv1d.git@v1.2.2.post1
pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.2
MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git@v2.2.0
apt purge -y python3-blinker
pip install flask flask-restful tiktoken tensorstore

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
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

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  --ignore tests/unit_tests/data \
  --ignore tests/unit_tests/dist_checkpointing \
  --ignore tests/unit_tests/inference/engines/test_dynamic_engine.py \
  --ignore tests/unit_tests/models \
  --ignore tests/unit_tests/test_checkpointing.py \
  --ignore tests/unit_tests/test_parallel_state.py \
  --ignore tests/unit_tests/test_tokenizer.py \
  --ignore tests/unit_tests/transformer \
  tests/unit_tests

clear_previous_runs
disable_pattern="not test_preprocess_data_bert"
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  -k "${disable_pattern}" \
  tests/unit_tests/data

clear_previous_runs
disable_pattern="not test_dp_sharding and "
disable_pattern+="not test_memory_usage and "
disable_pattern+="not test_remove_sharded_tensors"
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  -k "${disable_pattern}" \
  tests/unit_tests/dist_checkpointing

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  --deselect "tests/unit_tests/models/test_bert_model.py::TestBertModelAttentionDimensions::test_transformer_engine_version_1_7_to_1_10_rng_error" \
  --deselect "tests/unit_tests/models/test_t5_model.py::TestT5Model::test_forward_output_encoder_hidden_only" \
  --deselect "tests/unit_tests/models/test_t5_model.py::TestT5Model::test_forward_with_encoder_hidden_states" \
  --deselect "tests/unit_tests/models/test_t5_model.py::TestT5Model::test_post_process_forward" \
  tests/unit_tests/models

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  --deselect "tests/unit_tests/test_checkpointing.py::test_load_checkpoint[torch]" \
  --deselect "tests/unit_tests/test_checkpointing.py::test_save_checkpoint[torch]" \
  --deselect "tests/unit_tests/test_checkpointing.py::test_save_checkpoint[torch_dcp]" \
  tests/unit_tests/test_checkpointing.py

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  --deselect "tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp3-2]" \
  --deselect "tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp4-2]" \
  --deselect "tests/unit_tests/test_parallel_state.py::test_different_initialize_order_unconsistency[src_tp_pp5-2]" \
  tests/unit_tests/test_parallel_state.py

clear_previous_runs
disable_pattern="not test_gpt2_tiktok_tokenizer"
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  -k "${disable_pattern}" \
  tests/unit_tests/test_tokenizer.py

clear_previous_runs
torchrun \
  ${TORCHRUN_ARGS[@]} \
  -m pytest -vxs \
  ${PYTEST_COV_ARGS[@]} \
  --deselect "tests/unit_tests/transformer/test_retro_attention.py::TestRetroAttention::test_gpu_forward" \
  --deselect "tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_gpu_forward" \
  --deselect "tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_fused_rope_gpu_forward" \
  --deselect "tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_checkpointed_gpu_forward" \
  tests/unit_tests/transformer
