set -e

pip install -r requirements_ci.txt
pip install mock

# ROCm envs
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HIP_FORCE_DEV_KERNARG=1
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM=1

# RCCL envs
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export RCCL_MSCCL_ENABLE=0

# TransformerEngine envs
export NVTE_FLASH_ATTN=0
export NVTE_CK_USES_BWD_V3=0
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_UNFUSED_ATTN=0

# Megatron-LM envs
# CRITICAL: 50, ERROR: 40, WARNING: 30, INFO: 20, DEBUG: 10, NOTSET: 0
export MEGATRON_LOGGING_LEVEL=20

TORCHRUN_ARGS=(
--nproc_per_node 1
--nnodes 1
--node_rank 0
--master_addr localhost
--master_port 50326
)

clear_previous_runs() {
  ps axu | grep '[p]ython' | awk '{print $2}' | xargs -r -n 1 kill -9 2>/dev/null || true
  sleep 10
}

result_dir="./numerical_test_results/amd_mi300x"
rm -rf ${result_dir}

run_numerical_tests() {
  # Get raw module test results
  for x in {0..19}
  do
    mkdir -p ${result_dir}/${1}/module_test/${x}
    clear_previous_runs
    torchrun \
      ${TORCHRUN_ARGS[@]} \
      -m pytest -vxs \
      tests/numerical_tests/modules/test_${1}.py \
      --result-dir ${result_dir}/${1}/module_test/${x}
  done
  # Calculate module mean and std
  file_names=$(find ${result_dir}/${1}/module_test -type f -printf "%f\n" | sort | uniq)
  mkdir -p ${result_dir}/${1}/module_mean_and_std
  for name in ${file_names}
  do
    for x in {0..19}
    do
      echo "${result_dir}/${1}/module_test/${x}/${name}" >> ${result_dir}/${1}/module_mean_and_std/input_list.txt
    done
    python \
      tests/numerical_tests/utils/module_mean_and_std.py \
      --input-list ${result_dir}/${1}/module_mean_and_std/input_list.txt \
      --output-mean-file ${result_dir}/${1}/module_mean_and_std/${name}.mean.pt \
      --output-std-file ${result_dir}/${1}/module_mean_and_std/${name}.std.pt
    rm ${result_dir}/${1}/module_mean_and_std/input_list.txt
  done
  # Calculate intra-module similarity
  mkdir -p ${result_dir}/${1}/module_similarity
  for name in ${file_names}
  do
    for x in {0..19}
    do
      for y in {0..19}
      do
        if [ "$x" -lt "$y" ]; then
          python \
            tests/numerical_tests/utils/module_similarity.py \
            --stats-a ${result_dir}/${1}/module_test/${x}/${name} \
            --stats-b ${result_dir}/${1}/module_test/${y}/${name} \
            --output-file ${result_dir}/${1}/module_similarity/${name}.${x}-${y}.json
        fi
      done
    done
  done
  # Remove raw module test results
  rm -rf ${result_dir}/${1}/module_test
}

run_numerical_tests attention
run_numerical_tests bda
run_numerical_tests embedding
run_numerical_tests layer_norm
run_numerical_tests logits
run_numerical_tests mlp
run_numerical_tests rope

unset NVTE_FLASH_ATTN
unset NVTE_CK_USES_BWD_V3
unset NVTE_FUSED_ATTN
unset NVTE_FUSED_ATTN_CK
unset NVTE_FUSED_ATTN_AOTRITON
unset NVTE_UNFUSED_ATTN

run_numerical_tests loss

TORCHRUN_ARGS=(
--nproc_per_node 8
--nnodes 1
--node_rank 0
--master_addr localhost
--master_port 50326
)

run_numerical_tests moe_layer
