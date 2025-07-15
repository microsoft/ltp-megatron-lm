set -e

pip install -r requirements_ci.txt
pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# CUDA envs
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NCCL envs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=eth0
export NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml

# Megatron-LM envs
# CRITICAL: 50, ERROR: 40, WARNING: 30, INFO: 20, DEBUG: 10, NOTSET: 0
export MEGATRON_LOGGING_LEVEL=20

# Disable permute fusion since it's not supported in nvcr.io/nvidia/pytorch:25.01-py3
sed -i "s/'moe_permute_fusion': True/'moe_permute_fusion': False/g" tests/numerical_tests/modules/test_moe_layer.py

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

result_dir="./numerical_test_results"

run_numerical_tests() {
  rm -rf ${result_dir}
  for x in {0..19}
  do
    mkdir -p ${result_dir}/${1}/${x}
    clear_previous_runs
    torchrun \
      ${TORCHRUN_ARGS[@]} \
      -m pytest -vxs \
      tests/numerical_tests/modules/test_${1}.py \
      --result-dir ${result_dir}/${1}/${x}
  done
  file_names=$(find ${result_dir} -type f -printf "%f\n" | sort | uniq)
  for name in ${file_names}
  do
    rm -rf calc_input_list.txt
    for x in {0..19}
    do
      echo "${result_dir}/${1}/${x}/${name}" >> calc_input_list.txt
    done
    python \
      tests/numerical_tests/utils/calc_module_mean_and_variance.py \
      --input-list calc_input_list.txt \
      --output-file ${result_dir}/${1}/${name}.mean_and_std.pt
  done
}

run_numerical_tests attention
run_numerical_tests bda
run_numerical_tests embedding
run_numerical_tests layer_norm
run_numerical_tests logits
run_numerical_tests loss
run_numerical_tests mlp
run_numerical_tests rope

TORCHRUN_ARGS=(
--nproc_per_node 8
--nnodes 1
--node_rank 0
--master_addr localhost
--master_port 50326
)

run_numerical_tests moe_layer
