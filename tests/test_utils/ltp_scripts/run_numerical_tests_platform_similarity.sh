set -e

stats_dir_a="./numerical_test_results/nvidia_h200"
stats_dir_b="./numerical_test_results/amd_mi300x"
result_dir="./numerical_test_results/nvidia_h200_vs_amd_mi300x"

module_similarity() {
  file_names=$(find ${stats_dir_a}/${1}/module_mean_and_std -type f -printf "%f\n" | sort | uniq)
  mkdir -p ${result_dir}/${1}/module_similarity
  for name in ${file_names}
  do
    python \
      tests/numerical_tests/utils/module_similarity.py \
      --stats-a ${stats_dir_a}/${1}/module_mean_and_std/${name} \
      --stats-b ${stats_dir_b}/${1}/module_mean_and_std/${name} \
      --output-file ${result_dir}/${1}/module_similarity/${name}.json
  done
}

module_similarity attention
module_similarity bda
module_similarity embedding
module_similarity layer_norm
module_similarity logits
module_similarity loss
module_similarity mlp
module_similarity moe_layer
module_similarity rope
