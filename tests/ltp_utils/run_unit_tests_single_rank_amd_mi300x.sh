export WORLD_SIZE=1
export LOCAL_RANK=0

# The test_preprocess_data_bert case hangs with the following error due to
# security fix in latest nltk (https://github.com/nltk/nltk/issues/3266):
# 
#   File "/opt/conda/envs/py_3.10/lib/python3.10/site-packages/nltk/app/wordnet_app.py", line 664, in find_class
#     raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")
# _pickle.UnpicklingError: global 'copy_reg._reconstructor' is forbidden
# 
# Other deselected test cases hang with segmentation fault.

pytest -v \
  --deselect "tests/unit_tests/data/test_preprocess_data.py::test_preprocess_data_bert" \
  --deselect "tests/unit_tests/dist_checkpointing/test_torch_dist.py::TestCPUTensors::test_cpu_tensors_dont_take_too_much_space" \
  --deselect "tests/unit_tests/dist_checkpointing/test_serialization.py::TestSerialization::test_single_process_save_load" \
  --deselect "tests/unit_tests/dist_checkpointing/models/test_mamba.py::TestMambaReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp2-dest_tp_pp_exp2-False]" \
  --deselect "tests/unit_tests/dist_checkpointing/models/test_mamba.py::TestMambaReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp3-dest_tp_pp_exp3-False]" \
  --deselect "tests/unit_tests/dist_checkpointing/models/test_mamba.py::TestMambaReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp8-dest_tp_pp_exp8-True]" \
  --deselect "tests/unit_tests/dist_checkpointing/models/test_mamba.py::TestMambaReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp_exp9-dest_tp_pp_exp9-True]" \
  --deselect "tests/unit_tests/dist_checkpointing/models/test_mamba.py::TestMambaReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp_exp10-dest_tp_pp_exp10-True]" \
  --deselect "tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_extra_state[te_sequential-te_grouped-src_tp_pp_exp0-dest_tp_pp_exp0]" \
  --deselect "tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_extra_state[te_sequential-te_grouped-src_tp_pp_exp2-dest_tp_pp_exp2]" \
  --deselect "tests/unit_tests/dist_checkpointing/models/test_moe_experts.py::TestExpertLayerReconfiguration::test_sequential_grouped_mlp_extra_state[te_grouped-te_sequential-src_tp_pp_exp2-dest_tp_pp_exp2]" \
  --deselect "tests/unit_tests/post_training/test_modelopt_module_spec.py::TestModelOptMambaModel::test_sharded_state_dict_restore" \
  tests/unit_tests
