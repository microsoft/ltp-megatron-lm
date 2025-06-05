torchrun \
  --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 50326 \
  -m pytest -v \
  tests/unit_tests
