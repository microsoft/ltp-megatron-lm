torchrun \
  --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 50326 \
  -m pytest -xv \
  --cov-report=term \
  --cov-branch \
  --cov=megatron/core \
  --no-cov-on-fail \
  tests/unit_tests
