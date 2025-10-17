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

# Cache tokenizer
python cache_sigma_v2_tokenizer.py

torchrun \
  --nproc_per_node 8 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr msrveeorz00000G \
  --master_port 50005 \
  pretrain_gpt.py \
  --num-layers 28 \
  --hidden-size 2048 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --rotary-scaling-factor 1.0 \
  --num-query-groups 4 \
  --group-query-attention \
  --qk-layernorm \
  --kv-channels 128 \
  --num-attention-heads 16 \
  --normalization RMSNorm \
  --norm-epsilon 1e-06 \
  --use-rotary-position-embeddings \
  --position-embedding-type rope \
  --rotary-base 10000 \
  --rotary-percent 1.0 \
  --swiglu \
  --untie-embeddings-and-output-weights \
  --disable-bias-linear \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --cross-entropy-label-smoothing 0.0 \
  --num-experts 96 \
  --moe-layer-freq "([1]*28)" \
  --moe-ffn-hidden-size 1024 \
  --moe-router-load-balancing-type global_batch_loss \
  --moe-router-score-function sigmoid \
  --moe-aux-loss-score-function softmax \
  --moe-router-topk 6 \
  --moe-router-enable-expert-bias \
  --moe-router-bias-update-rate 1e-3 \
  --moe-aux-loss-coeff 1e-3 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --train-samples 3200 \
  --lr-decay-style WSD \
  --lr-wsd-decay-style linear \
  --lr-decay-samples 6400 \
  --lr-warmup-samples 800 \
  --lr-wsd-decay-samples 5600 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 5e-10 \
  --init-method-std 0.0125 \
  --output-layer-init-method-normal \
  --output-layer-init-method-normal-std 0.0125 \
  --clip-grad 1.0 \
  --bf16 \
  --use-flash-attn \
  --lr 2e-5 \
  --min-lr 1e-5 \
  --use-distributed-optimizer \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --log-throughput \
  --log-timers-to-tensorboard \
  --log-progress \
  --log-params-norm \
  --log-num-zeros-in-grad \
  --log-interval 1 \
  --eval-iters 10 \
  --eval-interval 9999999999 \
  --moe-per-layer-logging \
  --save-interval 1 \
  --save /mnt/checkpoints_ruizhe/OpenPAI-Pretrain-17BA2B-0515-init0_02-add-kaiming-and-rm-outputinit-dist \
  --ckpt-format torch_dist \
  --auto-detect-ckpt-format \
  --ckpt-isolated-save \
  --load /mnt/checkpoints_ruizhe/OpenPAI-Pretrain-17BA2B-0515-init0_02-add-kaiming-and-rm-outputinit \
  --no-load-optim \
  --transformer-impl transformer_engine \
  --moe-shared-expert-overlap \
  --moe-grouped-gemm \
  --moe-use-legacy-grouped-gemm \
  --moe-token-dispatcher-type alltoall \
  --moe-router-dtype fp32 \
  --manual-gc \
  --manual-gc-interval 20 \
  --manual-gc-after-checkpoint-interval 20 \
  --no-mmap-bin-files \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --ddp-bucket-size 4294967296 \
  --no-async-tensor-model-parallel-allreduce \
  --attention-backend fused \
  --data-args-path data_args_weight.txt \
  --dataloader-type single \
  --split 1000,0,0 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model sigma_v2_tokenizer \
  --tokenizer-huggingface-trust-remote-code \
  --seed 42 \
  --data-cache-path index_cache \
  --distributed-timeout-minutes 150 \
  --num-workers 1 \
  --num-dataset-builder-threads 1 \
  --scale-shuffle \
  --dataset-reset-key data_args_weight.txt \
  --no-gradient-accumulation-fusion \
  |& tee run_17BA2B_load_torch_save_dist.log
