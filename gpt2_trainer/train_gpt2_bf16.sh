#!/bin/bash
### Short bfloat16 training run of GPT2 base (12 layers, 12 heads, 768 dim, 1024 context) on OpenWebText
### Requires cupy>=14 and `pip install ml_dtypes` for bfloat16 support
###
### Usage (from the repo root):
###   bash gpt2_trainer/train_gpt2_bf16.sh                          # 1 GPU (GPU 0)
###   GPUS=0,1 bash gpt2_trainer/train_gpt2_bf16.sh                 # 2 GPUs (DDP)
###   GPUS=1 bash gpt2_trainer/train_gpt2_bf16.sh                   # pick a single GPU
###   bash gpt2_trainer/train_gpt2_bf16.sh --train_iterations 200   # any train_gpt2.py arg overrides the defaults below
###
### Tokens per optimizer step = BATCH_SIZE * ACCUM_STEPS * num_gpus * 1024. ACCUM_STEPS defaults to
### keeping that at ~65k tokens regardless of the number of GPUs.

GPUS="${GPUS:-${GPU:-0}}"
PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-65536}"
MASTER_PORT="${MASTER_PORT:-13333}"

NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
ACCUM_STEPS="${ACCUM_STEPS:-$(( TOKENS_PER_STEP / (BATCH_SIZE * NUM_GPUS * 1024) ))}"
if [[ "$ACCUM_STEPS" -lt 1 ]]; then ACCUM_STEPS=1; fi

### Add PWD to Sys Directory to enable MyTorch import to resolve ###
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="$GPUS"

if [[ "$NUM_GPUS" -gt 1 ]]; then
    CMD="$PYTHON -m mytorch.distributed.launch --num_gpus $NUM_GPUS --master_addr 127.0.0.1 --master_port $MASTER_PORT"
else
    CMD="$PYTHON"
fi

echo "GPUs: $GPUS | micro-batch $BATCH_SIZE x accumulation $ACCUM_STEPS x $NUM_GPUS GPU(s) = $(( BATCH_SIZE * ACCUM_STEPS * NUM_GPUS * 1024 )) tokens/step"

$CMD gpt2_trainer/train_gpt2.py \
    --project_name gpt2-base-owt-bf16 \
    --working_directory work_dir \
    --path_to_data data/openwebtext \
    --model_size base \
    --context_length 1024 \
    --dropout_p 0.0 \
    --fused \
    --mixed_precision bf16 \
    --batch_size_per_gpu $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --train_iterations 1000 \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --beta1 0.9 \
    --beta2 0.95 \
    --log_iter 10 \
    --eval_interval 250 \
    --eval_iterations 20 \
    --checkpoint_iterations 1000000 \
    --num_workers 4 \
    --print_banner \
    "$@"
