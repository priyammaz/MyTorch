#!/bin/bash

USE_DISTRIBUTED=false
NUM_GPUS=1
TARGET=""
TRITON_AUTOTUNE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        owt|shakespeare)
            TARGET="$1"
            ;;
        --distributed)
            USE_DISTRIBUTED=true
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift
            ;;
        --triton_autotune)
            TRITON_AUTOTUNE=true
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Usage: $0 [owt|shakespeare] [--distributed] [--num_gpus N] [--triton_autotune]"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "$TARGET" ]]; then
    echo "Error: You must specify a training job: 'owt' or 'shakespeare'"
    echo "Usage: $0 [owt|shakespeare] [--distributed] [--num_gpus N] [--triton_autotune]"
    exit 1
fi

if [[ "$TRITON_AUTOTUNE" == true ]]; then
    export TRITON_FLASH_AUTOTUNE_MODE="max"
fi

if [[ "$USE_DISTRIBUTED" == true ]]; then
    CMD="python -m mytorch.distributed.launch --num_gpus ${NUM_GPUS} --training_script"
else
    CMD="python"
fi

case "$TARGET" in
    owt)
        echo "Starting OpenWebText training..."
        $CMD train_gpt2.py \
            --project_name gpt2-base-owt \
            --working_directory work_dir \
            --checkpoint_iterations 10000 \
            --always_save_checkpoint \
            --context_length 1024 \
            --model_size base \
            --dropout_p 0.0 \
            --fused \
            --path_to_data data/openwebtext \
            --train_iterations 600000 \
            --eval_interval 1000 \
            --eval_iterations 200 \
            --batch_size 64 \
            --gradient_accumulation_steps 4 \
            --max_lr 6e-4 \
            --min_lr 6e-5 \
            --warmup_steps 2000 \
            --weight_decay 0.1 \
            --max_grad_norm 1.0 \
            --beta1 0.9 \
            --beta2 0.95 \
            --mixed_precision \
            --log_iter 25 \
            --log_wandb
        ;;
    shakespeare)
        echo "Starting Shakespeare training..."
        $CMD train_gpt2.py \
            --project_name gpt2-small-shakespeare \
            --working_directory work_dir \
            --context_length 256 \
            --model_size small \
            --dropout_p 0.0 \
            --fused \
            --path_to_data data/shakespeare \
            --train_iterations 2500 \
            --eval_interval 1000 \
            --eval_iterations 200 \
            --batch_size 32 \
            --gradient_accumulation_steps 1 \
            --max_lr 1e-3 \
            --min_lr 1e-4 \
            --warmup_steps 500 \
            --weight_decay 0.1 \
            --max_grad_norm 1.0 \
            --beta1 0.9 \
            --beta2 0.95 \
            --mixed_precision \
            --log_iter 25
        ;;
esac