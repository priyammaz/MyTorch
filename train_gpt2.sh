#!/bin/bash

NUM_GPUS=1
TARGET=""
TRITON_AUTOTUNE=false
CUPYX_DISTRIBUTED_HOST="127.0.0.1"
CUPYX_DISTRIBUTED_PORT="13333"
FUSED=false
MIXED_PRECISION=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        owt|shakespeare)
            TARGET="$1"
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift
            ;;
        --host)
            CUPYX_DISTRIBUTED_HOST="$2"
            shift
            ;;
        --port)
            CUPYX_DISTRIBUTED_PORT="$2"
            shift
            ;;
        --triton_autotune)
            TRITON_AUTOTUNE=true
            ;;
        --fused)
            FUSED=true
            ;;
        --mixed_precision)
            MIXED_PRECISION=true
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Usage: $0 [owt|shakespeare] [--num_gpus N] [--host HOST] [--port PORT] [--triton_autotune] [--fused] [--mixed_precision]"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "$TARGET" ]]; then
    echo "Error: You must specify a training job: 'owt' or 'shakespeare'"
    exit 1
fi

if [[ "$TRITON_AUTOTUNE" == true ]]; then
    export TRITON_FLASH_AUTOTUNE_MODE="max"
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
    DISTRIBUTED_ARGS="--num_gpus ${NUM_GPUS} --master_addr ${CUPYX_DISTRIBUTED_HOST} --master_port ${CUPYX_DISTRIBUTED_PORT}"
    CMD="python -m mytorch.distributed.launch ${DISTRIBUTED_ARGS} --training_script"
else
    CMD="python"
fi

EXTRA_ARGS=""
if [[ "$FUSED" == true ]]; then
    EXTRA_ARGS+=" --fused"
fi
if [[ "$MIXED_PRECISION" == true ]]; then
    EXTRA_ARGS+=" --mixed_precision"
fi

case "$TARGET" in
    owt)
        $CMD train_gpt2.py \
            --project_name gpt2-base-owt \
            --working_directory work_dir \
            --checkpoint_iterations 10000 \
            --always_save_checkpoint \
            --context_length 1024 \
            --model_size base \
            --dropout_p 0.0 \
            --path_to_data data/openwebtext \
            --train_iterations 600000 \
            --eval_interval 1000 \
            --eval_iterations 200 \
            --batch_size 384 \
            --gradient_accumulation_steps 12 \
            --max_lr 6e-4 \
            --min_lr 6e-5 \
            --warmup_steps 2000 \
            --weight_decay 0.1 \
            --max_grad_norm 1.0 \
            --beta1 0.9 \
            --beta2 0.95 \
            --log_iter 25 \
            $EXTRA_ARGS
        ;;
    shakespeare)
        $CMD train_gpt2.py \
            --project_name gpt2-small-shakespeare \
            --working_directory work_dir \
            --context_length 256 \
            --model_size small \
            --dropout_p 0.0 \
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
            --log_iter 25 \
            $EXTRA_ARGS
        ;;
esac