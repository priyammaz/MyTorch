#!/bin/bash

NUM_GPUS=1
TARGET=""
TRITON_AUTOTUNE=false
CUPYX_DISTRIBUTED_HOST="127.0.0.1"
CUPYX_DISTRIBUTED_PORT="13333"
FUSED=false
MIXED_PRECISION=false
DLPACK_DISABLE=false
LOG_WANDB=false

### Add PWD to Sys Directory to enable MyTorch import to resolve ###
export PYTHONPATH="$(pwd):$PYTHONPATH"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)
            NUM_GPUS="$1"
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
        --disable_dlpack)
            DLPACK_DISABLE=true
            ;;
        --log_wandb)
            LOG_WANDB=true
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Usage: $0 [--num_gpus N] [--host HOST] [--port PORT] [--triton_autotune] [--fused] [--mixed_precision] [--disable_dlpack] [--log_wandb]"
            exit 1
            ;;
    esac
    shift
done

if [[ "$TRITON_AUTOTUNE" == true ]]; then
    export TRITON_AUTOTUNE_MODE="max"
fi
if [[ "$DLPACK_DISABLE" == true ]]; then
    export DLPACK_DISABLE="true"
fi

if [[ "$NUM_GPUS" -gt 1 ]]; then
    DISTRIBUTED_ARGS="--num_gpus ${NUM_GPUS} --master_addr ${CUPYX_DISTRIBUTED_HOST} --master_port ${CUPYX_DISTRIBUTED_PORT}"
    CMD="python -m mytorch.distributed.launch ${DISTRIBUTED_ARGS}"
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
if [[ "$LOG_WANDB" == true ]]; then
    EXTRA_ARGS+=" --log_wandb"
fi

$CMD gpt2_trainer/finetune_gpt2_base_shakespeare.py \
    --project_name gpt2-base-ft-shakespeare \
    --working_directory work_dir \
    --load_from_experiment "gpt2-base-owt" \
    --dropout_p 0.1 \
    --path_to_data data/shakespeare_gpt2 \
    --num_layers_train 6 \
    --train_iterations 500 \
    --eval_interval 100 \
    --eval_iterations 10 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --max_lr 5e-4 \
    --min_lr 1e-4 \
    --warmup_steps 50 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --beta1 0.9 \
    --beta2 0.95 \
    --log_iter 25 \
    $EXTRA_ARGS
