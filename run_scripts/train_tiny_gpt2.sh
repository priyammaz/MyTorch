DATA_LINK="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

python train_tiny_gpt2.py \
    --context_length 256 \
    --embed_dim 384 \
    --num_heads 6 \
    --num_blocks 6 \
    --dropout 0.0 \
    --mlp_ratio 4 \
    --data_path $DATA_LINK \
    --train_iterations 5000 \
    --warmup_steps 500 \
    --batch_size 32 \
    --max_lr "3e-4" \
    --min_lr "5e-5" \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --gradient_accumulation 1 \
    --log_iter 100 \
    --gen_iter 500 \
    --gen_length 256 \
    --save_path "work_dir/char_shakespear_gpt2.safetensors"