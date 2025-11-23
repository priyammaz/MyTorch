#!/bin/bash

# Simple script to download/prepare data and train a ~500M param LLM!

# PATHS TO STUFF
EXPERIMENT_NAME="mytorch_llm_500m" # Name for the run for local checkpointing and WanbB
HF_CACHE_DIR="data/hf_cache" # Huggingface cache dir where temporary stuff will be stored (and deleted)
DOWNLOAD_PATH="data/FineWebEDU" # Where do you want to store very thing
PATH_TO_SAVE_TOKENIZER="nanochat_trainer/nanochat_tokenizer" # Where do you want to save your tokenizer.json
RAW_TEXT_DIRECTORY="$DOWNLOAD_PATH/raw_text" # where do you want to download raw parquet data files
TOKENIZED_DIRECTORY="$DOWNLOAD_PATH/tokenized" # where do you want to save pre-tokenized data
PRETRAIN_WORKING_DIRECTORY="work_dir/nanochat_pretrain" # Where to save pretraining checkpoints
MIDTRAIN_WORKING_DIRECTORY="work_dir/nanochat_midtrain" # Where to save midtraining checkpoints
SFT_WORKING_DIRECTORY="work_dir/nanochat_sft" # Where to save SFT checkpoints

### MODEL SHAPE (This is the config for a ~500M param LLM)
VOCAB_SIZE=65536 # 2**16 
CONTEXT_LENGTH=2048 # Total context this model will process (and data will be cut into)
NUM_BLOCKS=24 # Number of transformer blocks
EMBED_DIM=1280 # Embedding dimension 
NUM_Q_HEADS=20 # Number of Query heads
NUM_KV_HEADS=10 # Number of KV Heads (must evenly divide Q Heads for GQA)
MLP_RATIO=4 # MLP Ratio in Feed forward

### DATA CONFIG
CHINCHILLA_RATIO=20 # you can increase this to overtrain model on > chinchilla optimal
NUM_WORKERS=32 # Number of cpu workers for everything data related

### TRAINING CONFIG ###
PER_GPU_BATCH_SIZE=4 # Set to whatever doesn't OOM!
TARGET_TOKENS_PER_BATCH=524288 # Grad accumulation until we hit this tok/batch
MAX_LEARNING_RATE=0.0004 # Highest lr that we warmup to
MIN_LEARNING_RATE_RATIO=0.1 # Proportion of highest lr that we decay down to
WARMUP_RATIO=0.05 # What proportion of training we do warmup for
BETA1=0.9 # Adam Beta1
BETA2=0.95 # Adam Beta2 
WEIGHT_DECAY=0.1 # Adam Weight decay (non embedding params)
MAX_GRAD_NORM=1.0 # Max for grad clipping

# ===================================================================
# DATA/TOKENZIER PREP
# ===================================================================
# Create directories and set HF Home for dataset caching (can be deleted later)
mkdir -p $HF_CACHE_DIR
mkdir -p $RAW_TEXT_DIRECTORY
mkdir -p $TOKENIZED_DIRECTORY

export HF_HOME=$HF_CACHE_DIR

### DOWNLOAD SLICE OF FINEWEB ###
### With the default settings this will download 20 parquet files from 100BT split
### and will save a final 29 parquet files (about 55GB of data!)
python -m nanochat_trainer.scripts.download_fineweb_edu \
    --path_to_save $RAW_TEXT_DIRECTORY \
    --num_workers $NUM_WORKERS \
    --chinchilla_ratio $CHINCHILLA_RATIO \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --num_blocks $NUM_BLOCKS \
    --embed_dim $EMBED_DIM \
    --num_q_heads $NUM_Q_HEADS \
    --num_kv_heads $NUM_KV_HEADS \
    --mlp_ratio $MLP_RATIO

### TRAIN TOKENIZER ON FINEWEB ###
python -m nanochat_trainer.scripts.train_tokenizer \
    --comparison_tokenizer "gpt2" \
    --path_to_dataset $RAW_TEXT_DIRECTORY \
    --vocab_size $VOCAB_SIZE \
    --path_to_save_tokenizer $PATH_TO_SAVE_TOKENIZER

### Tokenize and Save Dataset 
python -m nanochat_trainer.scripts.prepare_fineweb \
    --path_to_tokenizer $PATH_TO_SAVE_TOKENIZER \
    --path_to_data $RAW_TEXT_DIRECTORY \
    --path_to_save $TOKENIZED_DIRECTORY \
    --num_workers $NUM_WORKERS \
    --max_seq_len $CONTEXT_LENGTH

### Delete Everything in Cache, Dont Need it Anymore ###
rm -r $HF_CACHE_DIR/*

### Delete the downloaded raw data, Dont need it Anymore ###
rm -r $RAW_TEXT_DIRECTORY

# ===================================================================
# PRETRAINING (the expensive part)
# ===================================================================
mytorchrun launch -m nanochat_trainer.scripts.pretrain_model \
    --work_dir $PRETRAIN_WORKING_DIRECTORY \
    --experiment_name $EXPERIMENT_NAME \
    --path_to_data $TOKENIZED_DIRECTORY \
    --chinchilla_ratio $CHINCHILLA_RATIO \
    --batch_size_per_gpu $PER_GPU_BATCH_SIZE \
    --tokens_per_batch $TARGET_TOKENS_PER_BATCH \
    --num_workers $NUM_WORKERS \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --num_blocks $NUM_BLOCKS \
    --embed_dim $EMBED_DIM \
    --num_q_heads $NUM_Q_HEADS \
    --num_kv_heads $NUM_KV_HEADS \
    --mlp_ratio $MLP_RATIO \
    --max_learning_rate $MAX_LEARNING_RATE \
    --min_learning_rate_ratio $MIN_LEARNING_RATE_RATIO \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --path_to_tokenizer $PATH_TO_SAVE_TOKENIZER