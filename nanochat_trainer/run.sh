#!/bin/bash

# Simple script to download/prepare data and train a ~500M param LLM!

# PATHS TO STUFF
HF_CACHE_DIR="data/hf_cache"
DOWNLOAD_PATH="data/FineWebEDU"
PATH_TO_SAVE_TOKENIZER="nanochat_trainer/nanochat_tokenizer"

### MODEL SHAPE (This is the config for a )
VOCAB_SIZE=65536 # 2**16 
CONTEXT_LENGTH=2048
NUM_BLOCKS=20
EMBED_DIM=1280
NUM_Q_HEADS=10
NUM_KV_HEADS=10
MLP_RATIO=4

### DATA CONFIG
CHINCHILLA_RATIO=20 # you can increase this to overtrain model on > chinchilla optimal
NUM_WORKERS=32

### TRAINING CONFIG ###
PRETRAIN_WORKING_DIRECTORY="work_dir/nanochat_pretrain"
MIDTRAIN_WORKING_DIRECTORY="work_dir/nanochat_midtrain"
SFT_WORKING_DIRECTORY="work_dir/nanochat_sft"

# ===================================================================
# DATA/TOKENZIER PREP
# ===================================================================
# Create directories and set HF Home for dataset caching (can be deleted later)
mkdir -p $HF_CACHE_DIR
mkdir -p "$DOWNLOAD_PATH/raw_text"
mkdir -p "$DOWNLOAD_PATH/tokenized"

export HF_HOME=$HF_CACHE_DIR

# ### DOWNLOAD SLICE OF FINEWEB ###
# ### With the default settings this will download 20 parquet files from 100BT split
# ### and will save a final 29 parquet files (about 55GB of data!)
# python -m nanochat_trainer.scripts.download_fineweb_edu \
#     --path_to_save "$DOWNLOAD_PATH/raw_text" \
#     --num_workers $NUM_WORKERS \
#     --chinchilla_ratio $CHINCHILLA_RATIO \
#     --vocab_size $VOCAB_SIZE \
#     --context_length $CONTEXT_LENGTH \
#     --num_blocks $NUM_BLOCKS \
#     --embed_dim $EMBED_DIM \
#     --num_q_heads $NUM_Q_HEADS \
#     --num_kv_heads $NUM_KV_HEADS \
#     --mlp_ratio $MLP_RATIO

# ### TRAIN TOKENIZER ON FINEWEB ###
# python -m nanochat_trainer.scripts.train_tokenizer \
#     --comparison_tokenizer "gpt2" \
#     --path_to_dataset "$DOWNLOAD_PATH/raw_text" \
#     --vocab_size 65536 \
#     --path_to_save_tokenizer $PATH_TO_SAVE_TOKENIZER

# ### Tokenize and Save Dataset 
# python -m nanochat_trainer.scripts.prepare_fineweb \
#     --path_to_tokenizer $PATH_TO_SAVE_TOKENIZER \
#     --path_to_data "$DOWNLOAD_PATH/raw_text" \
#     --path_to_save "$DOWNLOAD_PATH/tokenized" \
#     --num_workers $NUM_WORKERS \
#     --max_seq_len $CONTEXT_LENGTH

# ### Delete Everything in Cache, Dont Need it Anymore ###
# rm -r $HF_CACHE_DIR/*

# ### Delete the downloaded raw data, Dont need it Anymore ###
# rm -r "$DOWNLOAD_PATH/raw_text"

# ===================================================================
# PRETRAINING (the expensive part)
# ===================================================================
# python -m nanochat_trainer.scripts.pretrain_model
python -m nanochat_trainer.core.pipeline