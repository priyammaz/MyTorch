#!/bin/bash

# Simple script to download/prepare data and train a ~500M param LLM!

HF_CACHE_DIR="data/hf_cache"
DOWNLOAD_PATH="data/FineWebEDU"
PATH_TO_SAVE_TOKENIZER="nanochat_trainer/nanochat_tokenizer"
NUM_WORKERS=32
CONTEXT_LENGTH=2048

# ===================================================================
# Create directories and set HF Home for dataset caching (can be deleted later)
mkdir -p $HF_CACHE_DIR
mkdir -p "$DOWNLOAD_PATH/raw_text"
mkdir -p "$DOWNLOAD_PATH/tokenized"

export HF_HOME=$HF_CACHE_DIR

### DOWNLOAD SLICE OF FINEWEB ###
python -m nanochat_trainer.scripts.download_fineweb_edu \
    --num_model_parameters 500_000_000 \
    --path_to_save "$DOWNLOAD_PATH/raw_text" \
    --num_workers $NUM_WORKERS 

### TRAIN TOKENIZER ON FINEWEB ###
python -m nanochat_trainer.scripts.train_tokenizer \
    --comparison_tokenizer "gpt2" \
    --path_to_dataset "$DOWNLOAD_PATH/raw_text" \
    --vocab_size 65536 \
    --path_to_save_tokenizer $PATH_TO_SAVE_TOKENIZER

### Tokenize and Save Dataset 
python -m nanochat_trainer.scripts.prepare_fineweb \
    --path_to_tokenizer $PATH_TO_SAVE_TOKENIZER \
    --path_to_data "$DOWNLOAD_PATH/raw_text" \
    --path_to_save "$DOWNLOAD_PATH/tokenized" \
    --num_workers $NUM_WORKERS \
    --max_seq_len $CONTEXT_LENGTH

### Delete Everything in Cache, Dont Need it Anymore ###
rm -rf $HF_CACHE_DIR/*

### Delete the downloaded raw data, Dont need it Anymore ###
rm -rf "$DOWNLOAD_PATH/raw_text"
