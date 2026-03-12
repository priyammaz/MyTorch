#!/bin/bash

# Identical to the train_nanochat_500m.sh, except now we do a 1B model to really push the limits
# of our MyTorch system! If you run this after train_nanochat_500m.sh and leave the data paths the same
# it will just download the new parquet files we need and then retokenize and resave the data for pretraining!

### Set These To Where You Want to Save Your Stuff! ###
EXPERIMENT_NAME="mytorch_llm_1B" # Name for the run for local checkpointing and WandB
DATA_DIRECTORY="data" # Where do you want all data related stuff to be stored?
WORKING_DIRECTORY="work_dir" # Where do you want checkpoints to be stored?

### ALL PATHS ###
EXPERIMENT_WORKING_DIRECTORY="$WORKING_DIRECTORY/$EXPERIMENT_NAME" # Folder for this experiment in our working directory
HF_CACHE_DIR="$DATA_DIRECTORY/hf_cache" # Huggingface cache dir where temporary stuff will be stored (and deleted)
DOWNLOAD_FINEWEB_PATH="$DATA_DIRECTORY/FineWebEDU" # Where do you want to store very thing
DOWNLOAD_TASKS_PATH="$DATA_DIRECTORY/tasks"
RAW_TEXT_DIRECTORY="$DOWNLOAD_FINEWEB_PATH/raw_text" # where do you want to download raw parquet data files
TOKENIZED_DIRECTORY="$DOWNLOAD_FINEWEB_PATH/tokenized" # where do you want to save pre-tokenized data
PRETRAIN_WORKING_DIRECTORY="$EXPERIMENT_WORKING_DIRECTORY/nanochat_pretrain" # Where to save pretraining checkpoints
MIDTRAIN_WORKING_DIRECTORY="$EXPERIMENT_WORKING_DIRECTORY/nanochat_midtrain" # Where to save midtraining checkpoints
SFT_WORKING_DIRECTORY="$EXPERIMENT_WORKING_DIRECTORY/nanochat_sft" # Where to save SFT checkpoints
PATH_TO_SAVE_TOKENIZER=$EXPERIMENT_WORKING_DIRECTORY # Where do you want to save your tokenizer.json (we can just store in work dir for this experiment!)

### MODEL SHAPE (This is the config for a ~1B param LLM)
VOCAB_SIZE=65536 # 2**16 
CONTEXT_LENGTH=2048 # Total context this model will process (and data will be cut into)
NUM_BLOCKS=28 # Number of transformer blocks
EMBED_DIM=1536 # Embedding dimension 
NUM_Q_HEADS=24 # Number of Query heads
NUM_KV_HEADS=6 # Number of KV Heads (must evenly divide Q Heads for GQA) we use a 4x ratio like Llama
MLP_RATIO=4 # MLP Ratio in Feed forward

### DATA CONFIG ###
CHINCHILLA_RATIO=30 # 20 is chinchilla optimal, but lets over train a little more here to squeeze out some more performance (with diminishing returns)!
NUM_WORKERS=32 # Number of cpu workers for everything data related

### TRAINING CONFIG ###
PER_GPU_BATCH_SIZE=2 # Set to whatever doesn't OOM!
TARGET_TOKENS_PER_BATCH=524288 # Grad accumulation until we hit this tok/batch
MAX_LEARNING_RATE=0.0001 # Highest lr that we warmup to (be a little conservative, fp16 at high lr has led to training divergence)
MIN_LEARNING_RATE_RATIO=0.1 # Proportion of highest lr that we decay down to
WARMUP_RATIO=0.05 # What proportion of training we do warmup for
BETA1=0.9 # Adam Beta1
BETA2=0.95 # Adam Beta2 
WEIGHT_DECAY=0.1 # Adam Weight decay (non embedding params)
MAX_GRAD_NORM=1.0 # Max for grad clipping

### CHECKPOINTING CONFIG ###
CHECKPOINT_ITERATIONS=5000 # after how many steps do you want to create a checkpoint? more frequent uses more disk space!

# ===================================================================
# CREATE ALL THE DIRECTORIES
# ===================================================================
# Create directories and set HF Home for dataset caching (can be deleted later)
mkdir -p $HF_CACHE_DIR
mkdir -p $RAW_TEXT_DIRECTORY
mkdir -p $TOKENIZED_DIRECTORY
mkdir -p $PRETRAIN_WORKING_DIRECTORY
mkdir -p $MIDTRAIN_WORKING_DIRECTORY
mkdir -p $SFT_WORKING_DIRECTORY
mkdir -p $DOWNLOAD_TASKS_PATH
export HF_HOME=$HF_CACHE_DIR

# ===================================================================
# CACHE THIS MODELS CONFIG
# ===================================================================
python -m nanochat_trainer.scripts.save_model_meta \
    --path_to_store $EXPERIMENT_WORKING_DIRECTORY \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --num_blocks $NUM_BLOCKS \
    --embed_dim $EMBED_DIM \
    --num_q_heads $NUM_Q_HEADS \
    --num_kv_heads $NUM_KV_HEADS \
    --mlp_ratio $MLP_RATIO \
    --path_to_tokenizer $PATH_TO_SAVE_TOKENIZER

# ===================================================================
#  DOWNLOAD AND TOKENIZE ALL THE PRETRAINING DATA
# ===================================================================
### DOWNLOAD SLICE OF FINEWEB ###
### With the default settings this will download ~60 parquet files from 100BT split
### and will save a final ~90 parquet files (about 200GB of data!)
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
    --path_to_tokenizer $PATH_TO_SAVE_TOKENIZER \
    --checkpoint_iterations $CHECKPOINT_ITERATIONS \
    --log_wandb