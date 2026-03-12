#!/bin/bash

# Simple script to download/prepare data and train a ~500M param LLM!
# This will auto resume, if you stop anywhere, just rerun the script and
# it will just pick up from where it left off! 

# ===================================================================
# SET ALL OUR DIRECTORIES
# ===================================================================

### Set These To Where You Want to Save Your Stuff! ###
EXPERIMENT_NAME="mytorch_llm_500M" # Name for the run for local checkpointing and WandB
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

# ===================================================================
# DEFINE OUR MODEL STRUCTURE
# ===================================================================

### MODEL SHAPE (This is the config for a ~500M param LLM)
VOCAB_SIZE=65536 # 2**16 
CONTEXT_LENGTH=2048 # Total context this model will process (and data will be cut into)
NUM_BLOCKS=16 # Number of transformer blocks
EMBED_DIM=1280 # Embedding dimension 
NUM_Q_HEADS=10 # Number of Query heads
NUM_KV_HEADS=10 # Number of KV Heads (must evenly divide Q Heads for GQA)
MLP_RATIO=4 # MLP Ratio in Feed forward

# ===================================================================
# TRAINING CONFIGS
# ===================================================================

### DATALOADER CONFIG ###
NUM_WORKERS=32 # Number of cpu workers for everything data related

### GLOBAL TRAINING CONFIG -> Applies to all training scripts ###
PER_GPU_BATCH_SIZE=4 # Set to whatever doesn't OOM!
BETA1=0.9 # Adam Beta1
BETA2=0.95 # Adam Beta2 
WEIGHT_DECAY=0.1 # Adam Weight decay (non embedding params)
MAX_GRAD_NORM=1.0 # Max for grad clipping

### PRETRAINING CONFIG (~19000 Steps in the preset config) ###
PRETRAIN_CHINCHILLA_RATIO=20 # you can increase this to overtrain model on > chinchilla optimal
PRETRAIN_TARGET_TOKENS_PER_BATCH=524288 # Grad accumulation until we hit this tok/batch
PRETRAIN_MAX_LEARNING_RATE=0.00015 # Highest lr that we warmup to (be a little conservative, fp16 at high lr has led to training divergence)
PRETRAIN_MIN_LEARNING_RATE_RATIO=0.1 # Proportion of highest lr that we decay down to
PRETRAIN_WARMUP_RATIO=0.05 # What proportion of training we do warmup for
PRETRAIN_CHECKPOINT_ITERATIONS=5000 # After how many steps do you want to save a checkpoint? More frequent is more disk space!

### MIDTRAINING CONFIG (~800 Steps in preset config) ###
MIDTRAIN_EPOCHS=1 # How many epochs through the midtraining data do you want to do?
MIDTRAIN_TARGET_TOKENS_PER_BATCH=524288 # Grad accumulation until we hit this tok/batch
MIDTRAIN_MAX_LEARNING_RATE=0.00005 # Highest lr that we warmup to (less than pretraining lr to avoid catastrophic forgetting)
MIDTRAIN_MIN_LEARNING_RATE_RATIO=0.1 # Proportion of highest lr that we decay down to
MIDTRAIN_WARMUP_RATIO=0.05 # What proportion of training we do warmup for
MIDTRAIN_CHECKPOINT_ITERATIONS=500 # After how many steps do you want to save a checkpoint? More frequent is more disk space!

### SFT CONFIG (~700 Steps in preset config) ###
SFT_EPOCHS=1 # How many epochs through the midtraining data do you want to do?
SFT_EXAMPLES_PER_BATCH=32 # Grad accumulation until we hit this tok/batch
SFT_MAX_LEARNING_RATE=0.00001 # Highest lr that we warmup to (less than pretraining lr to avoid catastrophic forgetting)
SFT_MIN_LEARNING_RATE_RATIO=0.1 # Proportion of highest lr that we decay down to
SFT_WARMUP_RATIO=0.05 # What proportion of training we do warmup for
SFT_CHECKPOINT_ITERATIONS=500 # After how many steps do you want to save a checkpoint? More frequent is more disk space!

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
## DOWNLOAD SLICE OF FINEWEB ###
## With the default settings this will download ~20 parquet files from 100BT split
## and will save a final ~30 parquet files (about 55GB of data!)
python -m nanochat_trainer.scripts.download_fineweb_edu \
    --path_to_save $RAW_TEXT_DIRECTORY \
    --num_workers $NUM_WORKERS \
    --chinchilla_ratio $PRETRAIN_CHINCHILLA_RATIO \
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
    --chinchilla_ratio $PRETRAIN_CHINCHILLA_RATIO \
    --batch_size_per_gpu $PER_GPU_BATCH_SIZE \
    --tokens_per_batch $PRETRAIN_TARGET_TOKENS_PER_BATCH \
    --num_workers $NUM_WORKERS \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --num_blocks $NUM_BLOCKS \
    --embed_dim $EMBED_DIM \
    --num_q_heads $NUM_Q_HEADS \
    --num_kv_heads $NUM_KV_HEADS \
    --mlp_ratio $MLP_RATIO \
    --max_learning_rate $PRETRAIN_MAX_LEARNING_RATE \
    --min_learning_rate_ratio $PRETRAIN_MIN_LEARNING_RATE_RATIO \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $PRETRAIN_WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --checkpoint_iterations $PRETRAIN_CHECKPOINT_ITERATIONS \
    --log_wandb

# ===================================================================
# MIDTRAINING (Lets Make it Conversational)
# ===================================================================
### Download and prepare conversational datasets. The main datasets we 
### will be doing here as an example is "smoltalk", "arc_easy", 
### "arc_challenge", "mmlu", "gsm8k"
python -m nanochat_trainer.scripts.tasks_prep \
    --path_to_store $DOWNLOAD_TASKS_PATH \
    --path_to_tokenizer $PATH_TO_SAVE_TOKENIZER \
    --num_workers $NUM_WORKERS

### Midtrain the model now!
mytorchrun launch -m nanochat_trainer.scripts.midtrain_model \
    --work_dir $MIDTRAIN_WORKING_DIRECTORY \
    --experiment_name $EXPERIMENT_NAME \
    --path_to_starting_checkpoint $PRETRAIN_WORKING_DIRECTORY \
    --path_to_data $DOWNLOAD_TASKS_PATH \
    --epochs $MIDTRAIN_EPOCHS \
    --batch_size_per_gpu $PER_GPU_BATCH_SIZE \
    --tokens_per_batch $MIDTRAIN_TARGET_TOKENS_PER_BATCH \
    --num_workers $NUM_WORKERS \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --num_blocks $NUM_BLOCKS \
    --embed_dim $EMBED_DIM \
    --num_q_heads $NUM_Q_HEADS \
    --num_kv_heads $NUM_KV_HEADS \
    --mlp_ratio $MLP_RATIO \
    --max_learning_rate $MIDTRAIN_MAX_LEARNING_RATE \
    --min_learning_rate_ratio $MIDTRAIN_MIN_LEARNING_RATE_RATIO \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $MIDTRAIN_WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --checkpoint_iterations $MIDTRAIN_CHECKPOINT_ITERATIONS \
    --log_wandb

# ===================================================================
# SUPERVISED FINE TUNING (Single Conversation Samples)
# ===================================================================
### In SFT each batch will have different lengths, to avoid retriggering the 
### autotuner on our linear layer we just disable it here. A bit hacky
### but CUDA kernels are precompiled, and NVIDIA/CUBLAS already knows the ideal
### settings for a specific matmul, so it internally dispatches to the optimal
### matmul. The Autotuner from triton gives similar performance
### but has to be tuned to a specific shape. This didnt matter earlier as in 
### midtraining/pretraining our seq lens and batch size are always the same. 
### but now as they dynamically will change based on the longest sample in the batch
### we have this issue. So this setting here will essentially use CUPY matmuls do 
### do our linear ops so it will internally dispatch to cuBLAS and will be as performant
### as possible!!!
### An alternative is just pad each sample to the context length and then we wouldnt really care
### about all this extra stuff!
export DISABLE_FUSED_LINEAR="true" 
mytorchrun launch -m nanochat_trainer.scripts.sft_model \
    --work_dir $SFT_WORKING_DIRECTORY \
    --experiment_name $EXPERIMENT_NAME \
    --path_to_starting_checkpoint $MIDTRAIN_WORKING_DIRECTORY \
    --path_to_data $DOWNLOAD_TASKS_PATH \
    --epochs $SFT_EPOCHS \
    --batch_size_per_gpu $PER_GPU_BATCH_SIZE \
    --examples_per_batch $SFT_EXAMPLES_PER_BATCH \
    --num_workers $NUM_WORKERS \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --num_blocks $NUM_BLOCKS \
    --embed_dim $EMBED_DIM \
    --num_q_heads $NUM_Q_HEADS \
    --num_kv_heads $NUM_KV_HEADS \
    --mlp_ratio $MLP_RATIO \
    --max_learning_rate $SFT_MAX_LEARNING_RATE \
    --min_learning_rate_ratio $SFT_MIN_LEARNING_RATE_RATIO \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --weight_decay $WEIGHT_DECAY \
    --warmup_ratio $SFT_WARMUP_RATIO \
    --max_grad_norm $MAX_GRAD_NORM \
    --checkpoint_iterations $SFT_CHECKPOINT_ITERATIONS \
    --path_to_tokenizer $PATH_TO_SAVE_TOKENIZER