"""
Basic Training script that can be launched in a few ways:

Existing Bash Script: `gpt2_trainer/train_gpt2.py`

To run on single GPU (shakespeare): 
$ bash gpt2_trainer/train_gpt2.sh shakespeare

To enable Fused Ops/Mixed Precision
$ bash gpt2_trainer/train_gpt2.sh shakespeare --fused --mixed_precision

To train on multiple GPUs
$ bash gpt2_trainer/train_gpt2.sh shakespeare --fused --mixed_precision --num_gpus 4

Use mytorchrun launcher instead (requires mytorch-core install and `mytorchrun config` setup!)
$ bash gpt_trainer/train_gpt2_launcher.sh shakespeare

To reproduce GPT2 (124M) on OpenWebText:
$ bash gpt2_trainer/train_gpt2.sh owt --fused --mixed_precision --num_gpus 4
or assuming your `mytorchrun config` is setup:
$ bash gpt_trainer/train_gpt2_launcher.sh owt

"""

import os
import argparse
import numpy as np
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from mytorch.utils.data import Dataset, DataLoader
from models.gpt2 import GPT2, GPT2Config
from mytorch import Accelerator
from tqdm import tqdm
import pickle
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT2 with MyTorch")

    ### Experiment Config ###
    parser.add_argument("--project_name", type=str, default="GPT2Trainer")
    parser.add_argument("--run_name", type=str)

    ### Checkpointing Config ###
    parser.add_argument("--working_directory", type=str, default="work_dir")
    parser.add_argument("--checkpoint_iterations", type=int, default=10000)
    parser.add_argument("--always_save_checkpoint", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str)

    ### Model Config ###
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--model_size", type=str, choices=("small", "base", "large"), default="base")
    parser.add_argument("--dropout_p", type=float, default=0.)
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--fused", action="store_true")

    ### Training Config ###
    parser.add_argument("--path_to_data", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_iterations", type=int, default=150000)
    parser.add_argument("--use_chinchilla", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--eval_iterations", type=int, default=200)
    parser.add_argument("--batch_size_per_gpu", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--tokens_per_batch", type=int, default=491520)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--mixed_precision", action="store_true")

    ### Logging Config ###
    parser.add_argument("--log_iter", type=int, default=100)
    parser.add_argument("--log_wandb", action="store_true")

    ### Extra ###
    parser.add_argument("--print_banner", action="store_true")

    args = parser.parse_args()

    return args

args = parse_args()

### Init DDP ###
accelerator = Accelerator(mixed_precision=args.mixed_precision,
                          log_wandb=args.log_wandb)

if args.print_banner: # <- mytorchrun launch will print its own banner!
    accelerator.print(mytorch.banner)
    accelerator.env()

### Load Model Variant ###
if args.model_size == "small":
    embed_dim = 384
    num_heads = 6
    num_blocks = 6
    mlp_ratio = 4
elif args.model_size == "base":
    embed_dim = 768
    num_heads = 12
    num_blocks = 12
    mlp_ratio = 4
elif args.model_size == "large":
    embed_dim = 1280
    num_heads = 10
    num_blocks = 20
    mlp_ratio = 4

### Check Dataset ###
if not os.path.exists(args.path_to_data):
    raise ValueError("Verify your path to data, it should contain "
                     "a train.bin, val.bin and optionally a tokenizer.pkl")

required_files = ["train.bin", "val.bin"]
optional_files = ["tokenizer.pkl"]

for fname in required_files:
    fpath = os.path.join(args.path_to_data, fname)
    if not os.path.exists(fpath):
        raise ValueError(f"Missing required file: {fname} in {args.path_to_data}")

# Optionally check tokenizer
tokenizer_path = os.path.join(args.path_to_data, "tokenizer.pkl")
tokenizer_exists = os.path.exists(tokenizer_path)
if not tokenizer_exists:
    print("No tokenizer.pkl found, assuming default gpt2 tokenizer!")
    vocab_size = 50257
else:
    with open(tokenizer_path, "rb") as f:
        tokenizer_meta = pickle.load(f)

    vocab_size = tokenizer_meta["vocab_size"]

### DataLoader ###
class TokenLoader(Dataset):
    def __init__(self, path_to_bin, context_length):

        self.path_to_bin = path_to_bin
        self.context_length = context_length
        self.arr = np.memmap(path_to_bin, dtype=np.uint16, mode='r')
        self.num_tokens = self.arr.shape[0]
    
    def __len__(self):
        """
        We dont really have the number of "samples" in our data as we just
        grab random sets of consecutive tokens of size context_length. Lets just
        give a good guess!
        """
        return self.num_tokens // self.context_length

    def __getitem__(self, idx):

        start_idx = np.random.randint(low=0, high=self.num_tokens - self.context_length - 1)
        x = self.arr[start_idx:start_idx+self.context_length]
        y = self.arr[start_idx+1:start_idx+self.context_length+1]
        return mytorch.Tensor(x), mytorch.Tensor(y)

### If Grad Accum Steps is not provided we can compute it from our target tokens per batch ###
if args.gradient_accumulation_steps is not None:
    gradient_accumulation_steps = args.gradient_accumulation_steps
else:
    tokens_per_batch_goal = args.tokens_per_batch
    tokens_per_gpu = tokens_per_batch_goal // accelerator.num_processes
    needed_batch_size = tokens_per_gpu // args.context_length
    gradient_accumulation_steps = needed_batch_size // args.batch_size_per_gpu

### Set This Accelerator For Sync ###
accelerator.gradient_accumulation_steps = gradient_accumulation_steps

### Compute Tokens Per Batch ###
toks_per_batch = args.batch_size_per_gpu * gradient_accumulation_steps * accelerator.num_processes * args.context_length
accelerator.print(f"Tokens Processed Per Batch:", toks_per_batch)
accelerator.print(f"Using Micro-Batch Size of {args.batch_size_per_gpu} for {gradient_accumulation_steps} Gradient Accumulation Steps")

### Build Datasets ###
trainset = TokenLoader(os.path.join(args.path_to_data, 'train.bin'), context_length=args.context_length)
testset = TokenLoader(os.path.join(args.path_to_data, 'val.bin'), context_length=args.context_length)    
micro_batchsize = args.batch_size_per_gpu
trainloader = DataLoader(trainset, batch_size=micro_batchsize, num_workers=args.num_workers)
testloader = DataLoader(testset, batch_size=micro_batchsize, num_workers=args.num_workers)

### Create Checkpoint Directory ###
path_to_experiment = os.path.join(args.working_directory, args.project_name)
if args.run_name is not None:
    path_to_experiment = os.path.join(path_to_experiment, args.run_name)
os.makedirs(path_to_experiment, exist_ok=True)

### Config to store model information (so we can load precise setup for inference!) ###
training_config = {
                    "vocab_size": vocab_size,
                    "context_length": args.context_length,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "num_blocks": num_blocks,
                    "mlp_ratio": mlp_ratio,
                    "use_bias": args.use_bias,
                    "dropout": args.dropout_p,
                    "batch_size_per_gpu": args.batch_size_per_gpu,
                    "grad_accumulation_steps": gradient_accumulation_steps,
                    "tokens_per_batch": toks_per_batch,
                    "max_lr": args.max_lr,
                    "weight_decay": args.weight_decay,
                    "adam_beta1": args.beta1,
                    "adam_beta2": args.beta2,
                    "mixed_precision": args.mixed_precision,
                    "path_to_experiment": path_to_experiment,
                    "path_to_tokenizer": tokenizer_path if tokenizer_exists else None
                }

if accelerator.is_main_process:
    with open(os.path.join(path_to_experiment, "model_meta.pkl"), "wb") as f:
            pickle.dump(training_config, f)

if args.log_wandb:
    accelerator.init_tracker(project_name=args.project_name, 
                             run_name=args.run_name,
                             config=training_config)

### Load Model ###
gpt2config = GPT2Config(
    vocab_size=vocab_size, 
    max_seq_len=args.context_length,
    embed_dim=embed_dim, 
    num_heads=num_heads, 
    num_blocks=num_blocks, 
    attn_dropout_p=args.dropout_p, # Only works for non-fused attn 
    mlp_dropout_p=args.dropout_p, 
    mlp_ratio=mlp_ratio, 
    use_bias=args.use_bias, 
    use_fused_ops=(args.fused or (True if os.environ.get("USE_FUSED_OPS", "False") == "True" else False)) # <- mytorchrun can force all other ops to be fused 
)                                                                                                         #    but flash_attn needs this flag inside the        
                                                                                                          #    model to trigger, otherwise it will use naive
                                                                                                          #    a little messy but i want both mytorchrun and 
                                                                                                          #    normal distributed launch to work, and i dont
                                                                                                          #    want to check env flags inside the model
model = GPT2(gpt2config)

total_params = 0
for param in model.parameters():
    if param.requires_grad:
        total_params += np.prod(param.shape)
accelerator.print("Total Trainable Parameters:", total_params)

### Set Training Steps If Using Chinchilla ###
if args.use_chinchilla:
    accelerator.print("Using Chinchilla Scaling Law!")
    train_iterations = (total_params * 20) // toks_per_batch
else:
    train_iterations = args.train_iterations

accelerator.print(f"Training for {train_iterations} Iterations")

### Load Optimizer ###
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if len(param.shape) < 2: # <- dont apply weight decay to layernorm/biases
        no_decay_params.append(param)
    else:
        decay_params.append(param)
param_groups = [
    {"params": decay_params, "weight_decay": args.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0}
]
optimizer = optim.AdamW(param_groups, 
                        lr=args.max_lr,
                        beta1=args.beta1, 
                        beta2=args.beta2)
accelerator.print(optimizer)

### Load Scheduler ###
scheduler = mytorch.lr_scheduler.CosineLRScheduler(
    optimizer=optimizer, max_lr=args.max_lr, 
    min_lr=args.min_lr, total_steps=train_iterations,
    warmup_steps=args.warmup_steps
)

### Prepare Everything ###
model, optimizer, trainloader, testloader = accelerator.prepare(
    model, optimizer, trainloader, testloader
)

### Resume from Checkpoint ###
if args.resume_from_checkpoint is not None:

    ### This is if we pass in a full path to checkpoint ###
    if os.path.exists(args.resume_from_checkpoint):
        path_to_checkpoint = args.resume_from_checkpoint
    
    ### Otherwise we can pass the folder name in our experiment directory ###
    else:
        path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    
    ### Load our State (model and optimizer) ###
    accelerator.load_state(path_to_checkpoint)
    
    ### Start completed steps from checkpoint index ###
    completed_steps = int(args.resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")

    ### Advance our scheduler to the correct step ###
    scheduler.step_count = completed_steps

else:
    completed_steps = 0

### Load Loss Function ###
loss_fn = nn.CrossEntropyLoss(fused=args.fused)

### Train Model ###
pbar = tqdm(range(train_iterations), 
            disable=not accelerator.is_main_process(),
            initial=completed_steps)

t0 = None
train = True
while train:

    for inputs, targets in trainloader:
        
        # Time Batch 
        if t0 is None:
            t0 = time.time()

        # Move to correct device 
        inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)

        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        # Backward
        accelerator.backward(loss)
        
        # Clip gradients (and get the grads to check on training health)
        accelerator.clip_grad_norm_(args.max_grad_norm)

        # Step optimizer
        optimizer.step()
        optimizer.zero_grad()

        ### Accelerator tracks when accumulation is done, the flag is just sync_grad ###
        if accelerator.sync_grad:
            
            ### Get Time and reset start time ###
            t1 = time.time()
            dt = t1 - t0
            t0 = None
            
            ### Iter ###
            completed_steps += 1
            pbar.update(1)

            ### Update Scheduler ###
            scheduler.step()

            ### Gather metrics across GPUs
            if completed_steps % args.log_iter == 0:

                ### Gather (no-op if we are on single GPU) ###
                loss = accelerator.gather_for_metrics(loss)

                ### Logging stuff ###
                lr = scheduler.get_last_lr()[0] if isinstance(scheduler.get_last_lr(), list) else scheduler.get_last_lr()
                log_parts = [
                    f"Iter: {completed_steps:6d}",
                    f"Loss: {loss:7.4f}",
                    f"LR: {lr:9.2e}"
                ]
                ### Grab our stored grad_norm for checking on model health ###
                if accelerator.grad_norm is not None:
                    log_parts.append(f"GradNorm: {accelerator.grad_norm:7.3f}")
                log_parts.append(f"Toks/Sec: {int(toks_per_batch / dt):6d}")
                log_statement = " | ".join(log_parts)

                if accelerator.is_main_process():
                    tqdm.write(log_statement)

                ### Log with Wandb if enabled ###
                if args.log_wandb:  
                    logging_dict = {"loss": loss, "lr": scheduler.get_last_lr()}
                    if accelerator.grad_norm is not None:
                        logging_dict["grad_norm"] = accelerator.grad_norm 
                    accelerator.log(logging_dict, step=completed_steps)
 
            if completed_steps % args.checkpoint_iterations == 0 and args.always_save_checkpoint:
                accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{completed_steps}"))

            if completed_steps % args.eval_interval == 0:
                
                accelerator.print("Evaluating!")
                model.eval()

                val_losses = []

                for val_iter, (inputs, targets) in enumerate(testloader):
                    if val_iter >= args.eval_iterations:
                        break  # stop after desired number of eval iterations

                    inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)

                    with mytorch.no_grad():
                        logits = model(inputs)

                    loss = loss_fn(logits, targets)
                    loss_val = accelerator.gather_for_metrics(loss)
                    val_losses.append(loss_val)

                ### Log Loss ###
                val_losses = np.mean(val_losses)
                accelerator.print("Validation Loss:", val_losses)
                if args.log_wandb:
                    logging_dict = {"val_loss": val_losses}
                    accelerator.log(logging_dict, step=completed_steps)

                ### Set back into Training Mode ###
                model.train()

        if completed_steps >= train_iterations:
            accelerator.print("Completed Training!!!")
            train = False
            break

### Save final checkpoint once done ! ###
accelerator.save_state(os.path.join(path_to_experiment, f"final_checkpoint"), save_model_only=True)
accelerator.end_training()