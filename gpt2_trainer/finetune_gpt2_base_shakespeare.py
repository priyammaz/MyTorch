"""
We will finetune our GPT2 Base model on Shakespeare!
"""

import os
import argparse
import numpy as np
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from mytorch.utils.data import Dataset, DataLoader
from gpt2 import GPT2, GPT2Config
from mytorch import Accelerator
from tqdm import tqdm
import pickle
from safetensors.numpy import load_file

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT2 with MyTorch")

    ### Experiment Config ###
    parser.add_argument("--project_name", type=str, default="gpt2-base-ft-shakespeare")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--path_to_data", type=str, default="data/shakespeare_gpt2")

    ### Checkpointing Config ###
    parser.add_argument("--working_directory", type=str, default="work_dir")
    parser.add_argument("--load_from_experiment", type=str, default="gpt2-base-owt")
    parser.add_argument("--checkpoint_dir", type=str)

    ### Model Config ###
    parser.add_argument("--dropout_p", type=float, default=0.)
    parser.add_argument("--use_bias", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--num_layers_train", type=int, default=2)

    ### Training Config ###
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_iterations", type=int, default=2500)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iterations", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", action="store_true")

    ### Logging Config ###
    parser.add_argument("--log_iter", type=int, default=100)
    parser.add_argument("--log_wandb", action="store_true")

    args = parser.parse_args()  

    return args

args = parse_args()

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

### Load From Checkpoint ###
path_to_loading_experiment = os.path.join(args.working_directory, args.load_from_experiment)
if args.checkpoint_dir is None:
    checkpoints = sorted([f for f in os.listdir(path_to_loading_experiment) if "checkpoint" in f], 
                             key=lambda x: int(x.split("_")[-1]))
    if "final_checkpoint" in checkpoints:
        checkpoint = "final_checkpoint"
    else:
        checkpoint = checkpoints[-1]

    path_to_checkpoint = os.path.join(path_to_loading_experiment, checkpoint)
    print(f"Loading From Checkpoint: {path_to_checkpoint}")
else:
    path_to_checkpoint = os.path.join(path_to_loading_experiment, args.checkpoint_dir)

### Load Model Meta ###
path_to_model_meta = os.path.join(path_to_loading_experiment, "model_meta.pkl")
with open(path_to_model_meta, "rb") as f:
    model_meta = pickle.load(f)

### Load Model Checkpoint ###
path_to_load_from = os.path.join(path_to_checkpoint, "model.safetensors")
state_dict = load_file(path_to_load_from)

### DataLoader ###
class TokenLoader(Dataset):
    def __init__(self, path_to_bin, context_length):

        self.path_to_bin = path_to_bin
        self.context_length = context_length
        self.arr = np.memmap(path_to_bin, dtype=np.uint16, mode='r')
        self.num_tokens = self.arr.shape[0]
    
    def __len__(self):

        """
        We dont really have a len here as we will just be randomly sampling slices
        from out dataset. We only are doing this so our dataloader can prefetch data
        for us as the model is doing its thing. 

        So lets just set our len as an even multiple of the batch size, the multiple
        doenst really matter as we are training in steps and not epochs and the dataloader
        will just continue to cycle. 
        """
        return args.batch_size_per_gpu * 100

    def __getitem__(self, idx):

        start_idx = np.random.randint(low=0, high=self.num_tokens - self.context_length - 1)
        x = self.arr[start_idx:start_idx+self.context_length]
        y = self.arr[start_idx+1:start_idx+self.context_length+1]
        return mytorch.Tensor(x), mytorch.Tensor(y)

trainset = TokenLoader(os.path.join(args.path_to_data, 'train.bin'), context_length=model_meta["context_length"])
testset = TokenLoader(os.path.join(args.path_to_data, 'val.bin'), context_length=model_meta["context_length"])    

micro_batchsize = args.batch_size//args.gradient_accumulation_steps
trainloader = DataLoader(trainset, batch_size=micro_batchsize, num_workers=args.num_workers)
testloader = DataLoader(testset, batch_size=1, num_workers=args.num_workers) # not very much data in shakespeare

### Create Checkpoint Directory ###
path_to_experiment = os.path.join(args.working_directory, args.project_name)
if args.run_name is not None:
    path_to_experiment = os.path.join(path_to_experiment, args.run_name)
os.makedirs(path_to_experiment, exist_ok=True)

### Init DDP ###
accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                          mixed_precision=args.mixed_precision,
                          log_wandb=args.log_wandb)

### Config to store model information (so we can load precise setup for inference!) ###
training_config = {
                    "vocab_size": model_meta["vocab_size"],
                    "context_length": model_meta["context_length"],
                    "embed_dim": model_meta["embed_dim"],
                    "num_heads": model_meta["num_heads"],
                    "num_blocks": model_meta["num_blocks"],
                    "mlp_ratio": model_meta["mlp_ratio"],
                    "use_bias": model_meta["use_bias"],
                    "dropout": args.dropout_p,
                    "batch_size": args.batch_size,
                    "grad_accumulation_steps": args.gradient_accumulation_steps,
                    "max_lr": args.max_lr,
                    "weight_decay": args.weight_decay,
                    "adam_beta1": args.beta1,
                    "adam_beta2": args.beta2,
                    "mixed_precision": args.mixed_precision,
                    "path_to_experiment": path_to_experiment,
                    "path_to_tokenizer": None
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
    vocab_size=model_meta["vocab_size"], 
    max_seq_len=model_meta["context_length"],
    embed_dim=model_meta["embed_dim"], 
    num_heads=model_meta["num_heads"], 
    num_blocks=model_meta["num_blocks"], 
    attn_dropout_p=args.dropout_p, # Only works for non-fused attn 
    mlp_dropout_p=args.dropout_p, 
    mlp_ratio=model_meta["mlp_ratio"], 
    use_bias=args.use_bias, 
    use_fused_ops=args.fused
)
model = GPT2(gpt2config)

### Load Checkpoint ###
model.load_state_dict(state_dict)

### Lets Disable Grad for most of the model and only finetune the head and the last block! ###
num_blocks = model_meta["num_blocks"]
if args.num_layers_train != -1:
    accelerator.print(f"Freezing all but last {args.num_layers_train} layers")
    for name, param in model.named_parameters():
        if (
            "final_layer" in name
            or any(f"blocks.{i}" in name for i in range(num_blocks - args.num_layers_train, num_blocks))
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

total_params = 0
for param in model.parameters():
    if param.requires_grad:
        total_params += np.prod(param.shape)
accelerator.print("Total Trainable Parameters:", total_params)

### Load Optimizer ###
optimizer = optim.AdamW(model.parameters(), 
                        lr=args.max_lr, 
                        weight_decay=args.weight_decay,
                        beta1=args.beta1, 
                        beta2=args.beta2)

### Load Scheduler ###
scheduler = mytorch.lr_scheduler.CosineLRScheduler(
    optimizer=optimizer, max_lr=args.max_lr, 
    min_lr=args.min_lr, total_steps=args.train_iterations,
    warmup_steps=args.warmup_steps
)

### Prepare Everything ###
model, optimizer, trainloader, testloader = accelerator.prepare(
    model, optimizer, trainloader, testloader
)

### Load Loss Function ###
loss_fn = nn.CrossEntropyLoss(fused=args.fused)

### Train Model ###
pbar = tqdm(range(args.train_iterations), 
            disable=not accelerator.is_main_process())

completed_steps = 0
train = True
while train:

    for inputs, targets in trainloader:
        
        # Move to correct device 
        inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)

        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        # Backward
        accelerator.backward(loss)
        
        # Clip gradients (and get the grads to check on training health)
        grad_norm = accelerator.clip_grad_norm_(args.max_grad_norm)

        # Step optimizer
        optimizer.step()
        optimizer.zero_grad()

        ### Accelerator tracks when accumulation is done, the flag is just sync_grad ###
        if accelerator.sync_grad:
            
            ### One full accumulation complete ###
            completed_steps += 1
            pbar.update(1)

            ### Update Scheduler ###
            scheduler.step()

            ### Gather metrics across GPUs
            if completed_steps % args.log_iter == 0:

                ### Gather (no-op if we are on single GPU) ###
                loss = accelerator.gather_for_metrics(loss)

                ### Grab our stored grad_norm for checking on model health ###
                lr = scheduler.get_last_lr()[0] if isinstance(scheduler.get_last_lr(), list) else scheduler.get_last_lr()
                log_parts = [
                    f"Iter: {completed_steps:6d}",
                    f"Loss: {loss:7.4f}",
                    f"LR: {lr:9.2e}"
                ]
                if accelerator.grad_norm is not None:
                    log_parts.append(f"GradNorm: {accelerator.grad_norm:7.3f}")
                
                ### Print to Console ###
                log_statement = " | ".join(log_parts)
                if accelerator.is_main_process():
                    tqdm.write(log_statement)

                ### Log with Wandb if enabled ###
                if args.log_wandb:  
                    logging_dict = {"loss": loss, "lr": lr}
                    if accelerator.grad_norm is not None:
                        logging_dict["grad_norm"] = accelerator.grad_norm 
                    accelerator.log(logging_dict, step=completed_steps)

            if completed_steps % args.eval_interval == 0:
                
                accelerator.print("Evaluating!")
                model.eval()

                val_losses = []

                for val_iter, (inputs, targets) in enumerate(testloader):
   
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

        if completed_steps >= args.train_iterations:
            accelerator.print("Completed Training!!!")
            train = False
            break

### Save final checkpoint once done ! ###
accelerator.save_state(os.path.join(path_to_experiment, f"final_checkpoint"), save_model_only=True)
accelerator.end_training()