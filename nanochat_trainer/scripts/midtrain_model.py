"""
Midtraining and Pretraining are the same thing. Its just now we are training on conversational datasets
rather than just a bunch of data text from the internet! The datasets we train on are the: 
- smoltalk dataset
- mmlu
- gsm8k

The dataset mixture can be set via the MixtureDataset!

"""
import os
import numpy as np
import argparse
import mytorch
from mytorch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import time

from mytorch.accelerate import Accelerator
from nanochat_trainer.core.nanochat_gpt import GPT, GPTConfig
from nanochat_trainer.scripts.utils import get_last_checkpoint 
from nanochat_trainer.core.tasks import Task, MixtureDataset, GroupedDataset

import warnings

terminal_width = shutil.get_terminal_size().columns

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--path_to_starting_checkpoint", type=str, required=True) # this is new, we need to start from our pretrained checkpoint
    parser.add_argument("--path_to_data", type=str, default="data/tasks")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size_per_gpu", type=int, default=8)
    parser.add_argument("--tokens_per_batch", type=int, default=524288)
    parser.add_argument("--num_workers", type=int, default=8)

    ### Model Shape ###
    parser.add_argument("--vocab_size", default=2**16, type=int)
    parser.add_argument("--context_length", default=2048, type=int)
    parser.add_argument("--num_blocks", default=20, type=int)
    parser.add_argument("--embed_dim", default=1280, type=int)
    parser.add_argument("--num_q_heads", default=10, type=int)
    parser.add_argument("--num_kv_heads", default=10, type=int)
    parser.add_argument("--mlp_ratio", default=4, type=int)

    ### Optimizer Args ###
    parser.add_argument("--max_learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_learning_rate_ratio", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    ### Logging ###
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--log_iter", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=150)
    parser.add_argument("--eval_iterations", type=int, default=200)
    parser.add_argument("--checkpoint_iterations", type=int, default=500)

    args = parser.parse_args()
    
    return args

def trainer(args, 
            path_to_experiment, 
            starting_checkpoint=None, # This is the path to the last checkpoint created from pretraining
            resume_from_checkpoint=None # This is the path to the last checkpoint created during midtraining
    ):

    ### Load Accelerator ###
    accelerator = Accelerator(log_wandb=args.log_wandb)

    ### Quick check on checkpoints ###
    if (starting_checkpoint is None) and (resume_from_checkpoint is None):
        warnings.warn("You are starting Midtraining WITHOUT and pretrained weights!!! NOT RECOMMENDED!!!")

    ### Init tracker ###
    if args.log_wandb:
        accelerator.init_tracker(project_name=args.experiment_name, 
                                 run_name="midtraining",
                                 config=vars(args))
        
    ### Get path to work dir ###
    if not os.path.isdir(path_to_experiment) and accelerator.is_main_process:
        accelerator.print("Creating Working Directory:", path_to_experiment)
        os.makedirs(path_to_experiment, exist_ok=True)
    
    ### Setup Dataset ###
    trainset = MixtureDataset([
        Task(args.path_to_data, "smoltalk", keep_mask=False), # 460K samples of general conversation 
        Task(args.path_to_data, "mmlu", keep_mask=False), # 90K samples of multiple choice
        Task(args.path_to_data, "gsm8k", keep_mask=False), # 8K samples of arithmetic
    ])

    trainset = GroupedDataset(
        dataset=trainset, 
        chunk_size=args.context_length, 
        num_proc=args.num_workers
    )

    testset = MixtureDataset([
        Task(args.path_to_data, "smoltalk", "test", keep_mask=False), 
        Task(args.path_to_data, "mmlu", "test", keep_mask=False),
        Task(args.path_to_data, "gsm8k", "test", keep_mask=False), 
    ])

    testset = GroupedDataset(
        dataset=testset, 
        chunk_size=args.context_length, 
        num_proc=args.num_workers
    )

    ### Compute Gradient Accumulation Steps ###
    num_devices = accelerator.num_processes
    tokens_per_sample = args.context_length # Data has already been sliced into chunks of this size
    batch_per_device = args.batch_size_per_gpu
    tokens_per_step = num_devices * tokens_per_sample * batch_per_device
    gradient_accumulation_steps = args.tokens_per_batch // tokens_per_step

    ### Set our grad accum steps manually here ###
    accelerator.gradient_accumulation_steps = gradient_accumulation_steps

    ### Load Model ###
    config = GPTConfig(vocab_size=args.vocab_size, 
                       max_seq_len=args.context_length,
                       embed_dim=args.embed_dim, 
                       mlp_ratio=args.mlp_ratio, 
                       num_blocks=args.num_blocks,
                       num_q_heads=args.num_q_heads, 
                       num_kv_heads=args.num_kv_heads)
    model = GPT(config)

    ### Load our starting checkpoint from pretraining if we are not resuming our midtraining run! ###
    if (starting_checkpoint is not None) and (resume_from_checkpoint is None):
        accelerator.print("Starting from Stage 1 Pretraining Checkpoint: ", starting_checkpoint)
        state_dict = mytorch.load(starting_checkpoint)
        model.load_state_dict(state_dict)

    ### Get Total Training Iterations ###
    samples_per_step = gradient_accumulation_steps * batch_per_device * num_devices # number of chunks processed per step across all GPUs
    training_iterations = args.epochs * (len(trainset)//samples_per_step + 1) # total training iterations in this run

    accelerator.print("="*terminal_width)
    accelerator.print("!!TRAINING DETAILS!!")
    accelerator.print(f"Using {gradient_accumulation_steps} Grad Accumulation Steps for a "
                    f"total of {gradient_accumulation_steps * tokens_per_step} Tokens Per Iteration")
    accelerator.print(f"Steps Needed to Train: {training_iterations:,}")
    accelerator.print(f"Num Warmup Steps: {int(training_iterations*args.warmup_ratio):,}")
    accelerator.print("="*terminal_width)

    ### Get Dataloaders ###
    def basic_collator(batch):
        samples = [s["input_ids"] for s in batch]
        inputs = mytorch.Tensor([s[:-1] for s in samples], dtype=mytorch.int32)
        targets = mytorch.Tensor([s[1:] for s in samples], dtype=mytorch.int32)
        return inputs, targets

    trainloader = DataLoader(trainset.dataset, batch_size=args.batch_size_per_gpu, 
                            shuffle=True, num_workers=args.num_workers, 
                            collate_fn=basic_collator)
    testloader = DataLoader(testset.dataset, batch_size=args.batch_size_per_gpu, 
                            shuffle=True, num_workers=args.num_workers, 
                            collate_fn=basic_collator)
    
    ### Load Optimizer ###
    embedding_params = []
    non_embedding_params = []
    for name, param in model.named_parameters():
        if "embeddings" in name:
            embedding_params.append(param)
        else:
            non_embedding_params.append(param)

    ### No weight decay on embeddings as inspired by SmolLM! ### 
    param_groups = [
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": non_embedding_params, "weight_decay": args.weight_decay}
    ]

    optimizer = mytorch.optim.AdamW(param_groups, args.max_learning_rate, 
                                    beta1=args.beta1, 
                                    beta2=args.beta2)

    ### Load Scheduler ###
    scheduler = mytorch.lr_scheduler.CosineLRScheduler(
        optimizer=optimizer, 
        max_lr=args.max_learning_rate,
        min_lr=args.max_learning_rate * args.min_learning_rate_ratio, 
        total_steps=training_iterations, 
        warmup_steps=training_iterations * args.warmup_ratio
    )

    ### Prepare Everything ###
    model, optimizer, trainloader, testloader = accelerator.prepare(
        model, optimizer, trainloader, testloader
    )

    ### Resume from Checkpoint ###
    if resume_from_checkpoint is not None:

        ### This is if we pass in a full path to checkpoint ###
        if os.path.exists(resume_from_checkpoint):
            path_to_checkpoint = resume_from_checkpoint
        
        ### Otherwise we can pass the folder name in our experiment directory ###
        else:
            path_to_checkpoint = os.path.join(path_to_experiment, resume_from_checkpoint)
        
        ### Load our State (model and optimizer) ###
        accelerator.load_state(path_to_checkpoint)
        
        ### Start completed steps from checkpoint index ###
        completed_steps = int(resume_from_checkpoint.split("_")[-1])
        accelerator.print(f"Resuming from Iteration: {completed_steps}")

        ### Advance our scheduler to the correct step ###
        scheduler.step_count = completed_steps

    else:
        completed_steps = 0

    ### Train Model ###
    pbar = tqdm(range(training_iterations), 
                disable=not accelerator.is_main_process(),
                initial=completed_steps)

    t0 = time.time()
    train = True
    while train:

        for inputs, targets in trainloader:

            # Move to correct device 
            inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
     
            # Forward pass
            _, loss = model(inputs, targets)

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
                t0 = t1
        
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
                    log_parts.append(f"Toks/Sec: {int(args.tokens_per_batch / dt):6d}")
                    log_statement = " | ".join(log_parts)

                    if accelerator.is_main_process():
                        tqdm.write(log_statement)

                    ### Log with Wandb if enabled ###
                    if args.log_wandb:  
                        logging_dict = {"loss": loss, "lr": lr}
                        if accelerator.grad_norm is not None:
                            logging_dict["grad_norm"] = accelerator.grad_norm 
                        accelerator.log(logging_dict, step=completed_steps)

                if completed_steps % args.checkpoint_iterations == 0:
                    accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{completed_steps}"))

                if completed_steps % args.eval_interval == 0:
                    accelerator.print("Evaluating!")
                    model.eval()
                    val_losses = []

                    for val_iter, (inputs, targets) in enumerate(testloader):
                        inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
                    
                        with mytorch.no_grad():
                            _, loss = model(inputs, targets)
                        loss_val = accelerator.gather_for_metrics(loss)
                        val_losses.append(loss_val)

                        if val_iter >= args.eval_iterations:
                            break
                    
                    ### Log Loss ###
                    if len(val_losses) > 0:  # Make sure we have some losses to compute
                        val_losses = np.mean(val_losses)
                        accelerator.print("Validation Loss:", val_losses)
                        if args.log_wandb:
                            logging_dict = {"val_loss": val_losses}
                            accelerator.log(logging_dict, step=completed_steps)

                    ### Set back into Training Mode ###
                    model.train()

            if completed_steps >= training_iterations:
                accelerator.print("Completed Training!!!")
                train = False
                break
            
    ### Save final checkpoint once done ! ###
    accelerator.save_state(os.path.join(path_to_experiment, f"final_checkpoint"), save_model_only=True)
    accelerator.end_training()

if __name__ == "__main__":
    
    args = parse_args()

    ### Get path to experiment ###
    path_to_experiment = args.work_dir
    
    ### Get Last Checkpoint from Pretraining ###
    last_pretraining_checkpoint = get_last_checkpoint(args.path_to_starting_checkpoint)
    if last_pretraining_checkpoint == -1:
        last_pretraining_checkpoint = "final_checkpoint"
    path_to_last_pretraining_checkpoint = os.path.join(
        args.path_to_starting_checkpoint, last_pretraining_checkpoint, "model.safetensors"
    )

    ### Get Last Midtraining Checkpoint If Resuming ###
    last_midtraining_checkpoint = get_last_checkpoint(path_to_experiment)
    
    if last_midtraining_checkpoint != -1:
        trainer(
            args, 
            path_to_experiment, 
            path_to_last_pretraining_checkpoint, 
            last_midtraining_checkpoint
        )

    else:
        print("Midtraining is already complete!!")
