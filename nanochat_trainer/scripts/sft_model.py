"""
SFT Is basically the same thing as Midtraining again, but with three differences: 

1) Midtraining is when we shift our model from raw text to conversations, so we need a bunch of data for this,
   and we arent too concerned about topic specificity or anything. At this stage of midtraining, its really just
   more the merrier, though its helpful to provide task/topic specific token. SFT will be our main final stage of 
   training (except for RL which we may do later). In this case we want to make sure we use the highest quality 
   data possible. An example here is in midtraining we used the much larger MMLU, but in SFT we will use the higher quality ARC datasets. 

2) In Midtraining our loss was computed on ALL tokens. This includes all our special tokens. This was again important 
   as the model just needs to understand the conversational format. But in reality, we dont actually care about computing 
   loss on these tokens as these are provided in our chat template. The only special tokens we really care about are 
   those related to tool calling (like python start/end) and our assistant end token which is how the model tells us
   its done generating. This is why in our tokenizer parse_conversation method, we produce a mask which tells us which
   tokens to compute loss on and which to not compute loss on. We want to compute loss ONLY ON THE COMPLETIONS, basically
   only on the piece that we expect the model to produce the rest is just context we condtion that generation on!  

3) In Midtraining we simply packed examples together. This means multiple conversations were concatenated together until
   that sample hit the context length we wanted. But now each sample will only contain a single conversation! This means  
   we now need to pad our shorter conversation to the longer ones. We will just use the <|assistant_end|> token to do this
   as we never created a dedicated <|pad|> token. We will not compute loss on these extra pad tokens, but using an attention
   mask is a little optional. Technically pad tokens are extra stuff that we dont want to attend to in our attention computation
   but also our assistant conversation should ALWAYS END with <|assistant_end|>. There will be one <|assistant_end|>  at the 
   end of the conversation that we do compute loss on (the rest are ignored), but technically it should be fine to just not 
   use an attention mask as its basically just learning the positional information that <|assistant_end|> comes after everything else

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
from nanochat_trainer.core.tasks import Task, MixtureDataset
from nanochat_trainer.core.tokenizer import MyTokenizer

import warnings

terminal_width = shutil.get_terminal_size().columns

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--path_to_starting_checkpoint", type=str, required=True) # this is new, we need to start from our pretrained checkpoint
    parser.add_argument("--path_to_data", type=str, default="data/tasks")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--examples_per_batch", type=int, default=32)
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

    ### Path to Checkpoint ###
    parser.add_argument("--path_to_tokenizer", type=str)

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
                                 run_name="sft",
                                 config=vars(args))
        
    ### Get path to work dir ###
    if not os.path.isdir(path_to_experiment) and accelerator.is_main_process:
        accelerator.print("Creating Working Directory:", path_to_experiment)
        os.makedirs(path_to_experiment, exist_ok=True)
    
    ### Setup Dataset ###
    trainset = MixtureDataset([
        Task(args.path_to_data, "arc_easy", keep_mask=True), # 5.2K samples of multiple choice
        Task(args.path_to_data, "arc_challenge", keep_mask=True), # 2.3K samples of harder multiple choice
        Task(args.path_to_data, "gsm8k", keep_mask=True), # 8K samples of arithmetic
        Task(args.path_to_data, "smoltalk", num_samples=10000) # 10k samples of general converation
    ])

    testset = MixtureDataset([
        Task(args.path_to_data, "smoltalk", "test", keep_mask=True), 
    ])

    ### Compute Gradient Accumulation Steps ###
    num_devices = accelerator.num_processes
    examples_proc_per_step = num_devices * args.batch_size_per_gpu
    if examples_proc_per_step > args.examples_per_batch:
        batch_size_per_gpu = args.examples_per_batch // num_devices
        gradient_accumulation_steps = 1
        accelerator.print(f"Using {num_devices} GPUs each with {args.batch_size_per_gpu} exceeds wanted batch size of {args.examples_per_batch}")
        accelerator.print(f"Setting per gpu batch size to {batch_size_per_gpu}")
    else:
        batch_size_per_gpu = args.batch_size_per_gpu
        gradient_accumulation_steps = args.examples_per_batch // examples_proc_per_step

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
    training_iterations = args.epochs * (len(trainset)//args.examples_per_batch + 1) # total training iterations in this run

    accelerator.print("="*terminal_width)
    accelerator.print("!!TRAINING DETAILS!!")
    accelerator.print(f"Using {gradient_accumulation_steps} Grad Accumulation Steps")
    accelerator.print(f"Steps Needed to Train: {training_iterations:,}")
    accelerator.print(f"Num Warmup Steps: {int(training_iterations*args.warmup_ratio):,}")
    accelerator.print("="*terminal_width)

    ### Get Dataloaders ###
    def basic_collator(batch):
        
        samples = [s["input_ids"] for s in batch]
        masks = [s["mask"] for s in batch]

        ### Get the longest sample ###
        len_samples = [len(s) for s in samples]
        max_len = max(len_samples)

        ### Create Empty Tensors to Populate ###
        samples_ = mytorch.full(len(samples), max_len, fill_value=PAD_TOKEN_ID, dtype=mytorch.int32) # All pad
        masks_ = mytorch.full(len(samples), max_len, fill_value=0, dtype=mytorch.int32) # all ignore

        ### Populate Tensors with Data ###
        for i, (s, m) in enumerate(zip(samples, masks)):
            samples_[i, :len_samples[i]] = s
            masks_[i, :len_samples[i]] = m

        ### Get our Inputs and Targets ###
        inputs = samples_[:, :-1].clone()
        targets = samples_[:, 1:].clone()

        ### Get the Target Mask ###
        target_masks = masks_[:, 1:]

        ### Fill targets with IGNORE_INDEX at masked location ###
        targets[target_masks==0] = IGNORE_INDEX

        ### Truncate Inputs and Targets to Max Context Length ###
        max_ctx = args.context_length
        if inputs.shape[1] > max_ctx:
            inputs  = inputs[:, :max_ctx]
            targets = targets[:, :max_ctx]

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

    ### Set some GLOBAL Variables ###
    IGNORE_INDEX = -100 # fused Cross Entropy uses hardcoded -100 as its ignore index
    PAD_TOKEN_ID = MyTokenizer(
        os.path.join(args.path_to_tokenizer, "tokenizer.json")
    ).assistant_end_id

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
        print("SFT is already complete!!")
