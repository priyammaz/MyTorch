import os
import mytorch
from models.gpt2 import GPT2, GPT2Config
import argparse
import pickle
import tiktoken

def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT2 with MyTorch")

    ### Experiment Config ###
    parser.add_argument("path_to_project_dir", type=str, help="Path to project directory")
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--max_tokens_gen", type=int)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int)

    ### Inference Config ###
    parser.add_argument("--start", type=str, default="\n")

    args = parser.parse_args()

    return args

args = parse_args()

### Parse our Config File ###
path_to_model_config = os.path.join(args.path_to_project_dir, "model_meta.pkl")
with open(path_to_model_config, "rb") as f:
    model_config = pickle.load(f)

### Load Tokenizer ###
if model_config["path_to_tokenizer"] is not None:
    with open(model_config["path_to_tokenizer"], "rb") as f:
        tokenizer_meta = pickle.load(f)
    USE_TIKTOKEN = False
    tokenizer_char2idx = tokenizer_meta["char2idx"]
    tokenizer_idx2char = tokenizer_meta["idx2char"]
else:
    tokenizer = tiktoken.get_encoding("gpt2")
    USE_TIKTOKEN = True

### Load Model ###
gpt2_config = GPT2Config(
    vocab_size=model_config["vocab_size"], 
    max_seq_len=model_config["context_length"],
    embed_dim=model_config["embed_dim"], 
    num_heads=model_config["num_heads"], 
    num_blocks=model_config["num_blocks"], 
    mlp_ratio=model_config["mlp_ratio"], 
    use_bias=model_config["use_bias"], 
    use_fused_ops=args.fused
)
model = GPT2(gpt2_config)

### Get Weights ###
if os.path.exists(os.path.join(args.path_to_project_dir, "final_checkpoint")):
    path_to_model_weights = os.path.join(args.path_to_project_dir, "final_checkpoint", "model.safetensors")
else:
    if args.checkpoint_dir is None:
        if len(os.listdir(args.path_to_project_dir)) > 0:
            raise Exception(f"No final_checkpoint dir found in {args.path_to_project_dir}, and checkpoint_dir "
                            f"is not specified. The following checkpoints are available: {os.listdir(args.path_to_project_dir)} "
                            f"select the one you want with --checkpoint_dir !")
        else:
            raise Exception(f"No checkpoints found in your project directory {args.path_to_project_dir}!")

    else:
        path_to_model_weights = os.path.join(args.path_to_project_dir, args.checkpoint_dir, "model.safetensors")
        if not os.path.exists(path_to_model_weights):
            raise Exception(f"{path_to_model_weights} does not exist!")
        
state_dict = mytorch.load(path_to_model_weights)
model.load_state_dict(state_dict)
model = model.to(args.device)

### Get starting tokens for model ###
if not USE_TIKTOKEN:
    generated = [tokenizer_char2idx[c] for c in args.start]
else:
    generated = tokenizer.encode(args.start)

seed = mytorch.Tensor(generated, dtype=mytorch.int32).unsqueeze(0).to(args.device)
context = seed

# Generation loop
print(args.start, end="", flush=True)
for _ in range(args.max_tokens_gen if args.max_tokens_gen is not None else model_config["context_length"]):
    seed = mytorch.Tensor(
        generated[-model_config["context_length"]:], dtype=mytorch.int32
    ).unsqueeze(0).to(args.device)

    with mytorch.no_grad():
        pred = model(seed)

    # Get logits for last position
    last_logits = pred[:, -1, :]

    ### Scale by temperature ###
    last_logits = last_logits / args.temperature

    if args.topk is not None and args.topk > 0:

        # Get top-k indices (the top-k largest logits) ###
        topk_idx = last_logits.argsort(descending=True)[:, :args.topk]

        ### Replace all other logits with -inf ###
        mask = mytorch.full_like(last_logits, float("-inf"))
        mask[:, topk_idx] = last_logits[:, topk_idx]
        last_logits = mask

    # Random sample
    exp_logits = mytorch.exp(last_logits - mytorch.max(last_logits))
    probs = exp_logits / mytorch.sum(exp_logits)
    next_id = mytorch.multinomial(probs, num_samples=1)[0].numpy().item()

    # Append to generated sequence
    generated.append(next_id)

    # Decode and print the latest token
    if not USE_TIKTOKEN:
        token = tokenizer_idx2char[next_id]
    else:
        token = tokenizer.decode([next_id])

    print(token, end="", flush=True)

print("\n")