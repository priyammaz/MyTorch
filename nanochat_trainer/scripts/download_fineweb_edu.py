"""
This just downloads the FineWebEDU split, each parquet file is ~ 2.15 GB of data and about 750 Million Tokens
"""
import os
import requests
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from nanochat_trainer.core.nanochat_gpt import GPT, GPTConfig

TOKENS_PER_FILE = 750_000_000 # not on our tokenizer but based on the dataset "token_count" column
ROOT_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT/"

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_save", default="data/FineWebEDU/raw_data", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--chinchilla_ratio", default=20, type=float)

    parser.add_argument("--vocab_size", default=2**16, type=int)
    parser.add_argument("--context_length", default=2048, type=int)
    parser.add_argument("--num_blocks", default=20, type=int)
    parser.add_argument("--embed_dim", default=1280, type=int)
    parser.add_argument("--num_q_heads", default=10, type=int)
    parser.add_argument("--num_kv_heads", default=10, type=int)
    parser.add_argument("--mlp_ratio", default=4, type=int)

    args = parser.parse_args()
    
    return args

def get_num_params(args):

    config = GPTConfig(
        vocab_size=args.vocab_size, 
        sequence_length=args.context_length, 
        embed_dim=args.embed_dim, 
        mlp_ratio=args.mlp_ratio, 
        num_blocks=args.num_blocks,
        num_q_heads=args.num_q_heads, 
        num_kv_heads=args.num_kv_heads
    )

    model = GPT(config)
    
    total = 0
    for name, param in model.named_parameters():
        total += np.prod(param.shape)

    print(f"This Model Has {total:,} Parameters")
    return total

def compute_chincilla(num_params, chinchilla_ratio=20, just_a_little_extra=1.2):
    """
    chinchilla recommends roughly 20 times the tokens as we have parameters

    We also just grab a little extra data as our grouping will drop some and we need a train/test split
    so we have some buffer
    """
    wanted_tokens = int(num_params * chinchilla_ratio * just_a_little_extra)
    
    print(f"With the selected Chinchilla Ratio {chinchilla_ratio}, we need {wanted_tokens:,} Tokens for Training!")

    return wanted_tokens

def generate_parquet_names(n_files):
    """
    fineweb dataset parquet files pattern:
    000_00000.parquet
    000_00001.parquet
    ...
    000_00009.parquet
    001_00000.parquet
    ...

    So given how many files we want we can generate the names here

    """
    names = []
    for i in range(n_files):
        group1 = i // 10        # increments every 10 files
        group2 = i % 10         # cycles 0–9
        names.append(f"{group1:03d}_{group2:05d}.parquet")
    return names

def how_many_files(num_params, chinchilla_ratio=20):
    recommended_tokens = compute_chincilla(num_params, chinchilla_ratio) * 1.1 # Give a little buffer
    if recommended_tokens > 100_000_000_000:
        raise Exception("We are processing the 100B split of FineWeb!")
    return int((recommended_tokens // TOKENS_PER_FILE) + 1)

def download_file(args):
    url, local_filename = args
    local_path = Path(local_filename)
    local_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
        
        return f"SUCCESS: {local_path.name}"
    
    except requests.exceptions.RequestException as e:
        return f"FAILED: {local_path.name} ({url}) -> {e}"
    except Exception as e:
        return f"ERROR: {local_path.name} -> {e}"
    
def download_files_parallel(file_names, save_dir, max_workers=8):

    # Build tasks only for missing files
    tasks = []
    skipped = 0
    
    for file in file_names:
        save_path = os.path.join(save_dir, file)
        if os.path.exists(save_path):
            print(f"Skipped (already exists): {file}")
            skipped += 1
        else:
            tasks.append((ROOT_URL + file, save_path))
    
    if not tasks:
        print(f"\nAll {len(file_names)} files already exist. Nothing to download!")
        return
    
    print(f"\nStarting download of {len(tasks)} files "
          f"({skipped} already exist) using {min(max_workers, len(tasks))} workers...\n")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(download_file, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            print(future.result())
    
    print("\nAll downloads finished!")
if __name__ == "__main__":
    print("-"*50)
    print("Downloading FineWeb!")
    print("-"*50)

    args = parse_args()

    total_params = get_num_params(args)

    os.makedirs(args.path_to_save, exist_ok=True)
    num_files = how_many_files(total_params, args.chinchilla_ratio)
    file_names = generate_parquet_names(num_files)
    download_files_parallel(file_names, args.path_to_save, args.num_workers)