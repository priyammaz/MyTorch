"""
This just downloads the FineWebEDU split, each parquet file is ~ 2.15 GB of data and about 750 Million Tokens
"""
import os
import requests
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

TOKENS_PER_FILE = 750_000_000 # not on our tokenizer but based on the dataset "token_count" column
ROOT_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/100BT/"

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_model_parameters", default=500_000_000, type=int)
    parser.add_argument("--path_to_save", default="data/FineWebEDU/raw_data", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    
    args = parser.parse_args()
    return args

def compute_chincilla(num_params):
    """
    chinchilla recommends roughly 20 times the tokens as we have parameters
    """
    return num_params * 20

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

def how_many_files(num_params):
    recommended_tokens = compute_chincilla(num_params) * 1.1 # Give a little buffer
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
    os.makedirs(args.path_to_save, exist_ok=True)

    num_files = how_many_files(args.num_model_parameters)
    file_names = generate_parquet_names(num_files)
    download_files_parallel(file_names, args.path_to_save, args.num_workers)