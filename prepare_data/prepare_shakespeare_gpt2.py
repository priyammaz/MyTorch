"""
Prep for character level shakespear model identical to NanoGPT
https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
"""

import os
import requests
import numpy as np
import argparse
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def prep_shakespear_gpt2(path_to_store, test_split_pct=0.1):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    all_txt = requests.get(data_url).text

    ### Encode Data ###
    all_txt = enc.encode_ordinary(all_txt)

    ### Train/Test Split ###
    train_ids = all_txt[:int(len(all_txt)*(1-test_split_pct))]
    val_ids = all_txt[int(len(all_txt)*(1-test_split_pct)):]

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    ### Store in Bin Files ###
    if not os.path.exists(path_to_store):
        os.makedirs(path_to_store, exist_ok=True)
    
    path_to_train = os.path.join(path_to_store, "train.bin")
    path_to_test = os.path.join(path_to_store, "val.bin")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(path_to_train)
    val_ids.tofile(path_to_test)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Shakespear Data Preparation")
    parser.add_argument("--path_to_save", default="data/shakespeare_gpt2")
    parser.add_argument("--test_split_pct", type=float, default=0.05)

    args = parser.parse_args()

    prep_shakespear_gpt2(args.path_to_save, args.test_split_pct)


