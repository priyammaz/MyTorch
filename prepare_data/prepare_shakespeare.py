"""
Prep for character level shakespear model identical to NanoGPT
https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
"""

import os
import pickle
import requests
import numpy as np
import argparse

def prep_shakespeare(path_to_store, test_split_pct=0.1):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    all_txt = requests.get(data_url).text
    n = len(all_txt)
    print(f"Loaded {n} Characters of Data")

    ### Get all unique chars ###
    chars = sorted(list(set(all_txt)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    ### Quick Tokenizers ###
    char2idx = {c:i for i,c in enumerate(chars)}
    idx2char = {i:c for (c,i) in char2idx.items()}
    
    ### Quick encoding ###
    def encode(s):
        return [char2idx[c] for c in s]
    
    ### Train/Test Split ###
    train = all_txt[:int(n*(1-test_split_pct))]
    test = all_txt[int(n*(1-test_split_pct)):]

    train_ids = encode(train)
    val_ids = encode(test)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    ### Store Tokenizer ###    
    meta = {
        "vocab_size": vocab_size,
        "char2idx": char2idx,
        "idx2char": idx2char
    }

    ### Store in Bin Files ###
    if not os.path.exists(path_to_store):
        os.makedirs(path_to_store, exist_ok=True)
    
    path_to_train = os.path.join(path_to_store, "train.bin")
    path_to_test = os.path.join(path_to_store, "val.bin")
    path_to_tokenizer_pkl = os.path.join(path_to_store, "tokenizer.pkl")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(path_to_train)
    val_ids.tofile(path_to_test)

    with open(path_to_tokenizer_pkl, "wb") as f:
        pickle.dump(meta, f)
     
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Shakespear Data Preparation")
    parser.add_argument("--path_to_save", default="data/shakespeare")
    parser.add_argument("--test_split_pct", type=float, default=0.1)

    args = parser.parse_args()

    prep_shakespeare(args.path_to_save, args.test_split_pct)


