import requests
import numpy as np

def prep_shakespeare(test_split_pct=0.1):
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

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    return train_ids, val_ids, meta

import random

def sample(arr, num_tokens=256):
    # need one extra token for next-token prediction
    num_sample = num_tokens + 1

    if len(arr) < num_sample:
        raise ValueError("Array is shorter than the requested sample length")

    # random start index
    start = random.randint(0, len(arr) - num_sample)

    chunk = arr[start : start + num_sample]

    # inputs and targets (offset by 1)
    x = chunk[:-1]
    y = chunk[1:]

    return x, y


