"""
Pretraining on ~10B tokens of FineWeb with a ~500M param model!
"""
import os
import argparse
import mytorch
from mytorch.utils.data import DataLoader
from mytorch.accelerate import Accelerator
from datasets import load_from_disk
from tqdm import tqdm

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data", type=str, default="data/FineWebEDU/tokenized")
    args = parser.parse_args()
    
    return args

### Load Arguments ###
args = parse_args()

### Load Data Splits ###
path_to_train = os.path.join(args.path_to_data, "train")
path_to_test = os.path.join(args.path_to_data, "test")

trainset = load_from_disk(path_to_train)
testset = load_from_disk(path_to_test)

def basic_collator(batch):
    samples = [s["input_ids"] for s in batch]
    inputs = mytorch.Tensor([s[:-1] for s in samples], dtype=mytorch.int32)
    targets = mytorch.Tensor([s[1:] for s in samples], dtype=mytorch.int32)
    return inputs, targets

import mytorch
# trainset = mytorch.arange(100)
trainloader = DataLoader(trainset, batch_size=8, num_workers=16, collate_fn=basic_collator, shuffle=False) # no need to shuffle, data is preshuffled already!
import time
for inputs, targets in tqdm(trainloader):
    print(inputs.shape, targets.shape)
    break



if __name__ == "__main__":
    pass

    