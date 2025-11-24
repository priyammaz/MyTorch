"""
Simple script to pre-tokenize and save our pretraining dataset
"""
import os
import json
from itertools import chain
import argparse
from nanochat_trainer.core.tokenizer import MyTokenizer
from datasets import load_dataset, disable_caching

### No need to store all the intermediates here, as we will save the dataset in the end ###
disable_caching()

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_tokenizer", type=str, default="nanochat_trainer/nanochat_tokenizer/tokenizer.json")
    parser.add_argument("--path_to_data", type=str, default="data/FineWebEDU/raw_data")
    parser.add_argument("--path_to_save", type=str, default="data/FineWebEDU/tokenized")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--test_split_pct", type=float, default=0.005)

    args = parser.parse_args()
    return args

def tokenize_dataset(args):

    ### Load tokenizer ###
    tokenizer = MyTokenizer(os.path.join(args.path_to_tokenizer, "tokenizer.json"))
    
    ### Get Paths to Downloaded Parquet Files ###
    parquet_files = [os.path.join(args.path_to_data, f) for f in os.listdir(args.path_to_data) if "parquet" in f]
    training_files, testing_file = parquet_files[:-1], parquet_files[-1]
    
    ### Load Dataset ###
    dataset = load_dataset("parquet", data_files={"train": training_files, "test": testing_file}, num_proc=args.num_workers)
    dataset = dataset.select_columns("text")
    print(dataset)
    
    ### Tokenize Dataset
    def tokenize_samples(batch):
        texts = batch["text"]
        tokens = tokenizer.batch_encode(texts, prepend=tokenizer.bos_token)
        batch["input_ids"] = tokens
        return batch
    
    tokenized_datasets = dataset.map(
        tokenize_samples, 
        batched=True, 
        num_proc=args.num_workers, 
        remove_columns="text",
        desc=f"Tokenizing"
    )

    ### Chunk dataset
    max_seq_len = args.max_seq_len + 1 # we add 1 so we can have 2048 inputs and 2048 next token targets
                                       # as we will take each sequence and convert to inputs/targets
    def group_texts(examples):
   
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= max_seq_len:
            total_length = (total_length // max_seq_len) * max_seq_len

        result = {
            k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers,
        desc=f"Grouping into chunks of {args.max_seq_len}",
    )

    ### Shuffle Dataset ###
    tokenized_datasets = tokenized_datasets.shuffle(seed=42)

    train_split = tokenized_datasets["train"]
    test_split = tokenized_datasets["test"]

    print("Total Training Tokens:", f"{len(train_split) * args.max_seq_len:,}")
    print("Total Testing Tokens:", f"{len(test_split) * args.max_seq_len:,}")
    
    print(f"Saving {args.path_to_save}")
    path_to_train = os.path.join(args.path_to_save, "train")
    path_to_test = os.path.join(args.path_to_save, "test")
    train_split.save_to_disk(path_to_train, max_shard_size="2GB")
    test_split.save_to_disk(path_to_test, max_shard_size="2GB")

    ### Save some metadata about how many parquet files were needed to build this dataset ###
    ### This way if we download more files we can trigger a rebuild ###
    meta = {"num_parquet_files": len(parquet_files)}
    with open(os.path.join(args.path_to_save, "data_meta.json"), "w") as f:
        json.dump(meta, f)
    
if __name__ == "__main__":
    print("-"*50)
    print("Preparing Fineweb!")
    print("-"*50)

    args = parse_args()
    
    if os.path.exists(os.path.join(args.path_to_save, "train")) and os.path.exists(os.path.join(args.path_to_save, "test")):

        ### If it exists but we downloaded more parquet files we can check here as the metadata contains how many files were used ! ###
        path_to_meta = os.path.join(args.path_to_save, "data_meta.json")
        with open(path_to_meta, 'r') as file:
            metadata = json.load(file)
        num_files_from_prev = metadata["num_parquet_files"]

        parquet_files = [os.path.join(args.path_to_data, f) for f in os.listdir(args.path_to_data) if "parquet" in f]
        if num_files_from_prev == len(parquet_files):
            print("Dataset already exists! Skipping!!")
        else:
            print("Detected New Parquet Files, Re-Tokenizing and Saving the Dataset!!")
            tokenize_dataset(args)
    else:
        tokenize_dataset(args)