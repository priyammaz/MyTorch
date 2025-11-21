"""
Simple script to pre-tokenize and save our pretraining dataset
"""
import os
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
    
    ### Load Dataset ###
    dataset = load_dataset("parquet", data_dir=args.path_to_data, num_proc=args.num_workers)["train"]
    dataset = dataset.select_columns("text")
    
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

    ### Random Split ###
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=args.test_split_pct, seed=42)

    train_split = tokenized_datasets["train"]
    test_split = tokenized_datasets["test"]

    print("Total Training Tokens:", f"{len(train_split) * args.max_seq_len:,}")
    print("Total Testing Tokens:", f"{len(test_split) * args.max_seq_len:,}")
    
    print(f"Saving {args.path_to_save}")
    path_to_train = os.path.join(args.path_to_save, "train")
    path_to_test = os.path.join(args.path_to_save, "test")
    train_split.save_to_disk(path_to_train, max_shard_size="2GB")
    test_split.save_to_disk(path_to_test, max_shard_size="2GB")
    
    dataset.cleanup_cache_files()
    tokenized_datasets.cleanup_cache_files()
    
if __name__ == "__main__":
    print("-"*50)
    print("Preparing Fineweb!")
    print("-"*50)

    args = parse_args()
    tokenize_dataset(args)