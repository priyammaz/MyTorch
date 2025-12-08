"""
scripts/tasks_prep.py has created a bunch of tokenized datasets that each have a train/test split
but we need to actually load them! There are two ways we will do this:

1) MidTraining - This is the same as pretraining, but we do it on a bunch of conversation type datasets
                 rather than raw text like we did. This get the model to start understanding the conversation
                 format. To do this we just create blocks of size 2048 (our context length) out of our conversation
                 datasets and train! We also train on ALL tokens. Although we have a mask we will ignore those here
                 as we just want our model to understand the conversational format

2) SFT - Again the same, but this this each sample will be one conversation. Where before we could have multiple conversations
         concatenated together in one, we just do one at a time here. This also means we pad our data! Bunch of ways to do this
         but the easiest is just pad to the max context length. We will not compute loss on the padding anyway, and this keeps 
         the length from changing every batch so our triton autotune only need to run once! We also will not compute loss
         on the masked tokens (things like assistant_start or on queries, just on completions)

"""

import os
from itertools import chain
from datasets import Dataset, load_from_disk, concatenate_datasets
import warnings

_supported_datasets = ("smoltalk", "arc_easy", "arc_challenge", "mmlu", "gsm8k")

class Task:
    """
    At this point the scripts/tasks_prep.py should have been run. So we have in our data/tasks folder (or wherever you set to save)
    our tokenized tasks data. This is a simple loader to load that data!
    """
    def __init__(self, 
                 path_to_root, 
                 dataset,
                 split="train",
                 num_samples=None,
                 seed=42,
                 keep_mask=True,):
        
        assert dataset in _supported_datasets, f"Select from Supported Datasets: {_supported_datasets}"

        self.path_to_dataset = os.path.join(path_to_root, dataset)
        self.num_samples = num_samples
        self.seed = seed   
        
        ### Load Dataset ###
        self.dataset = load_from_disk(self.path_to_dataset)[split]
        
        ### Remove mask column if we dont need it ###
        if not keep_mask:
            self.dataset = self.dataset.remove_columns("mask")

        ### Shuffle Dataset ###
        self.dataset = self.dataset.shuffle(seed=seed)

        ### Keep Wanted Samples ###
        if num_samples is not None:
            if num_samples > len(self.dataset):
                warnings.warn(f"Requested {num_samples} from {dataset}, but there are only {len(self.dataset)} samples, using everything!")
            else:
                self.dataset = self.dataset.select(range(num_samples))

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return str(self.dataset)

class MixtureDataset:
    """
    We can provide what datasets we want here as a list and it will all be concatenated together!
    """
    def __init__(self, datasets, seed=42):

        # Quickly extract the dataset from the Task objects here
        unwrapped = []  
        for d in datasets:
            if isinstance(d, Task):
                unwrapped.append(d.dataset)
            elif isinstance(d, Dataset):
                unwrapped.append(d)
            else:
                raise Exception(f"Unexpected object of type {type(d)} passed in")

        ### Concatenate and shuffle ###
        self.dataset = concatenate_datasets(unwrapped).shuffle(seed=seed)

    def __repr__(self):
        return str(self.dataset)
    
    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
class GroupedDataset:
    """
    Groups dataset into equal length chunks of text
    """
    
    def __init__(self, 
                 dataset, 
                 chunk_size=2048,
                 num_proc=8,
                 seed=42):
     
        ### Unwrap if we need to 
        if isinstance(dataset, (MixtureDataset, Task)):
            dataset = dataset.dataset

        self.chunk_size = chunk_size
        self.dataset = dataset

        self.dataset = self.dataset.map(
            self.group_texts,
            batched=True,
            num_proc=num_proc,
            desc=f"Grouping into chunks of {chunk_size}",
        )

        ### Shuffle Dataset ###
        self.dataset = self.dataset.shuffle(seed=seed)

    def group_texts(self, examples):
   
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= self.chunk_size:
            total_length = (total_length // self.chunk_size) * self.chunk_size

        result = {
            k: [t[i : i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return str(self.dataset)

class PadDataset:
    """
    For SFT we need the model to know when to stop, we will just have every 
    question/answer pair seperate and padded to the max length! By default
    we dont have a pad token in our model so we can just use the <|assistant_start|>
    token as our tokenizer produces a mask and this is one of the tokens that we
    mask out (as we dont compute loss on tokens we manually insert!)
    """
    pass

if __name__ == "__main__":

    d = [Task("data/tasks", "smoltalk"),
         Task("data/tasks", "arc_easy"),
         Task("data/tasks", "arc_challenge")]

    GroupedDataset(MixtureDataset(datasets=d))