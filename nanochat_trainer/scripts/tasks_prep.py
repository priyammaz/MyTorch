"""
This is a collection of the different huggingface datasets for the different
downstream tasks we are interested in! Each dataset has a slightly different prep
so we will just write them all out here!

The Datasets we will do are:

- Arc: Multiple Choice Questions
- GSM8k: Basic arithmetic (will call our calculator tool)
- HumanEval: Coding benchmark
- SmolTalk: Coversation

Again, inspiration from NanoChat! 
https://github.com/karpathy/nanochat/tree/master/tasks

All data needs to first be parsed into:

{
    "messages": [
        {"role": "system", "content": ...}
        {"role": "user", "content": ...},
        {"role": "assistant", "content": ...},
        ...
    ]
}

And for ToolCalling in GSM8k we need:

{
    "messages": [
        {"role": "system", "content": ...}
        {"role": "user", "content": ...},
        {"role": "user", "content": [
            {"type": "python", "text": "...some expression to eval with python"},
            {"type": "python_output", "text": "...result from the toolcall"}
        ]} 
    ]
}

The output of every prep will be:

- Input Ids (the actual tokens)
- Mask (which tokens are valid to compute loss on)

"""
import os
import argparse
from datasets import load_dataset
from nanochat_trainer.core.tokenizer import MyTokenizer

class GenericDatasetPrep:
    def prep_sample(self):
        raise NotImplementedError
    def prepare(self):
        raise NotImplementedError

class SmolTalkPrep(GenericDatasetPrep):
    """
    HuggingFaceTB/smol-smoltalk

    460K samples of just general conversation data!

    This dataset is simple as it is already in the format that we want it!
    """ 
    def __init__(self, 
                 path_to_store,
                 tokenizer):
        
        self.dataset = load_dataset("HuggingFaceTB/smol-smoltalk").remove_columns("source")
        self.path_to_store = os.path.join(path_to_store, "smoltalk")
        self.tokenizer = tokenizer
    
    def prep_sample(self, sample):
        
        input_ids, mask = self.tokenizer.parse_conversation(sample)
        sample["input_ids"] = input_ids
        sample["mask"] = mask
        
        return sample
        
    
    def prepare(self, num_workers):
        """
        Lets tokenize and store this dataset
        """

        tokenized = self.dataset.map(self.prep_sample, remove_columns=["messages"], num_proc=num_workers)
        tokenized.save_to_disk(self.path_to_store)

class ArcPrep(GenericDatasetPrep):
    """
    Arc is a dataset of multiple choice questions. Each sample looks like:

    {
        "answerKey": "B",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["Shady areas increased.", "Food sources increased.", "Oxygen levels increased.", "Available water increased."]
        },
        "id": "Mercury_SC_405487",
        "question": "One year, the oak trees in a park began producing more acorns than usual. The next year, the population of chipmunks in the park also increased. Which best explains why there were more chipmunks the next year?"
    }

    We will have to convert this to our own format so it will look like:
    
    Multiple Choice Question: One year, the oak trees in a park began producing more acorns than usual. The next year, the population of chipmunks in the park also increased. Which best explains why there were more chipmunks the next year?
    
    - Shady areas increased.=A
    - Food sources increased.=B
    - Oxygen levels increased.=C
    - Available water increased.=D

    Respond only with the letter of the correct answer.

    The reason the letter comes after the anwser is to follow the render_mc method in nanochat
    https://github.com/karpathy/nanochat/blob/master/tasks/common.py, where it seems that smaller models
    have better binding if we do it this way instead!

    There are two splits: Arc-Challenge and Arc-Easy so we can add an option here to process both
    And each split also has a train/test/val
    
    """
    def __init__(self, 
                 split,
                 path_to_store,
                 tokenizer):
        
        assert split in ["ARC-Challenge", "ARC-Easy"]

        if split == "ARC-Challenge":
            save_name = "arc_challenge"
        else:
            save_name = "arc_easy"

        self.dataset = load_dataset("allenai/ai2_arc", split)
        self.path_to_store = os.path.join(path_to_store, save_name)
        self.tokenizer = tokenizer

    def prep_sample(self, sample):

        question = sample["question"]
        choices_text = sample["choices"]["text"]
        choices_labels = sample["choices"]["label"]
        assistant_message = sample["answerKey"].strip() # just a letter!
    
        user_message = ""
        user_message += f"Multiple Choice Question: {question}\n"
        for text, label in zip(choices_text, choices_labels):
            choice = f"- {text}={label}\n"
            user_message += choice
        user_message += "\nRespond only with the letter of the correct answer."
        
        prepped_sample = {
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]

        }

        input_ids, mask = self.tokenizer.parse_conversation(prepped_sample)

        sample["input_ids"] = input_ids
        sample["mask"] = mask     

        return sample 


    def prepare(self, num_workers):
        
        columns_to_drop = self.dataset.column_names["train"]
        tokenized = self.dataset.map(self.prep_sample, remove_columns=columns_to_drop, num_proc=num_workers)
        tokenized.save_to_disk(self.path_to_store)

class GSM8kPrep(GenericDatasetPrep):
    pass

class HumanEvalPrep(GenericDatasetPrep):
    pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_store", default="data/tasks")
    parser.add_argument("--path_to_tokenizer", default="nanochat_trainer/nanochat_tokenizer")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # Get tokenizer 
    tokenizer = MyTokenizer(os.path.join(args.path_to_tokenizer, "tokenizer.json"))

    # Init all datasets so they just download and are prepped #
    # smol = SmolTalkPrep(args.path_to_store, tokenizer).prepare(args.num_workers)
    arc_easy = ArcPrep("ARC-Easy", args.path_to_store, tokenizer).prepare(args.num_workers)
    



