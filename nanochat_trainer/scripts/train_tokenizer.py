"""
To make this as simple as possible, we will just grab an existing tokenizer and train it 
on our own dataset!
"""
import re
import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import Regex, pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
from tqdm import tqdm

from .sample_text import general_text, code_text, verification_text, japanese_text

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

### Regex pattern close to gpt4 taken from https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py
### main change is this limits number tokens to be upto 2 digits rather than 3

### '(?i:[sdmt]|ll|ve|re) -> matches things like 's, 'd, 'm, 't, 'll, 've, 're and its case insensitive (?i) so don't → don + 't
### [^\r\n\p{L}\p{N}]?+\p{L}+ -> matches somthing that start doesnt start with (because ^) with letter(L)/digit(N)  
###                              or not a newline character (\n) or carriage return (\r) followed by one ore more letters
###                              has only zero or one of such instances (?+) and is followe by one or more letters \p{L}+
###                              an example is @gmail or #HelloWorld will all stay together. Its a common pattern in web based text so we capture it here
### \p{N}{1,2} -> numbers upto 2 digits (gpt4 has 3, but we can limit the number of tokens consumed by digits here)
### ?[^\s\p{L}\p{N}]++[\r\n]* -> optionally match (?) somethign that is not (^) space (\s) a letter (\p{L}) or a digit (\p{N}]) 
###                              and it can be one or more (++) followed by an optional newline or carriger ([\r\n])
### \s*[\r\n] -> match any whitespace (\s*) then require exactly 1 newline ([\r\n])
### \s+(?!\S) -> any whitespace (\s+) until the next character is not a white space (?!\S), so just match one or more whitespace characters that 
###              that are followed by either the end of line string or another whitespace, but not a nonwhitespace character
### \s+ -> matches any whitespace with one or more occurance
### | -> or, each of these groups are valid, so we capture them from our raw text. 

### Basically this is an alternative to just simple space separation. We want to split our text in more meanigful ways.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def visible_len(s):
    return len(re.sub(r'\x1b\[[0-9;]*m', '', s))

def pad_visible(s, width, align='<'):
    visible = visible_len(s)
    padding = width - visible
    if padding <= 0:
        return s
    if align == '<': 
        return s + ' ' * padding
    else:               
        return ' ' * padding + s
    
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison_tokenizer", type=str, default="gpt2")
    parser.add_argument("--path_to_dataset", type=str, default="data/FineWebEDU/raw_data")
    parser.add_argument("--vocab_size", type=int, default=2**16)
    parser.add_argument("--path_to_save_tokenizer", type=str, default="nanochat_trainer/nanochat_tokenizer")
    
    args = parser.parse_args()
    return args

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|system_start|>", 
    "<|system_end|>",
    "<|user_start|>", 
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>", 
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>"
]

def compute_ratio(tokenizer, text):
    encoded = tokenizer.encode(text).ids
    decoded = tokenizer.decode(encoded)
    assert text == decoded

    encoded_bytes = text.encode('utf-8')
    ratio = len(encoded_bytes) / len(encoded)
    return ratio

def train_tokenizer(path_to_dataset, 
                    comparison_tokenizer="HuggingFaceTB/SmolLM3-3B", 
                    vocab_size=2**16,
                    path_to_save=None,
                    num_workers=8,
                    training_characters=2000000000,
                    chunk_size=1000):

    comparison_tokenizer = Tokenizer.from_pretrained(comparison_tokenizer)
    if path_to_save is None:
        path_to_save = "mytokenizer"

    ### Start from an existing Tokenizer ###
    tokenizer = Tokenizer(
        BPE(
            byte_fallback=True, # if our tokenizer sees a word not in its dictionary, then it uses the bytefallback trick, 
                               # allowing us to skip having an explicit <|unk|> token for OOV, basically just represent the unkown
                               # as its raw bytes (utf-8) 
            unk_token=None, # dont need an unk anymore
            fuse_unk=False # whether to fuse consecutive unknowns into one
        )
    )

    ### Dont need to normalize text as we are processing utf8 anyway 
    tokenizer.normalizer = None

    ### Update the Pretokenizer so we limit our number extration to 2 digits at the most ###
    pretokenizer = pre_tokenizers.Sequence(
        [   ### isolated means when we split something the matched pattern is its own token and the text around it also becomes its own tokens
            ### this way nothing is removed or lost!
            pre_tokenizers.Split(pattern=Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
        ]
    )
    
    ### Set the pretokenizer with the new regex pattern ###
    tokenizer.pre_tokenizer = pretokenizer

    ### Set the bytelevel decoder ###
    tokenizer.decoder = decoders.ByteLevel()

    ### No need for any post processors (we will build the template manually later) ###
    tokenizer.post_processor = None

    ### Get trainer ###
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True, 
        min_frequency=0, 
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), 
        special_tokens=SPECIAL_TOKENS
    )

    ### Get Dataset ###
    print("Loading Dataset")
    training_dataset = load_dataset("parquet", data_dir=path_to_dataset, num_proc=num_workers)["train"].shuffle(seed=42)

    cumulative_chars = 0
    num_rows = 0

    pbar = tqdm(range(training_characters), leave=False)
    for i, row in enumerate(training_dataset):
        cumulative_chars += len(row['text'])
        pbar.update(len(row['text']))
        if cumulative_chars >= training_characters:
            num_rows = i + 1
            break

    ### Keep first 100 (in training split) and last 100 (not in training split) rows for non-training eval ###
    training_eval = "".join(training_dataset[:100]["text"])
    non_training_eval = "".join(training_dataset[-100:]["text"])
    
    orig_ratios = {
        "Verification Text": compute_ratio(comparison_tokenizer, verification_text),
        "General Text": compute_ratio(comparison_tokenizer, general_text),
        "Code": compute_ratio(comparison_tokenizer, code_text),
        "Japanese": compute_ratio(comparison_tokenizer, japanese_text),
        "fw-train": compute_ratio(comparison_tokenizer, training_eval),
        "fw-eval": compute_ratio(comparison_tokenizer, non_training_eval)
    }


    print(f"Need first {num_rows} rows to get at least {training_characters} characters.")
    
    def data_yielder(num_rows, training_dataset, chunk_size):
        
        ### Round up to be divisible by chunk size ###
        num_rows = ((num_rows + chunk_size - 1) // chunk_size) * chunk_size

        for start_idx in range(0, num_rows, chunk_size):
            end_idx = start_idx + chunk_size
            samples = training_dataset[start_idx:end_idx]

            if len(samples) == 0:
                break

            yield samples["text"]

    sampler = data_yielder(num_rows, training_dataset, chunk_size)

    ### Train Tokenizer ###
    tokenizer.train_from_iterator(sampler, trainer)
    
    ### Set a large max length we dont really need this (we can tuncate on our own) ###
    tokenizer.model_max_length = 999999999

    ### Check Tokenizer ####
    encoded = tokenizer.encode(verification_text).ids
    decoded = tokenizer.decode(encoded)
    assert decoded == verification_text

    new_ratios = {
        "Verification Text": compute_ratio(tokenizer, verification_text),
        "General Text": compute_ratio(tokenizer, general_text),
        "Code": compute_ratio(tokenizer, code_text),
        "Japanese": compute_ratio(tokenizer, japanese_text),
        "fw-train": compute_ratio(tokenizer, training_eval),
        "fw-eval": compute_ratio(tokenizer, non_training_eval)
    }

    ### Visualize Results ###
    col_widths = [20, 15, 15, 14] 

    header = " | ".join(pad_visible(h, w) for h, w in zip(
        ["Dataset", "GPT2", "MyTokenizer", "Improvement"], col_widths
    ))
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    for key in orig_ratios.keys():
        orig_val = orig_ratios[key]    
        new_val  = new_ratios[key]     
        improvement_pct = (new_val - orig_val) / new_val * 100

        orig_base = f"{orig_val:.2f}"
        new_base  = f"{new_val:.2f}"
        imp_base  = f"{improvement_pct:6.2f}%"

        if new_val > orig_val:         
            orig_str = f"{RED}{orig_base}{RESET}"    
            new_str  = f"{GREEN}{new_base}{RESET}"
            imp_str  = f"{GREEN}{imp_base}{RESET}"   
        elif new_val < orig_val:  
            orig_str = f"{GREEN}{orig_base}{RESET}"
            new_str  = f"{RED}{new_base}{RESET}"
            imp_str  = f"{RED}{imp_base}{RESET}"
        else:                       
            orig_str = orig_base
            new_str  = new_base
            imp_str  = imp_base

        row = (
            f"{key:<20} | " +
            pad_visible(orig_str, col_widths[1]) + " | " +
            pad_visible(new_str,  col_widths[2]) + " | " +
            pad_visible(imp_str,  col_widths[3], align='>')
        )
        print(row)

    print(separator)
    
    path = os.path.join(path_to_save, "tokenizer.json")
    print(f"Saving Tokenizer To {path}")
    tokenizer.save(path)
        
if __name__ == "__main__":
    print("-"*50)
    print("Training Tokenizer!")
    print("-"*50)
    args = parse_args()
    os.makedirs(args.path_to_save_tokenizer, exist_ok=True)
    train_tokenizer(args.path_to_dataset, args.comparison_tokenizer, args.vocab_size, args.path_to_save_tokenizer)


