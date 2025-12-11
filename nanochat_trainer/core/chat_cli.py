"""
Quick CLI to enable asking a single question to the LLM!
"""
import os
import yaml
import argparse
import mytorch
from .nanochat_gpt import GPT, GPTConfig
from .tokenizer import MyTokenizer
from .pipeline import Pipeline

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    parser.add_argument("-p", "--prompt", type=str, required=True)
    parser.add_argument("-m", "--max_token_gens", type=int, default=256)
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-k", "--topk", type=float, default=80)
    
    args = parser.parse_args()

    return args

def load_yaml(dir):

    """
    Load Yaml file created during training that tell us the 
    model shape and where the tokenizer is saved
    """

    path_to_yaml = os.path.join(dir, "model_meta.yaml")
    if not os.path.exists(path_to_yaml):
        raise Exception(f"Can't Find Model Metadata at {path_to_yaml}!!")

    with open(path_to_yaml) as f:
        meta = yaml.safe_load(f)
    
    model_config = meta["model_config"]

    return model_config, meta["tokenizer_path"]

def load_pipeline(directory, 
                  model_config, 
                  path_to_tokenizer):


    """
    Load the pipeline so we can use it to inference!
    """

    ### Load Model ###
    config = GPTConfig(**model_config)
    model = GPT(config)

    ### After model is trained the weights are in the nanochat_sft folder ###
    path_to_weights = os.path.join(directory, "nanochat_sft", "final_checkpoint", "model.safetensors")
    weights = mytorch.load(path_to_weights)
    model.load_state_dict(weights)

    ### Load Tokenizer ###
    tokenizer = MyTokenizer(path_to_tokenizer)

    ### Load Pipeline ###
    pipe = Pipeline(model, tokenizer)

    return pipe

def generate(pipe, 
             msg,
             max_token_gens=256, 
             temperature=0.6, 
             topk=50):

    """
    We will not use the generate method from the pipeline here
    but rather the internal _generate method, just so we can yield
    one token at a time and print to console as it goes rather than
    waiting till the end!
    """
    message = {
        "messages": [
            {"role": "user", "content": msg}
        ]
    }    

    print("\nUSER INPUT")
    print(msg+"\n")

    generator = pipe.generate_for_chat(
        message=message,
        max_token_gens=max_token_gens, 
        temperature=temperature, 
        topk=topk 
    )
    
    print("ASSISTANT RESPONSE:")
    for t in generator:
        if isinstance(t, dict):
            message = t
        else:
            print(t, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    
    args = parse_args()
    meta, tokenizer_path = load_yaml(args.directory)
    pipeline = load_pipeline(args.directory, meta, tokenizer_path)

    generate(
        pipeline, 
        args.prompt, 
        args.max_token_gens,
        args.temperature,
        args.topk
    )



