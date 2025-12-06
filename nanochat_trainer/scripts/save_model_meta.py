import os
import yaml
import argparse 
from dataclasses import asdict
from nanochat_trainer.core.nanochat_gpt import GPTConfig

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_store", required=True)

    ### Model Shape ###
    parser.add_argument("--vocab_size", default=2**16, type=int)
    parser.add_argument("--context_length", default=2048, type=int)
    parser.add_argument("--num_blocks", default=20, type=int)
    parser.add_argument("--embed_dim", default=1280, type=int)
    parser.add_argument("--num_q_heads", default=10, type=int)
    parser.add_argument("--num_kv_heads", default=10, type=int)
    parser.add_argument("--mlp_ratio", default=4, type=int)

    ### Tokenizer Location ###
    parser.add_argument("--path_to_tokenizer", required=True)

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_args()

    model_config = GPTConfig(vocab_size=args.vocab_size, 
                             max_seq_len=args.context_length,
                             embed_dim=args.embed_dim, 
                             mlp_ratio=args.mlp_ratio, 
                             num_blocks=args.num_blocks,
                             num_q_heads=args.num_q_heads, 
                             num_kv_heads=args.num_kv_heads)
    
    model_config = asdict(model_config)

    model_meta = {
        "model_config": model_config, 
        "tokenizer_path": os.path.join(args.path_to_tokenizer, "tokenizer.json")
    }

    path_to_save = os.path.join(args.path_to_store, "model_meta.yaml")
    with open(path_to_save, "w") as file:
        yaml.dump(model_meta, file, default_flow_style=False, sort_keys=False)

    