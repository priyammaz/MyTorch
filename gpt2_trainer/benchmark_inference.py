"""
To test our KV Cache we simulate a much larger (untrained) model just
to inference and see our token throughput. 
"""
import numpy as np
import time
import argparse
import mytorch
import matplotlib.pyplot as plt
from models.gpt2 import GPT2, GPT2Config, Cache


def parse_args():
    parser = argparse.ArgumentParser(description="Quick GPT2 speed test with MyTorch")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--num_blocks", type=int, default=20)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--embed_dim", type=int, default=1280)
    parser.add_argument("--num_heads", type=int, default=10)
    parser.add_argument("--fused", action="store_true")
    return parser.parse_args()


def load_random_model(args, context_len):
    """Create a random GPT2 model for a given context length."""
    config = GPT2Config(
        vocab_size=args.vocab_size,
        max_seq_len=context_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        use_bias=False,
        use_fused_ops=True
    )
    model = GPT2(config).to(args.device)
    total_params = 0
    for _, param in model.named_parameters():
        param.astype("float16")
        total_params += np.prod(param.shape)

    print(f"Loaded GPT2 model on {args.device} | {total_params} Parameters | {args.num_blocks} blocks, "
          f"{args.embed_dim}-dim, ctx={context_len}, fused={args.fused}")
    
    return model, config


def benchmark(model, config, device, num_tokens=100, use_cache=False):
    """Generate tokens and measure speed."""
    generated = mytorch.randint(0, config.vocab_size, (1, 1), dtype=mytorch.int32, device=device)
    cache = Cache(config) if use_cache else None

    start_time = time.time()

    for _ in range(num_tokens):
        if use_cache and cache.get_seq_len != 0:
            x = generated[:, -1:]
        else:
            x = generated[:, -config.max_seq_len:]

        with mytorch.no_grad():
            out = model(x, cache=cache) if use_cache else model(x)

        logits = out[0] if use_cache else out
        if use_cache:
            cache = out[1]

        probs = mytorch.softmax(logits[:, -1, :], dim=-1)
        next_id = mytorch.multinomial(probs, num_samples=1)[0, 0]
        generated = mytorch.concatenate([generated, next_id.unsqueeze(0).unsqueeze(0)], dim=1)

    total_time = time.time() - start_time
    tps = num_tokens / total_time
    print(f"   cache={use_cache} | {tps:.2f} tok/sec ({total_time:.2f}s total)")
    return tps


def main():
    args = parse_args()

    lens = [512, 1024, 2048, 4096]
    cache_settings = [True, False]
    results = {"cache": [], "no_cache": []}

    for context_len in lens:
        for use_cache in cache_settings:
            model, config = load_random_model(args, context_len)
            tps = benchmark(model, config, args.device,
                            num_tokens=context_len, use_cache=use_cache)
            key = "cache" if use_cache else "no_cache"
            results[key].append(tps)


    plt.figure(figsize=(8, 5))
    plt.plot(lens, results["cache"], marker="o", label="With Cache")
    plt.plot(lens, results["no_cache"], marker="s", label="Without Cache")
    plt.xlabel("Context Length")
    plt.ylabel("Tokens per Second")
    plt.title("GPT2 Generation Speed vs Context Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("src/kv_cache_test.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
