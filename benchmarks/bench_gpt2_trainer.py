"""
Full training-step benchmark mirroring gpt2_trainer/train_gpt2.py (GPT2 base: 12 layers, 12 heads,
768 dim, 1024 context, 50257 vocab) for MyTorch vs an equivalent idiomatic PyTorch loop.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import numpy as np
import pandas as pd

### Normal GPT2 Shape ###
VOCAB, CTX, DIM, HEADS, BLOCKS = 50257, 1024, 768, 12, 12
MAX_LR, WD, BETAS, MAX_GRAD_NORM = 6e-4, 0.1, (0.9, 0.95), 1.0

def process_gpu_mib():
    """Total GPU memory held by this process as nvidia-smi sees it (includes CUDA context)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mem)
    except Exception:
        pass
    return None

def make_batch(batch_size):
    rng = np.random.default_rng(0)
    x = rng.integers(0, VOCAB, (batch_size, CTX + 1))
    return x[:, :-1], x[:, 1:]

def time_steps(step, sync, warmup, iters):
    for _ in range(warmup):
        step()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        sync()
        times.append(time.perf_counter() - t0)
    return times

def run_mytorch(args):
    import cupy as cp
    import torch
    import mytorch
    import mytorch.nn as nn
    import mytorch.optim as optim
    from mytorch import Accelerator
    from gpt2_trainer.gpt2 import GPT2, GPT2Config

    accelerator = Accelerator(mixed_precision={"fp32": False, "fp16": "fp16", "bf16": "bf16"}[args.precision])
    cfg = GPT2Config(vocab_size=VOCAB, max_seq_len=CTX, embed_dim=DIM, num_heads=HEADS, num_blocks=BLOCKS,
                     mlp_ratio=4, use_bias=False, use_fused_ops=True)
    model = GPT2(cfg)

    ### Same param groups / optimizer as the trainer ###
    decay = [p for _, p in model.named_parameters() if len(p.shape) >= 2]
    no_decay = [p for _, p in model.named_parameters() if len(p.shape) < 2]
    optimizer = optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=MAX_LR, beta1=BETAS[0], beta2=BETAS[1])
    model, optimizer = accelerator.prepare(model, optimizer)
    loss_fn = nn.CrossEntropyLoss(fused=True)

    xn, yn = make_batch(args.batch_size)
    x = mytorch.Tensor(xn, dtype=mytorch.int32).to(accelerator.device)
    y = mytorch.Tensor(yn, dtype=mytorch.int32).to(accelerator.device)

    def step():
        logits = model(x)
        loss = loss_fn(logits, y)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        return loss

    sync = lambda: cp.cuda.Device().synchronize()
    times = time_steps(step, sync, args.warmup, args.iters)

    ### Peak memory over one more step: cupy pool (hook) + torch allocator (fused kernels) ###
    class PeakHook(cp.cuda.MemoryHook):
        name = "PeakHook"
        def __init__(self):
            self.pool = cp.get_default_memory_pool()
            self.peak = self.pool.used_bytes()
        def malloc_postprocess(self, **kw):
            self.peak = max(self.peak, self.pool.used_bytes())

    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    hook = PeakHook()
    with hook:
        loss = step()
        sync()
    pool = cp.get_default_memory_pool()
    return dict(
        times=times,
        final_loss=float(loss.item()),
        peak_alloc_gib=(hook.peak + torch.cuda.max_memory_allocated()) / 2**30,
        reserved_gib=(pool.total_bytes() + torch.cuda.memory_reserved()) / 2**30,
        detail=f"cupy peak {hook.peak / 2**30:.2f} + torch peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
    )

def run_torch(args):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as TF

    class Block(tnn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = tnn.LayerNorm(DIM, bias=False)
            self.qkv = tnn.Linear(DIM, 3 * DIM, bias=False)
            self.out = tnn.Linear(DIM, DIM, bias=False)
            self.ln2 = tnn.LayerNorm(DIM, bias=False)
            self.fc1 = tnn.Linear(DIM, 4 * DIM, bias=False)
            self.fc2 = tnn.Linear(4 * DIM, DIM, bias=False)

        def forward(self, x):
            B, L, D = x.shape
            q, k, v = self.qkv(self.ln1(x)).reshape(B, L, HEADS, 3 * D // HEADS).transpose(1, 2).chunk(3, dim=-1)
            o = TF.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.out(o.transpose(1, 2).reshape(B, L, D))
            return x + self.fc2(TF.gelu(self.fc1(self.ln2(x))))

    class GPT2(tnn.Module):
        def __init__(self):
            super().__init__()
            self.tok = tnn.Embedding(VOCAB, DIM)
            self.pos = tnn.Embedding(CTX, DIM)
            self.blocks = tnn.ModuleList([Block() for _ in range(BLOCKS)])
            self.ln_f = tnn.LayerNorm(DIM, bias=False)
            self.head = tnn.Linear(DIM, VOCAB, bias=False)
            for m in self.modules():
                if isinstance(m, (tnn.Linear, tnn.Embedding)):
                    tnn.init.normal_(m.weight, std=0.02)
            self.head.weight = self.tok.weight

        def forward(self, idx):
            x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
            for b in self.blocks:
                x = b(x)
            return self.head(self.ln_f(x))

    model = GPT2().cuda()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=MAX_LR, betas=BETAS)
    fwd = torch.compile(model) if args.compile else model

    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")

    xn, yn = make_batch(args.batch_size)
    x, y = torch.as_tensor(xn, device="cuda"), torch.as_tensor(yn, device="cuda")

    def step():
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            logits = fwd(x)
            loss = TF.cross_entropy(logits.view(-1, VOCAB), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return loss

    times = time_steps(step, torch.cuda.synchronize, args.warmup, args.iters)
    torch.cuda.reset_peak_memory_stats()
    loss = step()
    torch.cuda.synchronize()
    return dict(
        times=times,
        final_loss=float(loss.item()),
        peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
        reserved_gib=torch.cuda.memory_reserved() / 2**30,
        detail="",
    )


CONFIGS = [
    # (label, framework, precision, extra env, extra args)
    ("PyTorch eager", "torch", "fp32", {}, []),
    ("MyTorch (Triton linear)", "mytorch", "fp32", {"DISABLE_FUSED_LINEAR": "false"}, []),
    ("MyTorch (cuBLAS linear)", "mytorch", "fp32", {}, []),
    ("PyTorch eager", "torch", "fp16", {}, []),
    ("MyTorch (Triton linear)", "mytorch", "fp16", {"DISABLE_FUSED_LINEAR": "false"}, []),
    ("MyTorch (cuBLAS linear)", "mytorch", "fp16", {}, []),
    ("PyTorch eager", "torch", "bf16", {}, []),
    ("MyTorch (Triton linear)", "mytorch", "bf16", {"DISABLE_FUSED_LINEAR": "false"}, []),
    ("MyTorch (cuBLAS linear)", "mytorch", "bf16", {}, []),
    ("PyTorch torch.compile", "torch", "bf16", {}, ["--compile"]),
]

def run_all(args):
    rows = []

    for label, fw, prec, env, extra in CONFIGS:
        cmd = [
            sys.executable, "-W", "ignore",
            "-m", "benchmarks.bench_gpt2_trainer",
            "--framework", fw,
            "--precision", prec,
            "--batch_size", str(args.batch_size),
            "--iters", str(args.iters),
            "--warmup", str(args.warmup),
        ] + extra

        print(f"running {label} {prec} ...", flush=True)

        proc = subprocess.run(
            cmd,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )

        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]

        if not lines:
            print(f"  FAILED:\n{proc.stderr[-1500:]}")
            continue

        r = json.loads(lines[-1])
        r["label"] = label
        rows.append(r)

        print(
            f"  {json.dumps({k: v for k, v in r.items() if k != 'label'})}",
            flush=True,
        )

    tokens = args.batch_size * CTX

    print(
        f"\nGPT2 base (12L/12H/768D, ctx {CTX}, vocab {VOCAB}), "
        f"micro-batch {args.batch_size}, full train step, "
        f"median of {args.iters} steps"
    )

    if not rows:
        print("No successful benchmark runs.")
        return

    df = pd.DataFrame(rows)

    df["tokens/s"] = tokens / df["step_ms"] * 1e3

    df["process GPU mem (GiB)"] = df["process_gib"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"
    )

    df["step (ms)"] = df["step_ms"].map(lambda x: f"{x:.0f}")
    df["tokens/s"] = df["tokens/s"].map(lambda x: f"{x:,.0f}")
    df["peak allocated (GiB)"] = df["peak_alloc_gib"].map(lambda x: f"{x:.2f}")
    df["reserved (GiB)"] = df["reserved_gib"].map(lambda x: f"{x:.2f}")

    df = df[
        [
            "precision",
            "label",
            "step (ms)",
            "tokens/s",
            "peak allocated (GiB)",
            "reserved (GiB)",
            "process GPU mem (GiB)",
        ]
    ]

    df = df.rename(columns={"label": "implementation"})

    print(df.to_markdown(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["mytorch", "torch"])
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        return run_all(args)

    r = (run_mytorch if args.framework == "mytorch" else run_torch)(args)
    times = r.pop("times")
    r.update(framework=args.framework, precision=args.precision, compile=args.compile,
             linear="triton" if os.environ.get("DISABLE_FUSED_LINEAR", "true").lower() == "false" else "cuBLAS",
             step_ms=float(np.median(times) * 1e3), step_ms_p90=float(np.percentile(times, 90) * 1e3),
             process_gib=(lambda m: m / 1024 if m is not None else None)(process_gpu_mib()))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
