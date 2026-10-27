"""
Training throughput benchmark for the ~500M param nanochat model (nanochat_trainer/core/nanochat_gpt.py):
16 blocks, 1280 dim, 10 heads, 2048 context, 65536 vocab, RoPE, parameterless RMSNorm, SwiGLU MLP, no biases,
tied embeddings. MyTorch vs an equivalent PyTorch model (eager and torch.compile), bf16 mixed precision.

One optimizer step = `accum` micro-batches of forward (loss inside the model) + backward, then grad clip (1.0),
AdamW (wd 0.1 on non-embedding params, betas 0.9/0.95) and zero grad, exactly like nanochat's pretrain loop.
Data is fixed random tokens so dataloading is excluded.

Reported (rank 0, one JSON line):
  toks_per_sec       global tokens per optimizer step / median step wall time (no intermediate syncs)
  step_ms(_p90)      median (p90) optimizer step wall time
  fwd_ms / bwd_ms    median synced forward / backward time of one micro-batch
  opt_ms             synced grad clip + optimizer step + zero grad (MyTorch's unscale/cast/norm/all-reduce runs
                     inside the last micro-batch's backward, like DDP's all-reduce does for PyTorch)
  cpu_ms_per_step    host CPU time (process_time) per optimizer step, i.e. Python/dispatch/kernel-launch work
  first_step_s       wall time of the very first optimizer step (Triton JIT / torch.compile / cuBLAS warmup)
  peak_alloc_gib     peak allocated GPU memory (cupy pool + torch allocator for MyTorch's Triton kernels)
  reserved_gib       memory held by the allocators at the end (cupy pool + torch caching allocator)
  process_gib        total GPU memory of the rank 0 process as nvidia-smi sees it (includes CUDA context)

--no_dlpack sets DLPACK_DISABLE=true so the fused kernels that support it (RMSNorm, flash attention, cross entropy)
get raw cupy pointers and cupy-allocated outputs instead of zero-copy torch tensors. SwiGLU has no raw pointer
path and always goes through DLPack.

$ python -m benchmarks.bench_llm500m --framework mytorch [--no_dlpack]
$ python -m benchmarks.bench_llm500m --framework torch [--compile]
$ python -m mytorch.distributed.launch --num_gpus 2 -m benchmarks.bench_llm500m --framework mytorch
$ torchrun --nproc_per_node 2 -m benchmarks.bench_llm500m --framework torch [--compile]
$ python -m benchmarks.bench_llm500m --all [--gpus 1 2]     # every config in its own process, prints a table
"""
import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

VOCAB, CTX, DIM, BLOCKS, HEADS, MLP_RATIO = 65536, 2048, 1280, 16, 10, 4
LR, WD, BETAS, MAX_GRAD_NORM = 1.5e-4, 0.1, (0.9, 0.95), 1.0


def make_batch(batch_size, rank):
    rng = np.random.default_rng(rank)
    xy = rng.integers(0, VOCAB, (batch_size, CTX + 1))
    return xy[:, :-1], xy[:, 1:]


def process_gpu_gib():
    """GPU memory of this process as nvidia-smi reports it (includes the CUDA context)"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        for line in out.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mem) / 1024
    except Exception:
        pass
    return None


def run_loop(args, world, sync, micro_step, opt_step):
    """
    micro_step(micro_idx, synced) runs forward + backward (+ optimizer calls on the accumulation boundary for
    MyTorch) and returns (loss, t_after_forward, opt_seconds or None). opt_step() runs clip + optimizer + zero
    grad for PyTorch (a no-op for MyTorch).
    """
    step_times, cpu_times, fwd, bwd, opt = [], [], [], [], []
    first_step_s = None
    for step in range(args.warmup + args.steps + args.synced_steps):
        synced = step >= args.warmup + args.steps
        sync()
        t0, c0 = time.perf_counter(), time.process_time()
        for micro in range(args.accum):
            t_micro = time.perf_counter()
            loss, t_fwd, opt_s = micro_step(micro, synced)
            if synced:
                fwd.append(t_fwd - t_micro)
                if opt_s is None:
                    sync()
                    bwd.append(time.perf_counter() - t_fwd)
                else:
                    bwd.append(time.perf_counter() - t_fwd - opt_s)
                    opt.append(opt_s)
        t_opt = time.perf_counter()
        opt_step()
        sync()
        if synced and not opt:
            opt.append(time.perf_counter() - t_opt)
        t_end, c_end = time.perf_counter(), time.process_time()
        if step == 0:
            first_step_s = t_end - t0
        if not synced and step >= args.warmup:
            step_times.append(t_end - t0)
            cpu_times.append(c_end - c0)
    step_s = float(np.median(step_times))
    tokens = args.batch_size * CTX * args.accum * world
    return dict(step_ms=step_s * 1e3, step_ms_p90=float(np.percentile(step_times, 90) * 1e3), toks_per_sec=tokens / step_s,
                tokens_per_step=tokens, fwd_ms=float(np.median(fwd) * 1e3), bwd_ms=float(np.median(bwd) * 1e3),
                opt_ms=float(np.median(opt) * 1e3), cpu_ms_per_step=float(np.median(cpu_times) * 1e3),
                first_step_s=first_step_s, loss=loss)


###############
### MyTorch ###
###############
def run_mytorch(args):
    if args.no_dlpack:
        os.environ["DLPACK_DISABLE"] = "true"  # read once when the fused ops are imported
    import cupy as cp
    import torch  # Triton kernels allocate through torch via DLPack
    import mytorch
    from mytorch import Accelerator
    from nanochat_trainer.core.nanochat_gpt import GPT, GPTConfig

    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.accum)
    model = GPT(GPTConfig(vocab_size=VOCAB, max_seq_len=CTX, embed_dim=DIM, mlp_ratio=MLP_RATIO, num_blocks=BLOCKS,
                          num_q_heads=HEADS, num_kv_heads=HEADS, use_fused_ops=True, tie_weights=True))
    emb = [p for n, p in model.named_parameters() if "embeddings" in n]
    rest = [p for n, p in model.named_parameters() if "embeddings" not in n]
    optimizer = mytorch.optim.AdamW([{"params": emb, "weight_decay": 0.0}, {"params": rest, "weight_decay": WD}],
                                    LR, beta1=BETAS[0], beta2=BETAS[1])
    model, optimizer = acc.prepare(model, optimizer)

    xn, yn = make_batch(args.batch_size, acc.rank)
    x = mytorch.Tensor(xn, dtype=mytorch.int32).to(acc.device)
    y = mytorch.Tensor(yn, dtype=mytorch.int32).to(acc.device)
    sync = lambda: cp.cuda.Device().synchronize()

    def micro_step(micro, synced):
        loss = model(x, y)[1]  # dont keep the logits alive until the next forward (the torch twin returns only the loss)
        if synced:
            sync()
        t_fwd = time.perf_counter()
        acc.backward(loss)
        opt_s = None
        if synced and micro == args.accum - 1:
            sync()
            t_opt = time.perf_counter()
        ### nanochat calls these every micro step, the wrappers no-op until the accumulation boundary ###
        acc.clip_grad_norm_(MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        if synced and micro == args.accum - 1:
            sync()
            opt_s = time.perf_counter() - t_opt
        return loss, t_fwd, opt_s

    class PeakHook(cp.cuda.MemoryHook):
        name = "PeakHook"
        def __init__(self):
            self.pool = cp.get_default_memory_pool()
            self.peak = self.pool.used_bytes()
        def malloc_postprocess(self, **kw):
            self.peak = max(self.peak, self.pool.used_bytes())

    gc.collect()
    hook = PeakHook()
    with hook:
        r = run_loop(args, acc.world_size, sync, micro_step, opt_step=lambda: None)
    r["loss"] = float(r["loss"].item())
    r["peak_alloc_gib"] = (hook.peak + torch.cuda.max_memory_allocated()) / 2**30
    r["reserved_gib"] = (cp.get_default_memory_pool().total_bytes() + torch.cuda.memory_reserved()) / 2**30
    r["process_gib"] = process_gpu_gib()
    ### Stops cupyx's TCP store process on multi-GPU runs, otherwise it outlives rank 0 (holding GPU memory) ###
    acc.end_training()
    return r, acc.rank, acc.world_size


###############
### PyTorch ###
###############
def build_torch_model():
    import torch
    import torch.nn as tnn
    import torch.nn.functional as TF

    head_dim = DIM // HEADS

    def rms(x):
        return TF.rms_norm(x, (x.shape[-1],), eps=1e-5)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    class Attention(tnn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = tnn.Linear(DIM, DIM, bias=False)
            self.k_proj = tnn.Linear(DIM, DIM, bias=False)
            self.v_proj = tnn.Linear(DIM, DIM, bias=False)
            self.out_proj = tnn.Linear(DIM, DIM, bias=False)

        def forward(self, x, cos, sin):
            B, L, D = x.shape
            q = self.q_proj(x).view(B, L, HEADS, head_dim)
            k = self.k_proj(x).view(B, L, HEADS, head_dim)
            v = self.v_proj(x).view(B, L, HEADS, head_dim)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            o = TF.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            return self.out_proj(o.transpose(1, 2).reshape(B, L, D))

    class FeedForward(tnn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = tnn.Linear(DIM, 2 * MLP_RATIO * DIM, bias=False)
            self.out_proj = tnn.Linear(MLP_RATIO * DIM, DIM, bias=False)

        def forward(self, x):
            a, b = self.up_proj(x).chunk(2, dim=-1)
            return self.out_proj(TF.silu(a) * b)

    class Block(tnn.Module):
        def __init__(self):
            super().__init__()
            self.attention, self.feedforward = Attention(), FeedForward()

        def forward(self, x, cos, sin):
            x = x + self.attention(rms(x), cos, sin)
            return x + self.feedforward(rms(x))

    class GPT(tnn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = tnn.Embedding(VOCAB, DIM)
            self.blocks = tnn.ModuleList([Block() for _ in range(BLOCKS)])
            self.lm_head = tnn.Linear(DIM, VOCAB, bias=False)
            for n, p in self.named_parameters():
                tnn.init.normal_(p, std=0.02 / math.sqrt(2 * BLOCKS) if "out_proj" in n else 0.02)
            self.lm_head.weight = self.embeddings.weight
            inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            freqs = torch.outer(torch.arange(CTX, dtype=torch.float32), inv_freq)
            freqs = torch.cat([freqs, freqs], dim=-1)
            ### (1, L, 1, head_dim) to broadcast over (B, L, H, head_dim) ###
            self.register_buffer("cos", freqs.cos()[None, :, None, :], persistent=False)
            self.register_buffer("sin", freqs.sin()[None, :, None, :], persistent=False)

        def forward(self, idx, targets):
            x = self.embeddings(idx)
            for b in self.blocks:
                x = b(x, self.cos, self.sin)
            logits = self.lm_head(rms(x))
            return TF.cross_entropy(logits.view(-1, VOCAB).float(), targets.reshape(-1))

    return GPT()


def run_torch(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    model = build_torch_model().cuda()
    emb = [p for n, p in model.named_parameters() if "embeddings" in n]
    rest = [p for n, p in model.named_parameters() if "embeddings" not in n]
    optimizer = torch.optim.AdamW([{"params": emb, "weight_decay": 0.0}, {"params": rest, "weight_decay": WD}],
                                  lr=LR, betas=BETAS, fused=True)
    net = torch.compile(model) if args.compile else model
    if distributed:
        net = DDP(net, device_ids=[rank])

    xn, yn = make_batch(args.batch_size, rank)
    x, y = torch.as_tensor(xn, device="cuda"), torch.as_tensor(yn, device="cuda")

    def micro_step(micro, synced):
        ctx = net.no_sync() if distributed and micro < args.accum - 1 else torch.enable_grad()
        with ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = net(x, y) / args.accum
            if synced:
                torch.cuda.synchronize()
            t_fwd = time.perf_counter()
            loss.backward()
        return loss, t_fwd, None

    def opt_step():
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    r = run_loop(args, world, torch.cuda.synchronize, micro_step, opt_step)
    r["loss"] = float(r["loss"].item() * args.accum)
    r["peak_alloc_gib"] = torch.cuda.max_memory_allocated() / 2**30
    r["reserved_gib"] = torch.cuda.memory_reserved() / 2**30
    r["process_gib"] = process_gpu_gib()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return r, rank, world


###########
### CLI ###
###########
def run_all(args):
    py = [sys.executable, "-W", "ignore"]

    common = [
        "--batch_size", str(args.batch_size),
        "--accum", str(args.accum),
        "--steps", str(args.steps),
        "--warmup", str(args.warmup),
        "--synced_steps", str(args.synced_steps),
    ]

    rows = []

    for gpus in args.gpus:
        for label, fw, extra in [
            ("MyTorch", "mytorch", []),
            ("MyTorch (no DLPack)", "mytorch", ["--no_dlpack"]),
            ("PyTorch eager", "torch", []),
            ("PyTorch compile", "torch", ["--compile"]),
        ]:
            if gpus == 1:
                cmd = (
                    py
                    + ["-m", "benchmarks.bench_llm500m", "--framework", fw]
                    + extra
                    + common
                )

            elif fw == "mytorch":
                cmd = (
                    py
                    + [
                        "-m", "mytorch.distributed.launch",
                        "--num_gpus", str(gpus),
                        "--master_port", "13477",
                        "-m", "benchmarks.bench_llm500m",
                        "--framework", fw,
                    ]
                    + extra
                    + common
                )

            else:
                cmd = (
                    py
                    + [
                        "-m", "torch.distributed.run",
                        "--nproc_per_node", str(gpus),
                        "--master_port", "29577",
                        "-m", "benchmarks.bench_llm500m",
                        "--framework", fw,
                    ]
                    + extra
                    + common
                )

            print(f"running {label} on {gpus} GPU(s) ...", flush=True)

            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in range(gpus)),
                "PYTHONPATH": os.getcwd(),
            }

            try:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    start_new_session=True,
                )

            except subprocess.TimeoutExpired as e:
                stdout = e.stdout or b""
                stderr = e.stderr or b""

                if isinstance(stdout, str):
                    output = stdout + stderr
                else:
                    output = (stdout + stderr).decode(errors="ignore")

                print(
                    f"  TIMED OUT after {args.timeout}s:\n{output[-2500:]}",
                    flush=True,
                )
                continue

            lines = [
                line for line in proc.stdout.splitlines()
                if line.startswith("{")
            ]

            if not lines:
                print(
                    f"  FAILED:\n{(proc.stdout + proc.stderr)[-2500:]}",
                    flush=True,
                )
                continue

            r = json.loads(lines[-1])
            r["label"] = label
            rows.append(r)

            print("  " + json.dumps(r), flush=True)

    os.makedirs("benchmark_results", exist_ok=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    print(
        f"\n~500M nanochat model (16L/10H/1280D, ctx {CTX}, vocab {VOCAB}, "
        f"SwiGLU, RoPE), bf16, micro-batch {args.batch_size} x "
        f"accumulation {args.accum} per GPU, median of {args.steps} steps"
    )

    if not rows:
        print("No successful benchmark runs.")
        return

    df = pd.DataFrame(rows)

    print("\n#### Speed")

    speed = df[
        [
            "world",
            "label",
            "tokens_per_step",
            "toks_per_sec",
            "step_ms",
            "step_ms_p90",
            "fwd_ms",
            "bwd_ms",
            "opt_ms",
            "cpu_ms_per_step",
            "first_step_s",
        ]
    ].copy()

    speed = speed.rename(
        columns={
            "world": "GPUs",
            "label": "implementation",
            "tokens_per_step": "tokens/step",
            "toks_per_sec": "tokens/s",
            "step_ms": "step (s)",
            "step_ms_p90": "step p90 (s)",
            "fwd_ms": "fwd / micro (ms)",
            "bwd_ms": "bwd / micro (ms)",
            "opt_ms": "optimizer (ms)",
            "cpu_ms_per_step": "host CPU / step (ms)",
            "first_step_s": "first step (s)",
        }
    )

    speed["step (s)"] /= 1e3
    speed["step p90 (s)"] /= 1e3

    speed["tokens/step"] = speed["tokens/step"].map(lambda x: f"{x:,}")
    speed["tokens/s"] = speed["tokens/s"].map(lambda x: f"{x:,.0f}")

    for col in ["step (s)", "step p90 (s)"]:
        speed[col] = speed[col].map(lambda x: f"{x:.2f}")

    for col in [
        "fwd / micro (ms)",
        "bwd / micro (ms)",
        "optimizer (ms)",
        "host CPU / step (ms)",
    ]:
        speed[col] = speed[col].map(lambda x: f"{x:,.0f}")

    speed["first step (s)"] = speed["first step (s)"].map(
        lambda x: f"{x:.1f}"
    )

    print(speed.to_markdown(index=False))
    
    print("\n#### Memory (rank 0)")

    memory = df[
        [
            "world",
            "label",
            "peak_alloc_gib",
            "reserved_gib",
            "process_gib",
            "loss",
        ]
    ].copy()

    memory = memory.rename(
        columns={
            "world": "GPUs",
            "label": "implementation",
            "peak_alloc_gib": "peak allocated (GiB)",
            "reserved_gib": "allocator reserved (GiB)",
            "process_gib": "process total, nvidia-smi (GiB)",
            "loss": "loss after run",
        }
    )

    for col in [
        "peak allocated (GiB)",
        "allocator reserved (GiB)",
        "process total, nvidia-smi (GiB)",
    ]:
        memory[col] = memory[col].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "n/a"
        )

    memory["loss after run"] = memory["loss after run"].map(
        lambda x: f"{x:.3f}"
    )

    print(memory.to_markdown(index=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["mytorch", "torch"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no_dlpack", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--synced_steps", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per config in --all")
    ap.add_argument("--out", default="benchmark_results/bench_llm500m.json")
    args = ap.parse_args()

    if args.all:
        return run_all(args)

    r, rank, world = (run_mytorch if args.framework == "mytorch" else run_torch)(args)
    if rank == 0:
        r.update(framework=args.framework, compile=args.compile, no_dlpack=args.no_dlpack, world=world,
                 batch_size=args.batch_size, accum=args.accum)
        print(json.dumps(r), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
