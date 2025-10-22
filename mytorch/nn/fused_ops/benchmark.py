"""
This will be a collection of benchmarks and numerical verifications
of all of our operations! Triton is also optimized for mixed precision
ops, so although we support float32 ops we will only benchmark float16! 
"""

import torch
import cupy as cp
import triton
import argparse
from .matmul import fused_grouped_matmul
from .softmax import fused_softmax_forward, fused_softmax_backward

##############
### MATMUL ###
##############

def test_matmul(use_dlpack=True):
    a = cp.random.normal(size=(256,256)).astype(cp.float16)
    b = cp.random.normal(size=(256,256)).astype(cp.float16)
    out_ref = cp.matmul(a,b)
    out = fused_grouped_matmul(a,b, use_dlpack=use_dlpack)
    cp.testing.assert_allclose(out, out_ref, atol=1e-2, rtol=1e-2)
    print(f"Matmul Success, DLPACK={use_dlpack}: Max Diff", cp.max(cp.abs(out-out_ref)))

configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"], 
        x_vals = [256 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["torch", "cupy", "triton_cupy", "triton_cupy_dlpack"],
        line_names = ["PyTorch", "Cupy", "Triton Cupy", "Triton Cupy DLPack"],
        styles = [("green", "-"), ("blue", "-"), ("red", "-"), ("orange", "-")],
        ylabel = "TFLOPS", 
        plot_name = "Matmul Performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark_matmul(M, N, K, provider):
    
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'cupy':
        a = cp.random.normal(size=(M, K)).astype(cp.float16)
        b = cp.random.normal(size=((K, N))).astype(cp.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cp.matmul(a, b), quantiles=quantiles)
    if provider == 'triton_cupy':
        a = cp.random.normal(size=(M, K)).astype(cp.float16)
        b = cp.random.normal(size=((K, N))).astype(cp.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_grouped_matmul(a, b, use_dlpack=False), quantiles=quantiles)
    if provider == 'triton_cupy_dlpack':
        a = cp.random.normal(size=(M, K)).astype(cp.float16)
        b = cp.random.normal(size=((K, N))).astype(cp.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fused_grouped_matmul(a, b, use_dlpack=True), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(min_ms), perf(max_ms)

def test_and_bench_matmul():
    print("="*100)
    print("Testing/Benchmarking MatMul")
    test_matmul(True)
    test_matmul(False)
    benchmark_matmul.run(show_plots=True)

###############
### SOFTMAX ###
###############

def naive_softmax(x):
   
    # Subtract max for numerical stability
    row_max = cp.max(x, axis=1, keepdims=True)  # shape (n_rows, 1)
    x_stable = x - row_max

    # Exponentiate
    x_exp = cp.exp(x_stable)

    # Sum along rows
    row_sum = cp.sum(x_exp, axis=1, keepdims=True)

    # Divide
    out = x_exp / row_sum

    return out

def test_softmax_fwd(use_dlpack=True):
    N, M = 256, 1000
    x_torch = torch.randn(N, M, device='cuda', dtype=torch.float16)
    softmax_ref = cp.array(x_torch.softmax(axis=-1).cpu().numpy())
    x_cupy = cp.array(x_torch.detach().cpu().numpy(), dtype=cp.float16)
    softmax = fused_softmax_forward(x_cupy, use_dlpack=use_dlpack)
    cp.testing.assert_allclose(softmax, softmax_ref, atol=1e-2, rtol=1e-2)
    print(f"Softmax FWD Success, DLPACK={use_dlpack}: Max Diff", cp.max(cp.abs(softmax-softmax_ref)))

def test_softmax_bwd(use_dlpack=True):
    N, M = 256, 1000

    torch_output_grad = torch.randn(N, M, device="cuda", dtype=torch.float16)
    x_torch = torch.randn(N, M, device='cuda', dtype=torch.float16, requires_grad=True)
    softmax_torch = x_torch.softmax(axis=-1)
    softmax_torch.backward(torch_output_grad, retain_graph=True)
    grad_ref = cp.array(x_torch.grad.cpu().detach().numpy())

    cp_output_grad = cp.array(torch_output_grad.detach().cpu().numpy(), dtype=cp.float16)
    x_cupy = cp.array(x_torch.detach().cpu().numpy(), dtype=cp.float16)
    softmax = fused_softmax_forward(x_cupy, use_dlpack=use_dlpack)
    grad = fused_softmax_backward(cp_output_grad, softmax, use_dlpack=use_dlpack)

    cp.testing.assert_allclose(grad, grad_ref, atol=1e-2, rtol=1e-2)
    print(f"Softmax BWD Success, DLPACK={use_dlpack}: Max Diff", cp.max(cp.abs(grad-grad_ref)))

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)], 
        line_arg='provider', 
        line_vals=['torch', 'triton_cupy', 'triton_cupy_dlpack'],
        line_names=["Torch", "Triton Cupy", "Triton Cupy DLPack"], 
        styles=[('blue', '-'), ('red', '-'), ('orange', '-')], 
        ylabel="GB/s",
        plot_name="Softmax Forward Performance",  
        args={'M': 4096}, 
    ))
def benchmark_softmax_forward(M, N, provider):
    
    x_torch = torch.randn(M, N, device="cuda", dtype=torch.float32)

    x_cupy = cp.array(x_torch.detach().cpu().numpy(), dtype=cp.float16)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x_torch, axis=-1))
    if provider == 'triton_cupy':
        ms = triton.testing.do_bench(lambda: fused_softmax_forward(x_cupy, use_dlpack=False))
    if provider == 'triton_cupy_dlpack':
        ms = triton.testing.do_bench(lambda: fused_softmax_forward(x_cupy))

    gbps = lambda ms: 2 * x_torch.numel() * x_torch.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],  # sequence length
        line_arg='provider',
        line_vals=['torch', 'triton_cupy', 'triton_cupy_dlpack'],
        line_names=["Torch", "Triton (Cupy)", "Triton (DLPack)"],
        styles=[('red', '-'), ('blue', '-'), ('orange', '-')],
        ylabel="GB/s",
        plot_name="Softmax Backward Performance",
        args={'M': 4096},
    )
)
def benchmark_softmax_backward(M, N, provider):
    
    x_torch = torch.randn(M, N, device="cuda", dtype=torch.float16, requires_grad=True)
    grad_output_torch = torch.randn(M, N, device="cuda", dtype=torch.float16)

    # Reference PyTorch softmax backward
    if provider == 'torch':
        y = x_torch.softmax(dim=-1)
        def run():
            y.backward(grad_output_torch, retain_graph=True)
        ms = triton.testing.do_bench(run)

    else:
        # Convert to cupy arrays
        x_cupy = cp.array(x_torch.detach().cpu().numpy(), dtype=cp.float16)
        grad_output_cupy = cp.array(grad_output_torch.detach().cpu().numpy(), dtype=cp.float16)
        softmax_cupy = fused_softmax_forward(x_cupy, use_dlpack=(provider == 'triton_cupy_dlpack'))

        def run():
            fused_softmax_backward(grad_output_cupy, softmax_cupy, use_dlpack=(provider == 'triton_cupy_dlpack'))
        ms = triton.testing.do_bench(run)

    num_bytes = 4 * x_torch.numel() * x_torch.element_size()

    gbps = num_bytes * 1e-9 / (ms * 1e-3)

    return gbps

def test_and_bench_softmax():
    print("="*100)
    print("Testing/Benchmarking Softmax")
    test_softmax_fwd(True)
    test_softmax_fwd(False)
    benchmark_matmul.run(show_plots=True)
    benchmark_softmax_forward.run(show_plots=True)
    benchmark_softmax_backward.run(show_plots=True)

# #####################
# ### CROSS ENTROPY ###
# #####################

BENCHMARKS = {"matmul": test_and_bench_matmul,
              "softmax": test_and_bench_softmax}

def main():

    parser = argparse.ArgumentParser("Benchmark Different Fused Operations")
    
    parser = argparse.ArgumentParser(description="Benchmark fused ops")
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["all"],
        choices=tuple(BENCHMARKS.keys()),
        help="Which ops to benchmark: matmul, softmax, cross_entropy, or 'all'"
    )

    args = parser.parse_args()

    if "all" in args.ops:
        selected = list(BENCHMARKS.keys())
    else:
        selected = [op for op in args.ops if op in BENCHMARKS]

    print(f"Running benchmarks for: {', '.join(selected)}\n")

    for op in selected:
        print(op)
        BENCHMARKS[op]()

if __name__ == "__main__":
    main()