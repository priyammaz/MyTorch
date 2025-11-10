"""
This will be a collection of benchmarks and numerical verifications
of all of our operations! Triton is also optimized for mixed precision
ops, so although we support float32 ops we will only benchmark float16! 
"""

import torch
import cupy as cp
import triton
import argparse
import pytest
from pathlib import Path
from .matmul import fused_grouped_matmul, blocked_matmul
from .softmax import fused_softmax_forward, fused_softmax_backward
from .conv import fused_conv2d_forward, fused_conv2d_backward
from .layernorm import fused_layernorm_forward, fused_layernorm_backward
from .flash_attention import fused_sdpa_forward, fused_sdpa_backward


##############
### MATMUL ###
##############
@pytest.mark.parametrize("M, N, K", [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (256, 512, 128),
    (512, 256, 128),
])
@pytest.mark.parametrize("use_dlpack", [True, False])
def test_matmul(M, N, K, use_dlpack):
    # Create random inputs
    a = cp.random.normal(size=(M, K)).astype(cp.float16)
    b = cp.random.normal(size=(K, N)).astype(cp.float16)
    
    # Reference (CuPy’s built-in matmul)
    out_ref = cp.matmul(a, b)
    
    # Your fused kernel
    out = fused_grouped_matmul(a, b, use_dlpack=use_dlpack)
    
    # Validate results
    cp.testing.assert_allclose(out, out_ref, atol=1e-2, rtol=1e-2)
    
    # Log useful info
    max_diff = float(cp.max(cp.abs(out - out_ref)))
    print(f"Matmul Success: M={M}, N={N}, K={K}, DLPACK={use_dlpack} | Max Diff={max_diff:.4f}")

configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"], 
        x_vals = [512 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["torch", "torch_blocked", "cupy", "triton_cupy", "triton_cupy_dlpack"],
        line_names = ["PyTorch", "Matmul No Groups", "Cupy", "Triton Cupy Grouped", "Triton Cupy DLPack Grouped"],
        styles = [("blue", "-"), ("purple", "-"), ("green", "-"), ("red", "-"), ("orange", "-")],
        ylabel = "TFLOPS", 
        plot_name = "matmul",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark_matmul(M, N, K, provider):
    
    if provider == 'torch':
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'torch_blocked':
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: blocked_matmul(a,b))
    if provider == 'cupy':
        a = cp.random.normal(size=(M, K)).astype(cp.float16)
        b = cp.random.normal(size=((K, N))).astype(cp.float16)
        ms = triton.testing.do_bench(lambda: cp.matmul(a, b))
    if provider == 'triton_cupy':
        a = cp.random.normal(size=(M, K)).astype(cp.float16)
        b = cp.random.normal(size=((K, N))).astype(cp.float16)
        ms = triton.testing.do_bench(lambda: fused_grouped_matmul(a, b, use_dlpack=False))
    if provider == 'triton_cupy_dlpack':
        a = cp.random.normal(size=(M, K)).astype(cp.float16)
        b = cp.random.normal(size=((K, N))).astype(cp.float16)
        ms = triton.testing.do_bench(lambda: fused_grouped_matmul(a, b, use_dlpack=True))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms)

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

@pytest.mark.parametrize("N, M", [
    (32, 64),
    (256, 512),
    (2, 11),
])
@pytest.mark.parametrize("use_dlpack", [True, False])
def test_softmax(N, M, use_dlpack):
    # Setup inputs
    x_torch = torch.randn(N, M, device='cuda', dtype=torch.float16, requires_grad=True)
    torch_output_grad = torch.randn(N, M, device='cuda', dtype=torch.float16)

    # PyTorch forward and backward reference
    softmax_ref = cp.array(x_torch.softmax(dim=-1).cpu().detach().numpy())
    softmax_torch = x_torch.softmax(dim=-1)
    softmax_torch.backward(torch_output_grad, retain_graph=True)
    grad_ref = cp.array(x_torch.grad.cpu().detach().numpy())

    # CuPy inputs for Triton implementation
    x_cupy = cp.array(x_torch.detach().cpu().numpy(), dtype=cp.float16)
    cp_output_grad = cp.array(torch_output_grad.detach().cpu().numpy(), dtype=cp.float16)

    # Triton forward pass
    softmax = fused_softmax_forward(x_cupy, use_dlpack=use_dlpack)
    cp.testing.assert_allclose(softmax, softmax_ref, atol=1e-2, rtol=1e-2)

    # Triton backward pass
    grad = fused_softmax_backward(cp_output_grad, softmax, use_dlpack=use_dlpack)
    cp.testing.assert_allclose(grad, grad_ref, atol=1e-2, rtol=1e-2)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)], 
        line_arg='provider', 
        line_vals=['torch', 'triton_cupy', 'triton_cupy_dlpack', "naive"],
        line_names=["Torch", "Triton Cupy", "Triton Cupy DLPack", "Naive"], 
        styles=[('blue', '-'), ('red', '-'), ('orange', '-'), ('purple', "-")], 
        ylabel="GB/s",
        plot_name="softmax_fwd",  
        args={'M': 4096}, 
    ))
def benchmark_softmax_forward(M, N, provider):
    
    x_torch = torch.randn(M, N, device="cuda", dtype=torch.float16)

    x_cupy = cp.array(x_torch.detach().cpu().numpy(), dtype=cp.float16)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x_torch, axis=-1))
    if provider == 'triton_cupy':
        ms = triton.testing.do_bench(lambda: fused_softmax_forward(x_cupy, use_dlpack=False))
    if provider == 'triton_cupy_dlpack':
        ms = triton.testing.do_bench(lambda: fused_softmax_forward(x_cupy))
    if provider == "naive":
        ms = triton.testing.do_bench(lambda: naive_softmax(x_cupy))

    gbps = lambda ms: 2 * x_torch.numel() * x_torch.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],  # sequence length
        line_arg='provider',
        line_vals=['torch', 'triton_cupy', 'triton_cupy_dlpack'],
        line_names=["Torch", "Triton (Cupy)", "Triton (DLPack)"],
        styles=[('red', '-'), ('orange', '-'), ('blue', '-')],
        ylabel="GB/s",
        plot_name="softmax_bwd",
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

    gbps = lambda ms: 4 * x_torch.numel() * x_torch.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)

##############
### CONV2D ###
##############

def naive_conv2d(input, weight, bias=None, stride=1, padding=0):
    
    B, C_in, H, W = input.shape
    C_out, _, K, _ = weight.shape
    S,P = stride[0], padding[0]

    H_out = (H + 2*P - K)//S + 1
    W_out = (H + 2*P - K)//S + 1

    if P > 0:
        x_padded = cp.pad(input, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
    else:
        x_padded = input

    shape = (B, C_in, K, K, H_out, W_out)
    strides = (
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2], 
        x_padded.strides[3],
        S*x_padded.strides[2],
        S*x_padded.strides[3]
    )

    cols = cp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

    cols = cols.reshape(B, C_in*K*K, H_out*W_out).transpose(0,2,1)
    cols_flat = cols.reshape(B*H_out*W_out, -1)

    weights_flat = weight.reshape(C_out, -1).T

    output = cp.empty((cols_flat.shape[0], weights_flat.shape[1]))
    cp.matmul(cols_flat, weights_flat, out=output)
    if bias is not None:
        cp.add(output, bias, out=output)

    output = output.reshape(B, H_out*W_out, C_out).transpose(0,2,1).reshape(B, C_out, H_out, W_out)
    return output


@pytest.mark.parametrize("use_dlpack", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("N, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding, dilation", [
    (1, 1, 3, 3, 1, 3, 3, 1, 0, 1),
    (1, 2, 3, 3, 2, 2, 2, 1, 0, 1),
    (1, 128, 16, 16, 32, 1, 1, 1, 0, 1),
    (1, 8, 64, 64, 16, 3, 3, 2, 1, 1),
    (2, 8, 55, 55, 16, 3, 5, 1, 1, 1),
])
def test_conv2d(N, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding, dilation, use_bias, use_dlpack):

    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)

    ### Create Torch Tensors as Reference ###
    factory_kwargs = {'device': 'cuda', 'dtype': torch.float16}
    input = torch.randn(N, C_in, H_in, W_in, requires_grad=True, **factory_kwargs)
    weight = torch.randn(C_out, C_in, K_h, K_w, requires_grad=True, **factory_kwargs)

    if use_bias:
        bias = torch.randn(C_out, requires_grad=True, **factory_kwargs)
    else:
        bias = None

    ### Create Cupy Copies ###
    input_cp = cp.array(input.detach().cpu().numpy())
    weight_cp = cp.array(weight.detach().cpu().numpy())
    if bias is not None:
        bias_cp = cp.array(bias.detach().cpu().numpy())
    else:
        bias_cp = None

    ### Do Conv2d in Torch ###
    output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation)
    doutput = torch.randn_like(output)
    output.backward(doutput)

    ### Store doutput in cupy ###
    doutput_cp = cp.array(doutput.detach().cpu().numpy())

    ### Store output as ref ###
    output_ref = cp.array(output.detach().cpu().numpy())

    ### Store Grads as Ref ###
    dinput, input.grad = input.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None
    
    dinput_ref = cp.array(dinput.detach().cpu().numpy())
    dweight_ref = cp.array(dweight.detach().cpu().numpy())

    if use_bias:
        dbias, bias.grad = bias.grad.clone(), None
        dbias_ref = cp.array(dbias.detach().cpu().numpy())

    ### Test triton kernel on Cupy ###
    triton_output = fused_conv2d_forward(input_cp, weight_cp, bias_cp, stride, padding, dilation, use_dlpack)

    d = fused_conv2d_backward(doutput_cp, input_cp, weight_cp, bias_cp, H_in, W_in, K_h, K_w, stride, padding, dilation, use_dlpack)
    if use_bias:
        triton_dinput, triton_dweight, triton_dbias = d
    else:
        triton_dinput, triton_dweight = d

    cp.testing.assert_allclose(output_ref, triton_output, atol=1e-1, rtol=1e-1)
    cp.testing.assert_allclose(dinput_ref, triton_dinput, atol=1e-1, rtol=1e-1)
    cp.testing.assert_allclose(dweight_ref, triton_dweight, atol=1e-1, rtol=1e-1)

    if use_bias:
        cp.testing.assert_allclose(dbias_ref, triton_dbias, atol=1e-1, rtol=1e-1)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H_in"],  # input height (will also use for width)
        x_vals=[8, 16, 32, 64, 128, 256, 512],
        line_arg="provider",
        line_vals=["torch", "triton_cupy", "triton_cupy_dlpack", "naive"],
        line_names=["Torch", "Triton Cupy", "Triton Cupy DLPack", "Naive"],
        styles=[("blue", "-"), ("red", "-"), ("orange", "-"), ('purple', "-")],
        ylabel="TFLOPs/s",
        plot_name="conv2d_fwd",
        args={"N": 1, "C_in": 64, "C_out": 128, "K_h": 3, "K_w": 3, "stride": 2, "padding": 1, "dilation": 1},
    )
)
def benchmark_conv2d_forward(N, C_in, C_out, H_in, K_h, K_w, stride, padding, dilation, provider):
    device = "cuda"
    dtype = torch.float16

    # Input + weight
    x_torch = torch.randn((N, C_in, H_in, H_in), device=device, dtype=dtype)
    w_torch = torch.randn((C_out, C_in, K_h, K_w), device=device, dtype=dtype)
    bias_torch = torch.randn((C_out,), device=device, dtype=dtype)

    # Convert to cupy arrays
    x_cp = cp.array(x_torch.detach().cpu().numpy(), dtype=cp.float16)
    w_cp = cp.array(w_torch.detach().cpu().numpy(), dtype=cp.float16)
    b_cp = cp.array(bias_torch.detach().cpu().numpy(), dtype=cp.float16)

    # Define stride/padding/dilation tuples
    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)

    # Select provider
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.nn.functional.conv2d(x_torch, w_torch, bias_torch, stride, padding, dilation))
    elif provider == "triton_cupy":
        ms = triton.testing.do_bench(lambda: fused_conv2d_forward(x_cp, w_cp, b_cp, stride, padding, dilation, use_dlpack=False))
    elif provider == "triton_cupy_dlpack":
        ms = triton.testing.do_bench(lambda: fused_conv2d_forward(x_cp, w_cp, b_cp, stride, padding, dilation, use_dlpack=True))
    elif provider == "naive":
        ms = triton.testing.do_bench(lambda: naive_conv2d(x_cp, w_cp, b_cp, stride, padding))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Compute total FLOPs:
    # For Conv2D: 2 * N * C_out * H_out * W_out * (C_in * K_h * K_w)
    H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) // stride[0] + 1
    W_out = (H_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) // stride[1] + 1
    flops = 2 * N * C_out * H_out * W_out * (C_in * K_h * K_w)

    tflops = flops * 1e-12 / (ms * 1e-3)

    return tflops

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H_in"],  # input height (we'll also use for width)
        x_vals=[8, 16, 32, 64, 128, 256, 512],
        line_arg="provider",
        line_vals=["torch", "triton_cupy", "triton_cupy_dlpack"],
        line_names=["Torch", "Triton Cupy", "Triton Cupy DLPack"],
        styles=[("blue", "-"), ("red", "-"), ("orange", "-")],
        ylabel="TFLOPs/s",
        plot_name="conv2d_bwd",
        args={"N": 16, "C_in": 64, "C_out": 128, "K_h": 3, "K_w": 3, "stride": 1, "padding": 1, "dilation": 1, "use_bias": True},
    )
)
def benchmark_conv2d_backward(N, C_in, C_out, H_in, K_h, K_w, stride, padding, dilation, use_bias, provider):
    device = "cuda"
    dtype = torch.float16

    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)

    x_torch = torch.randn((N, C_in, H_in, H_in), device=device, dtype=dtype, requires_grad=True)
    w_torch = torch.randn((C_out, C_in, K_h, K_w), device=device, dtype=dtype, requires_grad=True)
    b_torch = torch.randn((C_out,), device=device, dtype=dtype, requires_grad=True) if use_bias else None

    out_torch = torch.nn.functional.conv2d(x_torch, w_torch, b_torch, stride, padding, dilation)
    doutput_torch = torch.randn_like(out_torch)

    x_cp = cp.array(x_torch.detach().cpu().numpy(), dtype=cp.float16)
    w_cp = cp.array(w_torch.detach().cpu().numpy(), dtype=cp.float16)
    b_cp = cp.array(b_torch.detach().cpu().numpy(), dtype=cp.float16) if use_bias else None
    doutput_cp = cp.array(doutput_torch.detach().cpu().numpy(), dtype=cp.float16)

    if provider == "torch":
        out = torch.nn.functional.conv2d(x_torch, w_torch, b_torch, stride, padding, dilation)
        def torch_backward():
            out.backward(doutput_torch, retain_graph=True)

        ms = triton.testing.do_bench(torch_backward)

    elif provider == "triton_cupy":
        def triton_backward():
            fused_conv2d_backward(
                doutput_cp,
                x_cp,
                w_cp,
                b_cp,
                H_in,
                H_in,
                K_h,
                K_w,
                stride,
                padding,
                dilation,
                use_dlpack=False,
            )

        ms = triton.testing.do_bench(triton_backward)

    elif provider == "triton_cupy_dlpack":
        def triton_dlpack_backward():
            fused_conv2d_backward(
                doutput_cp,
                x_cp,
                w_cp,
                b_cp,
                H_in,
                H_in,
                K_h,
                K_w,
                stride,
                padding,
                dilation,
                use_dlpack=True,
            )

        ms = triton.testing.do_bench(triton_dlpack_backward)

    H_out = (H_in + 2 * padding[0] - dilation[0] * (K_h - 1) - 1) // stride[0] + 1
    W_out = (H_in + 2 * padding[1] - dilation[1] * (K_w - 1) - 1) // stride[1] + 1
    flops_forward = 2 * N * C_out * H_out * W_out * (C_in * K_h * K_w)
    flops_backward = 2 * flops_forward 

    tflops = flops_backward * 1e-12 / (ms * 1e-3)

    return tflops

#################
### LayerNorm ###
#################


@pytest.mark.parametrize("M, N", [
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (256, 512),
    (512, 256),
])
@pytest.mark.parametrize("use_dlpack", [True, False])
def test_layernorm(M, N, use_dlpack):
    dtype = torch.float16
    eps = 1e-5
    x_torch = torch.randn((M, N), dtype=dtype, device="cuda", requires_grad=True)
    weight_torch = torch.randn((N,), dtype=dtype, device="cuda", requires_grad=True)
    bias_torch = torch.randn((N,), dtype=dtype, device="cuda", requires_grad=True)
    dy = torch.randn_like(x_torch)

    x_cp = cp.asarray(x_torch.cpu().detach().numpy())
    weight_cp = cp.asarray(weight_torch.cpu().detach().numpy())
    bias_cp = cp.asarray(bias_torch.cpu().detach().numpy())
    dy_cp = cp.asarray(dy.cpu().detach().numpy())

    y_triton = fused_layernorm_forward(x_cp, weight_cp, bias_cp, eps, use_dlpack=use_dlpack)
    
    if isinstance(y_triton, tuple):
        y_triton, x_hat, inv_var = y_triton
    
    y_torch = torch.nn.functional.layer_norm(x_torch, (N,), weight_torch, bias_torch, eps)

    cp.testing.assert_allclose(cp.asarray(y_torch.detach().cpu().numpy()), y_triton, rtol=1e-1, atol=1e-1)
    print(f"Forward pass OK: M={M}, N={N}, DLPACK={use_dlpack}")

    grads_cp = fused_layernorm_backward(x_hat, inv_var, dy_cp, weight_cp, eps, use_dlpack=use_dlpack)
    dx_triton, dweight_triton, dbias_triton = grads_cp
 
    y_torch.backward(dy, retain_graph=True)
    dx_torch = x_torch.grad
    dweight_torch = weight_torch.grad
    dbias_torch = bias_torch.grad

    cp.testing.assert_allclose(cp.asarray(dx_torch.detach().cpu().numpy()), dx_triton, rtol=1e-1, atol=1e-1)
    cp.testing.assert_allclose(cp.asarray(dweight_torch.detach().cpu().numpy()), dweight_triton, rtol=1e-1, atol=1e-1)
    cp.testing.assert_allclose(cp.asarray(dbias_torch.detach().cpu().numpy()), dbias_triton, rtol=1e-1, atol=1e-1)
    
    print(f"Backward pass OK: M={M}, N={N}")
          

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 33)],
        line_arg='provider',
        line_vals=['triton_cupy', 'triton_cupy_dlpack', 'torch'],
        line_names=['Triton Cupy', 'Triton Cupy DLPack', 'Torch'],
        styles=[("red", "-"), ("orange", "-"), ('blue', '-')],
        ylabel='GB/s',
        plot_name='layernorm_fwd',
        args={'M': 4096, 'dtype': torch.float16},
    )
)
def benchmark_layernorm_forward(M, N, dtype, provider, eps=1e-5, device="cuda"):
    # Create data
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device, requires_grad=True)
    weight = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    dy = 0.1 * torch.randn_like(x)
    quantiles = [0.5, 0.2, 0.8]

    x_cp = cp.asarray(x.cpu().detach().numpy())
    weight_cp = cp.asarray(weight.cpu().detach().numpy())
    bias_cp = cp.asarray(bias.cpu().detach().numpy())

    def y_fwd():
        if provider == "triton_cupy":
            return fused_layernorm_forward(x_cp, weight_cp, bias_cp, eps, use_dlpack=False)
        if provider == "triton_cupy_dlpack":
            return fused_layernorm_forward(x_cp, weight_cp, bias_cp, eps, use_dlpack=True)
        elif provider == "torch":
            return torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    
    return gbps(ms), gbps(min_ms), gbps(max_ms)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 33)],
        line_arg='provider',
        line_vals=['triton_cupy', 'triton_cupy_dlpack', 'torch'],
        line_names=['Triton Cupy', 'Triton Cupy DLPack', 'Torch'],
        styles=[("red", "-"), ("orange", "-"), ('blue', '-')],
        ylabel='GB/s',
        plot_name='layernorm_bwd',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'},
    )
)
def benchmark_layernorm_backward(M, N, dtype, provider, mode='backward', eps=1e-5, device="cuda"):
    # Create data
    x = -2.3 + 0.5 * torch.randn((M, N), dtype=dtype, device=device, requires_grad=True)
    weight = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand((N,), dtype=dtype, device=device, requires_grad=True)
    dy = 0.1 * torch.randn_like(x)
    quantiles = [0.5, 0.2, 0.8]

    x_cp = cp.asarray(x.cpu().detach().numpy())
    weight_cp = cp.asarray(weight.cpu().detach().numpy())
    bias_cp = cp.asarray(bias.cpu().detach().numpy())
    dy_cp = cp.asarray(dy.cpu().detach().numpy())

    def y_fwd():
        if provider == "triton_cupy":
            return fused_layernorm_forward(x_cp, weight_cp, bias_cp, eps, use_dlpack=False)
        if provider == "triton_cupy_dlpack":
            return fused_layernorm_forward(x_cp, weight_cp, bias_cp, eps, use_dlpack=True)
        elif provider == "torch":
            return torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Forward benchmark
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    # Backward benchmark
    elif mode == 'backward':
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

        if provider == "torch":
            y = y_fwd()
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: y.backward(dy, retain_graph=True),
                quantiles=quantiles,
                grad_to_none=[x],
                rep=500
            )

        elif provider == "triton_cupy":
            y, x_hat, inv_var = y_fwd()
            y = cp.asarray(x.cpu().detach().numpy())
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_layernorm_backward(x_hat, inv_var, dy_cp, weight_cp, eps, use_dlpack=False),
                quantiles=quantiles,
                grad_to_none=[x],
                rep=500
            )

        elif provider == "triton_cupy_dlpack":
            y, x_hat, inv_var = y_fwd()
            y = cp.asarray(x.cpu().detach().numpy())
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_layernorm_backward(x_hat, inv_var, dy_cp, weight_cp, eps, use_dlpack=True),
                quantiles=quantiles,
                grad_to_none=[x],
                rep=500
            )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Return mean, min, max GB/s
    return gbps(ms), gbps(min_ms), gbps(max_ms)

#######################
### FLASH ATTENTION ###
#######################


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("L", [1, 4, 32, 64, 100, 128])
def test_sdpa(causal, L):
    device = "cuda"
    dtype = torch.float16
    B, H, Dh = 1, 16, 64

    # ----------------------
    # Reference PyTorch
    # ----------------------
    Q = torch.randn(B, H, L, Dh, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, L, Dh, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, L, Dh, device=device, dtype=dtype, requires_grad=True)
    dO = torch.randn(B, H, L, Dh, device=device, dtype=dtype, requires_grad=False)

    out_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    out_ref.backward(dO)

    dQ_ref = cp.asarray(Q.grad.detach().clone())
    dK_ref = cp.asarray(K.grad.detach().clone())
    dV_ref = cp.asarray(V.grad.detach().clone())

    Q.grad = None
    K.grad = None
    V.grad = None

    # ----------------------
    # Triton (CuPy)
    # ----------------------
    Q_cp = cp.asarray(Q.detach())
    K_cp = cp.asarray(K.detach())
    V_cp = cp.asarray(V.detach())
    dO_cp = cp.asarray(dO.detach())
    
    Q, K, V, O_cp, M = fused_sdpa_forward(Q_cp, K_cp, V_cp, causal=causal)
    dQ_cp, dK_cp, dV_cp = fused_sdpa_backward(dO_cp, Q_cp, K_cp, V_cp, O_cp, M, causal=causal)

    # ----------------------
    # Compare results
    # ----------------------
    cp.testing.assert_allclose(O_cp, cp.asarray(out_ref.detach()), rtol=1e-2, atol=1e-2)
    cp.testing.assert_allclose(dQ_cp, dQ_ref, rtol=1e-2, atol=1e-2)
    cp.testing.assert_allclose(dK_cp, dK_ref, rtol=1e-2, atol=1e-2)
    cp.testing.assert_allclose(dV_cp, dV_ref, rtol=1e-2, atol=1e-2)

    print(f"Passed SDPA test: causal={causal}, L={L}, dtype=float16")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["torch", "triton_cupy", "triton_cupy_dlpack"],
        line_names=["PyTorch SDPA", "Triton Cupy", "Triton Cupy DLPack"],
        styles=[("green","-"),("red","--"),("orange","-.")],
        ylabel="TFLOPS",
        plot_name="flash_attn_fwd",
        args={"mode": "fwd", "causal": True, "dtype": "float16"},
    )
)
def bench_sdpa_forward(SEQ_LEN, mode, provider, causal, dtype, device="cuda"):
    
    BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
    sm_scale = 1.0 / HEAD_DIM**0.5
    is_fp16 = dtype=="float16"
    
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, 
                        dtype=torch.float16 if is_fp16 else torch.float32,
                        device=device, requires_grad=(mode=="bwd"))
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    if provider in ["torch"]:
        
        dO = torch.randn_like(q) if mode=="bwd" else None
        # Triton outputs for torch inputs
        O = torch.empty_like(q)
        M = torch.empty(BATCH, N_HEADS, SEQ_LEN, dtype=torch.float32, device=device)
        D = torch.empty_like(M)
    
    elif provider in ["triton_cupy_dlpack", "triton_cupy"]:
        # Create CuPy arrays then convert to PyTorch via DLPack (zero-copy)
        dtype_cp = cp.float16 if is_fp16 else cp.float32
        q = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp)
        k = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp)
        v = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp)
        dO = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp) if mode=="bwd" else None
        
        dO = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp) if mode=="bwd" else None
        # preallocate outputs
        O = cp.empty_like(q)
        M = cp.empty((BATCH, N_HEADS, SEQ_LEN), dtype=cp.float32)
        D = cp.empty_like(M)
    
    if provider=="torch":
        if mode=="fwd":
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
        else:
            O = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
            fn = lambda: O.backward(dO, retain_graph=True)
    elif provider in ["triton_cupy_dlpack", "triton_cupy"]:
        if mode=="fwd":
            fn = lambda: fused_sdpa_forward(q, k, v, causal, softmax_scale=None)
        else:
            fn = lambda: fused_sdpa_backward(dO, q, k, v, O, M, causal, softmax_scale=sm_scale)
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    
    flops_per_matmul = 2.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode=="bwd":
        total_flops *= 2.5
    tflops = total_flops / (ms * 1e-3) / 1e12
    return tflops


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["torch", "triton_cupy", "triton_cupy_dlpack"],
        line_names=["PyTorch SDPA", "Triton Cupy", "Triton Cupy DLPack"],
        styles=[("green","-"),("red","--"),("orange","-.")],
        ylabel="TFLOPS",
        plot_name="flash_attn_bwd",
        args={"mode": "bwd", "causal": True, "dtype": "float16"},
    )
)
def bench_sdpa_backward(SEQ_LEN, mode, provider, causal, dtype, device="cuda"):
    
    BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
    sm_scale = 1.0 / HEAD_DIM**0.5
    is_fp16 = dtype=="float16"
    
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, 
                        dtype=torch.float16 if is_fp16 else torch.float32,
                        device=device, requires_grad=(mode=="bwd"))
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    
    if provider in ["torch"]:
        
        dO = torch.randn_like(q) if mode=="bwd" else None
        # Triton outputs for torch inputs
        O = torch.empty_like(q)
        M = torch.empty(BATCH, N_HEADS, SEQ_LEN, dtype=torch.float32, device=device)
        D = torch.empty_like(M)
    
    elif provider in ["triton_cupy_dlpack", "triton_cupy"]:
        # Create CuPy arrays then convert to PyTorch via DLPack (zero-copy)
        dtype_cp = cp.float16 if is_fp16 else cp.float32
        q = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp)
        k = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp)
        v = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp)
        dO = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp) if mode=="bwd" else None
        
        dO = cp.random.normal(size=(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)).astype(dtype_cp) if mode=="bwd" else None
        # preallocate outputs
        O = cp.empty_like(q)
        M = cp.empty((BATCH, N_HEADS, SEQ_LEN), dtype=cp.float32)
        D = cp.empty_like(M)
    
    if provider=="torch":
        if mode=="fwd":
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
        else:
            O = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
            fn = lambda: O.backward(dO, retain_graph=True)
    elif provider in ["triton_cupy_dlpack", "triton_cupy"]:
        if mode=="fwd":
            fn = lambda: fused_sdpa_forward(q, k, v, causal, softmax_scale=None)
        else:
            fn = lambda: fused_sdpa_backward(dO, q, k, v, O, M, causal, softmax_scale=sm_scale)
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    
    flops_per_matmul = 2.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if mode=="bwd":
        total_flops *= 2.5
    tflops = total_flops / (ms * 1e-3) / 1e12
    return tflops

TESTS_AND_BENCH = {"matmul": 
                        {
                            "test": ["test_matmul"], 
                            "bench": [benchmark_matmul]
                        },
                   
                   "softmax":
                        {
                            "test": ["test_softmax"],
                            "bench": [benchmark_softmax_forward, benchmark_softmax_backward]
                        },

                    "conv2d":
                        {
                            "test": ["test_conv2d"],
                            "bench": [benchmark_conv2d_forward, benchmark_conv2d_backward]
                        },

                    "layernorm":

                        {
                            "test": ["test_layernorm"],
                            "bench": [benchmark_layernorm_forward, benchmark_layernorm_backward]
                        },

                    "sdpa":

                        {
                            "test": ["test_sdpa"],
                            "bench": [bench_sdpa_forward, bench_sdpa_backward]
                        }
                   }

def main():

    parser = argparse.ArgumentParser("Benchmark Different Fused Operations")
    parser = argparse.ArgumentParser(description="Benchmark fused ops")
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["all"],
        choices=tuple(TESTS_AND_BENCH.keys()),
        help="Which ops to benchmark: matmul, softmax, cross_entropy, or 'all'"
    )
    parser.add_argument("--benchmark_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--save_path", type=str)
    
    args = parser.parse_args()
    
    print(f"Will Save Benchmark Results to {args.save_path}")
    test_path = Path(__file__).resolve()

    if "all" in args.ops:
        selected = list(TESTS_AND_BENCH.keys())
    else:
        selected = [op for op in args.ops if op in TESTS_AND_BENCH]

    print(f"Running benchmarks for: {', '.join(selected)}\n")

    for op in selected:
        meta = TESTS_AND_BENCH[op]
        test_func = meta["test"]
        bench_func = meta["bench"]

        if not args.benchmark_only: 
            for test in test_func:
                pytest.main([f"{test_path}::{test}", "-v", "-s"])

        if not args.test_only:
            for bench in bench_func:
                bench.run(print_data=True, save_path=args.save_path)

if __name__ == "__main__":
    main()