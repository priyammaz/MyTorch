"""
Simple implementation of RMSNorm! This is basically the same as the 
Layernorm, just simpler. I used the LigerKernel as a reference!
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py
"""
import cupy as cp
import torch
import triton
import triton.language as tl
from .utils import calc_num_warps
from .flags import DLPACK_DISABLE

# Your Triton kernels here
@triton.jit
def rms_norm_forward_kernel(
    output_ptr, 
    output_stride,
    rstd_ptr,
    input_ptr, 
    input_stride, 
    gamma_ptr, 
    DTYPE_FLAG: tl.constexpr, 
    eps: tl.constexpr, 
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):  
    row_idx = tl.program_id(0)

    ### Cast Pointers ###
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    output_ptr = tl.cast(output_ptr, tl.pointer_type(pointer_dtype))
    input_ptr = tl.cast(input_ptr, tl.pointer_type(pointer_dtype))
    rstd_ptr = tl.cast(rstd_ptr, tl.pointer_type(pointer_dtype))
    
    ### If Gamma exists then we will also use it ###
    if gamma_ptr is not None:
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(pointer_dtype))
        
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    ### Advance all our pointers ###
    output_ptr += row_idx * output_stride
    input_ptr += row_idx * input_stride
    rstd_ptr += row_idx
    
    ### Load our Data ###
    input_row = tl.load(input_ptr + col_offsets, mask=mask, other=0.)
    if gamma_ptr is not None:
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.)
    
    ### Compute RMS ###
    mean_square = tl.sum(input_row * input_row, axis=0) / n_cols
    inv_root_mean_square = tl.rsqrt(mean_square + eps)
    tl.store(rstd_ptr, inv_root_mean_square)
    
    ### Store our output ###
    input_row *= inv_root_mean_square
    if gamma_ptr is not None:
        output = input_row * gamma
    else:
        output = input_row
    tl.store(output_ptr + col_offsets, output, mask=mask)

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"]*args["ROW_BLOCK_SIZE"])})
@triton.jit
def rmsnorm_gamma_kernel_backward(
    dgamma_ptr, 
    x_hat_ptr, 
    dy_ptr,
    x_hat_stride, 
    dy_stride,
    d_gamma_row_stride, 
    DTYPE_FLAG: tl.constexpr,
    n_rows: tl.constexpr, 
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr, 
    ROW_BLOCK_SIZE: tl.constexpr
):
    
    """
    y = x_hat * gamma

    Then dgamma = (dL/dy) * x_hat
    """
    col_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    pointer_type = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    dgamma_ptr = tl.cast(dgamma_ptr, tl.pointer_type(pointer_type))
    x_hat_ptr = tl.cast(x_hat_ptr, tl.pointer_type(pointer_type))
    dy_ptr = tl.cast(dy_ptr, tl.pointer_type(pointer_type))
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = tl.arange(0, ROW_BLOCK_SIZE)
    col_range = col_idx * BLOCK_SIZE + col_offsets
    row_range = row_idx * ROW_BLOCK_SIZE + row_offsets
    
    col_mask = col_range < n_cols
    row_mask = row_range < n_rows
    mask = row_mask[:, None] & col_mask[None, :]
    
    dy_ptr += row_range[:, None] * dy_stride + col_range[None, :]
    x_hat_ptr += row_range[:, None] * x_hat_stride + col_range[None, :]
    
    dy = tl.load(dy_ptr, mask=mask, other=0.)
    x_hat = tl.load(x_hat_ptr, mask=mask, other=0.)
    dgamma = tl.sum(dy * x_hat, axis=0)
    
    dgamma_offsets = row_idx * d_gamma_row_stride + col_range
    tl.store(dgamma_ptr + dgamma_offsets, dgamma, mask=col_mask)

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def rmsnorm_kernel_backward(
    dx_ptr,              # (N, E) output gradient wrt input
    dy_ptr,              # (N, E) upstream gradient
    x_ptr,               # (N, E) original input from forward
    gamma_ptr,           # (E,) scale parameter
    rstd_ptr,            # (N,) per-row reciprocal std (1/RMS)
    dgamma_ptr,          # (E,) gradient for gamma - ALWAYS FP32 for accumulation
    dx_row_stride,
    dy_row_stride,
    x_row_stride,
    DTYPE_FLAG: tl.constexpr, 
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    """
    RMSNorm backward pass for input gradients.
    
    Given: y = (x / RMS(x)) * gamma = x * rstd * gamma
    where rstd = 1/sqrt(mean(x^2) + eps)
    
    The gradient formula is:
    dL/dx = rstd * gamma * [dy - (1/n) * x_hat * sum(dy * x_hat * gamma)]
    where x_hat = x * rstd
    
    Equivalent form (matching Liger reference):
    dx = rstd * [m - (1/n) * rstd^2 * sum(m * x) * x]
    where m = dy * gamma
    """
    row_idx = tl.program_id(0)
    
    ### Map Pointers To Correct Dtype ###
    pointer_type = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    dx_ptr = tl.cast(dx_ptr, tl.pointer_type(pointer_type))
    dy_ptr = tl.cast(dy_ptr, tl.pointer_type(pointer_type))
    x_ptr = tl.cast(x_ptr, tl.pointer_type(pointer_type))
    rstd_ptr = tl.cast(rstd_ptr, tl.pointer_type(pointer_type))
    
    if gamma_ptr is not None:
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(pointer_type))
    
    # dgamma is ALWAYS fp32 for atomic accumulation
    if dgamma_ptr is not None:
        dgamma_ptr = tl.cast(dgamma_ptr, tl.pointer_type(tl.float32))
    
    ### Get correct row pointers ###
    dy_row_start_ptr = dy_ptr + row_idx * dy_row_stride
    x_row_start_ptr = x_ptr + row_idx * x_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    dy_ptrs = dy_row_start_ptr + col_offsets
    x_ptrs = x_row_start_ptr + col_offsets
    
    ### Mask out invalid positions ###
    mask = col_offsets < n_cols
    
    ### Load Data - convert to float32 for computation ###
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row_idx).to(tl.float32)
    
    if gamma_ptr is not None:
        gamma_ptrs = gamma_ptr + col_offsets
        gamma = tl.load(gamma_ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        gamma = 1.0
    
    ### Compute normalized input ###
    x_hat = x * rstd
    
    ### Compute m = dy * gamma ###
    m = dy * gamma
    
    ### Core backward formula for RMSNorm ###
    # dx = rstd * [m - (1/n) * rstd^2 * sum(m * x) * x]
    sum_m_x = tl.sum(m * x, axis=0)
    dx = rstd * m
    dx += rstd * (-(1.0 / n_cols) * (rstd * rstd * sum_m_x) * x)
    
    ### Store dx result ###
    dx_row_start_ptr = dx_ptr + row_idx * dx_row_stride
    dx_ptrs = dx_row_start_ptr + col_offsets
    tl.store(dx_ptrs, dx.to(pointer_type), mask=mask)
    
    ### Accumulate dgamma in FP32 ###
    if dgamma_ptr is not None:
        dgamma = dy * x_hat  # Already in FP32
        tl.atomic_add(dgamma_ptr + col_offsets, dgamma, mask=mask)  # Keep in FP32!

def fused_rmsnorm_forward(x, gamma, eps=1e-5, training=True, use_dlpack=True):
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if use_dlpack:

        x = torch.utils.dlpack.from_dlpack(x)
        if gamma is not None:
            gamma = torch.utils.dlpack.from_dlpack(gamma)

        y = torch.empty_like(x)
        rstd = torch.empty(n_rows, dtype=x.dtype, device=x.device)
        
        rms_norm_forward_kernel[(n_rows,)](
            y, 
            y.stride(0),
            rstd,
            x, 
            x.stride(0),
            gamma,
            DTYPE_FLAG=0 if x.dtype == torch.float32 else 1,
            eps=eps,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )

        y = cp.from_dlpack(y)
        rstd = cp.from_dlpack(rstd)

        return y, rstd

def fused_rmsnorm_backward(x, rstd, dy, gamma=None, use_dlpack=True):
    """
    Backward pass for fused RMSNorm using Triton kernel.

    Args:
        x: Input tensor (N, E)
        rstd: Reciprocal RMS per row (N,)
        dy: Upstream gradient (N, E)
        gamma: Scale parameter (E,)
        use_dlpack: Whether to use DLPack for interop

    Returns:
        dx: Gradient w.r.t input (N, E)
        dgamma: Gradient w.r.t gamma (E,)
    """
    n_rows, n_cols = dy.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if not DLPACK_DISABLE and use_dlpack:
        # Convert to torch
        x = torch.utils.dlpack.from_dlpack(x)
        dy = torch.utils.dlpack.from_dlpack(dy)
        rstd = torch.utils.dlpack.from_dlpack(rstd)
        gamma = torch.utils.dlpack.from_dlpack(gamma) if gamma is not None else None

        # KEY: Use FP32 for gradient accumulation
        grad_dtype = torch.float32
        
        # Allocate grads
        dx = torch.empty_like(dy)
        dgamma = torch.zeros(n_cols, dtype=grad_dtype, device=dy.device) if gamma is not None else None

        # Compute strides
        stride_x = x.stride(0)
        stride_dx = dx.stride(0)
        stride_dy = dy.stride(0)

        # Dtype flag
        dtype_flag = 0 if dy.dtype == torch.float32 else 1

        # Launch kernel
        grid = (n_rows,)
        rmsnorm_kernel_backward[grid](
            dx, dy, x, gamma, rstd, dgamma,
            stride_dx, stride_dy, stride_x,
            dtype_flag, n_cols, BLOCK_SIZE,
        )

        # Convert back to CuPy
        dx = cp.from_dlpack(dx)
        
        # Convert dgamma back to original dtype
        if gamma is not None:
            dgamma = cp.from_dlpack(dgamma.to(gamma.dtype))

    else:
        # CuPy path
        grad_dtype = cp.float32
        
        dx = cp.empty_like(dy)
        dgamma = cp.zeros((n_cols,), dtype=grad_dtype) if gamma is not None else None

        stride_x = x.strides[0] // x.itemsize
        stride_dx = dx.strides[0] // dx.itemsize
        stride_dy = dy.strides[0] // dy.itemsize
        dtype_flag = 0 if dy.dtype == cp.float32 else 1
        grid = (n_rows,)

        rmsnorm_kernel_backward[grid](
            dx.data.ptr,
            dy.data.ptr,
            x.data.ptr,
            gamma.data.ptr if gamma is not None else None,
            rstd.data.ptr,
            dgamma.data.ptr if gamma is not None else None,
            stride_dx,
            stride_dy,
            stride_x,
            dtype_flag,
            n_cols,
            BLOCK_SIZE,
        )
        
        # Convert back to original dtype
        if gamma is not None:
            dgamma = dgamma.astype(gamma.dtype)

    return dx, dgamma


if __name__ == "__main__":
    import numpy as np
    def test_rmsnorm(dtype=np.float16, batch_size=4, hidden_size=8, eps=1e-6):
 
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create input and weights
        x_np = np.random.randn(batch_size, hidden_size).astype(dtype)
        gamma_np = np.random.randn(hidden_size).astype(dtype)
        
        # CuPy (for Triton kernels)
        x_cp = cp.array(x_np)
        gamma_cp = cp.array(gamma_np)
        
        # PyTorch
        torch_dtype = torch.float32 if dtype == np.float32 else torch.float16
        x_torch = torch.tensor(x_np, device='cuda', dtype=torch_dtype, requires_grad=True)
        gamma_torch = torch.tensor(gamma_np, device='cuda', dtype=torch_dtype, requires_grad=True)
        
        # Triton forward
        y_triton, rstd_triton = fused_rmsnorm_forward(
            x_cp, gamma_cp, eps=eps, training=True, use_dlpack=True
        )
        torch_rmsnorm = torch.nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=True).cuda()

        torch_rmsnorm.weight.data = gamma_torch.clone()
        y_torch = torch_rmsnorm(x_torch)
        

        y_triton_np = cp.asnumpy(y_triton)
        y_torch_np = y_torch.detach().cpu().numpy()
        
        forward_diff = np.abs(y_triton_np - y_torch_np)
        forward_match = np.allclose(y_triton_np, y_torch_np, rtol=1e-2, atol=1e-2)
        
        print(f"Output max diff: {forward_diff.max():.2e}")
        
        # Create upstream gradient - SAME dtype as input!
        grad_output_np = np.random.randn(batch_size, hidden_size).astype(dtype)
        grad_output_cp = cp.array(grad_output_np)
        grad_output_torch = torch.tensor(grad_output_np, device='cuda', dtype=torch_dtype)
        
        # Triton backward
        dx_triton, dgamma_triton = fused_rmsnorm_backward(
            x_cp, rstd_triton, grad_output_cp, gamma_cp, use_dlpack=True
        )
        
        # PyTorch backward
        y_torch.backward(grad_output_torch)
        dx_torch = x_torch.grad
        dgamma_torch = torch_rmsnorm.weight.grad
        
        # Compare dx
        dx_triton_np = cp.asnumpy(dx_triton)
        dx_torch_np = dx_torch.detach().cpu().numpy()
        dx_diff = np.abs(dx_triton_np - dx_torch_np)
        dx_rel_diff = dx_diff / (np.abs(dx_torch_np) + 1e-8)
        dx_match = np.allclose(dx_triton_np, dx_torch_np, rtol=1e-2, atol=1e-2)
        
        print(f"\ndx max diff: {dx_diff.max():.2e}")
        print(f"dx max relative diff: {dx_rel_diff.max():.2e}")
        
        # Compare dgamma
        dgamma_triton_np = cp.asnumpy(dgamma_triton)
        dgamma_torch_np = dgamma_torch.detach().cpu().numpy()
        dgamma_diff = np.abs(dgamma_triton_np - dgamma_torch_np)
        dgamma_rel_diff = dgamma_diff / (np.abs(dgamma_torch_np) + 1e-8)
        dgamma_match = np.allclose(dgamma_triton_np, dgamma_torch_np, rtol=1e-2, atol=1e-2)
        
        print(f"\ndgamma max diff: {dgamma_diff.max():.2e}")
        print(f"dgamma max relative diff: {dgamma_rel_diff.max():.2e}")

        print(f"Forward:  {'PASS' if forward_match else 'FAIL'}")
        print(f"dx:       {'PASS' if dx_match else 'FAIL'}")
        print(f"dgamma:   {'PASS' if dgamma_match else 'FAIL'}")
        
        all_pass = forward_match and dx_match and dgamma_match
        print(f"\nOverall:  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
        
        return all_pass


    test_fp32 = test_rmsnorm(dtype=np.float32, batch_size=64, hidden_size=8)
    test_fp16 = test_rmsnorm(dtype=np.float16, batch_size=64, hidden_size=8)
    test_fp16_large = test_rmsnorm(dtype=np.float16, batch_size=367, hidden_size=768)
    