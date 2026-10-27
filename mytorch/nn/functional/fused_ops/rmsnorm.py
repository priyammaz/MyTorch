"""
Simple implementation of RMSNorm! This is basically the same as the 
Layernorm, just simpler. I used the LigerKernel as a reference!
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py
"""
import cupy as cp
import triton
import triton.language as tl
from .utils import calc_num_warps
from .utils import TritonPtr
from .utils import get_dtype_flag

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def rms_norm_forward_kernel(
    output_ptr, 
    output_stride,
    rstd_ptr, # store for backward pass
    input_ptr, 
    input_stride, 
    gamma_ptr, 
    DTYPE_FLAG: tl.constexpr, 
    eps: tl.constexpr, 
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):  
    row_idx = tl.program_id(0)

    ### Cast Pointers ###
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)
    output_ptr = tl.cast(output_ptr, tl.pointer_type(pointer_dtype))
    input_ptr = tl.cast(input_ptr, tl.pointer_type(pointer_dtype))
    if rstd_ptr is not None:
        rstd_ptr = tl.cast(rstd_ptr, tl.pointer_type(pointer_dtype))
    
    ### If Gamma exists then we will also use it ###
    if gamma_ptr is not None:
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(pointer_dtype))
        
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    ### Advance all our pointers ###
    output_ptr += row_idx * output_stride
    input_ptr += row_idx * input_stride

    if rstd_ptr is not None:
        rstd_ptr += row_idx
    
    ### Load our Data ###
    input_row = tl.load(input_ptr + col_offsets, mask=mask, other=0.)
    if gamma_ptr is not None:
        gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.)
    
    ### Compute RMS ###
    mean_square = tl.sum(input_row * input_row, axis=0) / n_cols
    inv_root_mean_square = tl.rsqrt(mean_square + eps)
    
    ### Store our output ###
    normed = input_row * inv_root_mean_square
    if gamma_ptr is not None:
        output = normed * gamma
    else:
        output = normed
    tl.store(output_ptr + col_offsets, output, mask=mask)

    ### Store our rstd if we are training ###
    if rstd_ptr is not None:
        tl.store(rstd_ptr, inv_root_mean_square)
    
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
    pointer_type = tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)
    dx_ptr = tl.cast(dx_ptr, tl.pointer_type(pointer_type))
    dy_ptr = tl.cast(dy_ptr, tl.pointer_type(pointer_type))
    x_ptr = tl.cast(x_ptr, tl.pointer_type(pointer_type))
    rstd_ptr = tl.cast(rstd_ptr, tl.pointer_type(pointer_type))
    
    if gamma_ptr is not None:
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(pointer_type))
        gamma_ptrs = gamma_ptr + col_offsets
        gamma = tl.load(gamma_ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        gamma = 1.0
    
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
    """
    RMSNorm forward pass using Triton kernel.
    
    Args:
        x: Input tensor (N, E)
        gamma: Scale parameter (E,)
        eps: Epsilon for numerical stability
        training: Whether in training mode (unused, kept for API compatibility)
        use_dlpack: Whether to use DLPack for interop
    
    Returns:
        y: Output (N, E)
        rstd: Reciprocal RMS per row (N,)
    """
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    y = cp.empty_like(x)

    with cp.cuda.Device(x.device):
        rstd = cp.empty((n_rows,), dtype=x.dtype)
    
    # Compute strides in elements
    x_row_stride = x.strides[0] // x.itemsize
    y_row_stride = y.strides[0] // y.itemsize
    
    # Map dtype to Triton flag
    dtype_flag = get_dtype_flag(x.dtype)
    
    # Launch kernel
    grid = (n_rows,)
    rms_norm_forward_kernel[grid](
        TritonPtr(y),
        y_row_stride,
        TritonPtr(rstd) if rstd is not None else None,  
        TritonPtr(x),
        x_row_stride,
        TritonPtr(gamma) if gamma is not None else None,
        DTYPE_FLAG=dtype_flag,
        eps=eps,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if training:
        return y, rstd
    
    return y
    
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

    # CuPy path
    grad_dtype = cp.float32
    
    dx = cp.empty_like(dy)
    dgamma = cp.zeros((n_cols,), dtype=grad_dtype) if gamma is not None else None

    stride_x = x.strides[0] // x.itemsize
    stride_dx = dx.strides[0] // dx.itemsize
    stride_dy = dy.strides[0] // dy.itemsize
    dtype_flag = get_dtype_flag(dy.dtype)
    grid = (n_rows,)

    rmsnorm_kernel_backward[grid](
        TritonPtr(dx),
        TritonPtr(dy),
        TritonPtr(x),
        TritonPtr(gamma) if gamma is not None else None,
        TritonPtr(rstd),
        TritonPtr(dgamma) if gamma is not None else None,
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