"""
SwiGLU activation functions follow the pattern of:
swiglu(x,y) = x * silu(y) = x * y * sigmoid(y) 
Fixed to match Liger Kernel implementation exactly
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""
import cupy as cp
import triton
import triton.language as tl
from .activations import silu_forward_kernel, silu_backward_kernel
from .utils import calc_num_warps
from .utils import TritonPtr
from .utils import get_dtype_flag

@triton.jit
def silu(x):
    """Inline SiLU function"""
    return x * tl.sigmoid(x)

@triton.jit
def swiglu_forward_kernel(
    a_ptr, 
    b_ptr, 
    out_ptr, 
    stride_a,
    stride_b,
    stride_out,
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)
    
    a_ptr = tl.cast(a_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))
    b_ptr = tl.cast(b_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))
    out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))

    # Locate start index, a and b have their own row strides so the two halves of a chunked
    # projection (rows of 2 * n_cols) can be read in place
    a_ptr += program_id * stride_a
    b_ptr += program_id * stride_b
    out_ptr += program_id * stride_out

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data, sigmoid requires float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    
    # Compute: out = silu(a) * b
    out_row = silu(a_row).cast(b_row.dtype) * b_row
    
    tl.store(out_ptr + col_offsets, out_row, mask=mask)

@triton.jit
def swiglu_backward_kernel(
    dout_ptr, 
    a_ptr,
    b_ptr, 
    stride_dout,
    stride_a,
    stride_b,
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    
    a_ptr = tl.cast(a_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))
    b_ptr = tl.cast(b_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))
    dout_ptr = tl.cast(dout_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))

    # Locate start index
    dout_ptr += program_id * stride_dout
    a_ptr += program_id * stride_a
    b_ptr += program_id * stride_b
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data
    dout_row = tl.load(dout_ptr + col_offsets, mask=mask, other=0)

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    
    # Recompute forward values to save memory
    silu_a = silu_forward_kernel(a_row)

    db_row = dout_row * silu_a
    da_row = dout_row * silu_backward_kernel(a_row) * b_row
    
    # Overwrite original a, b tensors with our grads to avoid extra memory allocation
    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)

def _rows(x, n_cols):
    """
    (..., n_cols) -> (n_rows, n_cols) WITHOUT copying when possible. The kernels only need each row to be
    unit strided along the last dim (row stride can be anything), which holds for the chunked halves of a
    contiguous projection. Anything else gets one contiguous copy.
    """
    if hasattr(x, "is_contiguous"):  # torch
        if x.dim() == 0 or x.stride(-1) != 1:
            x = x.contiguous()
        rows = x.reshape(-1, n_cols)
        return rows if rows.stride(-1) == 1 else rows.contiguous()
    if x.strides[-1] != x.itemsize:
        x = cp.ascontiguousarray(x)
    rows = x.reshape(-1, n_cols)
    return rows if rows.strides[-1] == rows.itemsize else cp.ascontiguousarray(rows)

def fused_swiglu_forward(a, b, use_dlpack=True):

    ori_shape = a.shape
    n_cols = ori_shape[-1]
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = calc_num_warps(BLOCK_SIZE)

    a = _rows(a, n_cols)
    b = _rows(b, n_cols)
    n_rows = a.shape[0]

    with cp.cuda.Device(a.device.id):
        c = cp.empty((n_rows, n_cols), dtype=a.dtype)

        swiglu_forward_kernel[(n_rows,)](
            TritonPtr(a),
            TritonPtr(b),
            TritonPtr(c),
            a.strides[0] // a.itemsize,
            b.strides[0] // b.itemsize,
            c.strides[0] // c.itemsize,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            DTYPE_FLAG=get_dtype_flag(a.dtype)
        )

    # Return a and b as well (views of the inputs when no copy was needed) for the backward
    return a, b, c.reshape(ori_shape)

def fused_swiglu_backward(output_grad, a, b, use_dlpack=True):
    """
    a, b are the arrays returned by fused_swiglu_forward (any (..., n_cols) shape works). They are overwritten in
    place with the gradients (they may be views of the original inputs, which the forward never needs again)
    """

    ori_shape = output_grad.shape
    n_cols = ori_shape[-1]
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = calc_num_warps(BLOCK_SIZE)

    a = _rows(a, n_cols)
    b = _rows(b, n_cols)
    dc = _rows(output_grad, n_cols)
    n_rows = dc.shape[0]

    with cp.cuda.Device(a.device.id):
        swiglu_backward_kernel[(n_rows,)](
            TritonPtr(dc),
            TritonPtr(a),
            TritonPtr(b),
            dc.strides[0] // dc.itemsize,
            a.strides[0] // a.itemsize,
            b.strides[0] // b.itemsize,
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            DTYPE_FLAG=get_dtype_flag(a.dtype)
        )

    # a and b now contain gradients
    return a.reshape(ori_shape), b.reshape(ori_shape)