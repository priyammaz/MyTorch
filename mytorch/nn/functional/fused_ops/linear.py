"""
The linear layer is identical to our `fused_ops/matmul.py` with the inclusion of a fused bias in the 
forward pass!

### Forward Pass ###
y = x@W + b

x: [B, I]
W: [I, O]
b: [O,]

#### Backward Pass ###

Technically the backward pass is just a bunch of matmuls, and there isnt anything to fuse. Cupy matmul
is just as fast as our own grouped matmul so theres no gains here to fuse them! 

"""
import cupy as cp
import torch
import triton
import triton.language as tl
from .activations import activation_switcher_forward, _avail_activations
from .flags import DLPACK_DISABLE, AUTOTUNE_MODE

def get_cuda_autotune_config():

    if AUTOTUNE_MODE == "none":
        return [triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8)]
    
    if AUTOTUNE_MODE == "max":
        return [
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        ]

@triton.autotune(configs=get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def fused_linear_forward_kernel(
    input_ptr, 
    weight_ptr, 
    postact_ptr, 
    preact_ptr, 
    bias_ptr,
    M, N, K,
    stride_am, 
    stride_ak, 
    stride_bk, 
    stride_bn, 
    stride_cm, 
    stride_cn, 
    act_func: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
    DTYPE_FLAG: tl.constexpr
):
    pointer_type = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    input_ptr = tl.cast(input_ptr, tl.pointer_type(pointer_type))
    weight_ptr = tl.cast(weight_ptr, tl.pointer_type(pointer_type))
    preact_ptr = tl.cast(preact_ptr, tl.pointer_type(pointer_type))
    if postact_ptr is not None:
        postact_ptr = tl.cast(postact_ptr, tl.pointer_type(pointer_type))
    if bias_ptr is not None:
        bias_ptr = tl.cast(bias_ptr, tl.pointer_type(pointer_type))

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = input_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(input_ptrs, mask=a_mask, other=0.0)
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        b = tl.load(weight_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        input_ptrs += BLOCK_SIZE_K * stride_ak
        weight_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(input_ptr.dtype.element_ty)

    ### Main Difference Here where we fuse the bias into our single kernel ###
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N))
        bias = bias.to(input_ptr.dtype.element_ty)
        c += bias[None, :]

    c_offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(preact_ptr + c_offsets, c, mask=c_mask)

    ### If activation function is provided then use it! ###
    if act_func is not None:
        c = activation_switcher_forward(act_func, c)
        tl.store(postact_ptr + c_offsets, c, mask=c_mask)

def fused_linear_forward(a, w, b=None, act_func="relu", use_dlpack=True):
    """
    Fused Linear Forward Pass using Triton kernel.
    Computes: y = a @ w + b
    a: [B, I]
    w: [I, O]
    b: [O,] (optional)
    """

    # Validate shapes and dtypes
    B, I = a.shape
    I2, O = w.shape
    assert I == I2, "Inner dimensions must match"
    assert a.dtype in (cp.float16, cp.float32), "Only float16/float32 supported"
    assert a.dtype == w.dtype, f"Mismatched dtypes: {a.dtype} vs {w.dtype}"
    if b is not None:
        assert b.shape == (O,), f"Bias shape must be ({O},)"
        assert b.dtype == a.dtype, f"Bias dtype must match input dtype"

    # Check for contiguous memory layout
    if not a.flags.c_contiguous:
        a = cp.ascontiguousarray(a)
    if not w.flags.c_contiguous:
        w = cp.ascontiguousarray(w)
    if b is not None and not b.flags.c_contiguous:
        b = cp.ascontiguousarray(b)

    ### Verify activtion function if provided ###
    if act_func is not None:
        assert act_func in _avail_activations, f"Fused activations must be selected from {_avail_activations}, got {act_func}"
   
    if not DLPACK_DISABLE and use_dlpack:
        a_torch = torch.utils.dlpack.from_dlpack(a)
        w_torch = torch.utils.dlpack.from_dlpack(w)
        b_torch = torch.utils.dlpack.from_dlpack(b) if b is not None else None

        if not a_torch.is_contiguous():
            a_torch = a_torch.contiguous()
        if not w_torch.is_contiguous():
            w_torch = w_torch.contiguous()
        if b_torch is not None and not b_torch.is_contiguous():
            b_torch = b_torch.contiguous()
            
        # Allocate output
        preact_torch = torch.empty((B, O), device=a_torch.device, dtype=a_torch.dtype)
        postact_torch = torch.empty_like(preact_torch) if act_func is not None else None

        # Define grid for Triton launch
        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_M']) *
            triton.cdiv(O, meta['BLOCK_SIZE_N']),
        )

        fused_linear_forward_kernel[grid](
            a_torch, 
            w_torch, 
            postact_torch,
            preact_torch, 
            b_torch,
            B, 
            O, 
            I,
            a_torch.stride(0), 
            a_torch.stride(1),
            w_torch.stride(0), 
            w_torch.stride(1),
            preact_torch.stride(0), 
            preact_torch.stride(1),
            act_func,
            DTYPE_FLAG=0 if a_torch.dtype == torch.float32 else 1
        )


        # Return CuPy tensor via DLPack
        if act_func is not None:
            return cp.from_dlpack(preact_torch), cp.from_dlpack(postact_torch)
        else:
            return cp.from_dlpack(preact_torch), None

    else:

        if not a.flags.c_contiguous:
            a = cp.ascontiguousarray(a)
        if not w.is_contiguous():
            w = cp.ascontiguousarray(w)
        if b is not None and not b.is_contiguous():
            b = cp.ascontiguousarray(b)

        with cp.cuda.Device(a.device.id):
            preact_cp = cp.empty((B, O), dtype=a.dtype)
            postact_cp = cp.empty_like(preact_cp) if act_func is not None else None

        grid = lambda meta: (
            triton.cdiv(B, meta['BLOCK_SIZE_M']) *
            triton.cdiv(O, meta['BLOCK_SIZE_N']),
        )

        fused_linear_forward_kernel[grid](
            a.data.ptr,
            w.data.ptr,
            preact_cp.data.ptr,
            postact_cp.data.ptr,
            b.data.ptr if b is not None else None,
            B, O, I,
            a.strides[0] // a.itemsize,
            a.strides[1] // a.itemsize,
            w.strides[0] // w.itemsize,
            w.strides[1] // w.itemsize,
            preact_cp.strides[0] // preact_cp.itemsize,
            preact_cp.strides[1] // preact_cp.itemsize,
            DTYPE_FLAG=0 if a.dtype == cp.float32 else 1
        )

        return preact_cp