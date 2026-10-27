"""
Grouped MatMul as described in Triton Docs
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

And really helpful explanation from Evintunador
https://github.com/evintunador/triton_docs_tutorials/blob/main/06_matmul/matmul.py

C = A@B -> [M x K] @ [K x N] = [M x N]

We wont be using this Matmul directly anywhere as Cupy already is using the highly optimized
CUDNN in the backend. This is more of an attempt to understand grouping as we will use it in
our linear layers (with a fused bias) and convolution layers
"""

import cupy as cp
import triton
import triton.language as tl
from .flags import AUTOTUNE_MODE
from .utils import get_dtype_flag
from .utils import TritonPtr
from .autotuner_bench import do_bench_cudagraph

@triton.jit
def naive_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn
):
    # Program ID — each "program" computes one element of C
    pid = tl.program_id(axis=0)  # 0..M*N - 1

    # Compute (m, n) coordinates of this output element
    m = pid // N
    n = pid % N

    # If we're out of bounds, exit
    if (m >= M) or (n >= N):
        return

    # Compute the dot product between A[m, :] and B[:, n]
    acc = 0.0
    for k in range(0, K):
        a_val = tl.load(A_ptr + m * stride_am + k * stride_ak)
        b_val = tl.load(B_ptr + k * stride_bk + n * stride_bn)
        acc += a_val * b_val

    # Write result to output
    tl.store(C_ptr + m * stride_cm + n * stride_cn, acc)

def get_blocked_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5,
                      num_warps=2),
    ]
@triton.autotune(configs = get_blocked_cuda_autotune_config(), key=['M', 'N', 'K'], do_bench=do_bench_cudagraph)
@triton.jit
def blocked_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr
):
    
    pid = tl.program_id(axis=0)

    ### Get how many columns of blocks we have ###
    num_block_cols = tl.cdiv(N, BLOCK_SIZE_N)

    ### Recover our actual block index ###
    block_m = pid // num_block_cols # What row are we on?
    block_n = pid % num_block_cols # What col are we on?
    
    ### Compute Offsets to grab full block of data ###
    offs_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    ### Initializer Accumulator - use float32 for accumulation ###
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    ### Loop over K in chunks of BLOCK_SIZE_K ###
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        
        ### Load Blocks of A and B with proper masking ###
        # Mask for A: check both M and K dimensions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Mask for B: check both K and N dimensions
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        ### Accumulate Result ###
        ### investigate later: There is a discrepancy between pytorch and triton in float32
        ### if we dont include allow_fp32=False we wil have a higher error, but also, 
        ### we then hurt performance on float32. Leaving as is for now! 

        ### This is a well documented issue
        ### https://github.com/triton-lang/triton/issues/4574
        ### https://github.com/triton-lang/triton/issues/5204
        ### https://github.com/triton-lang/triton/issues/2843

        accumulator += tl.dot(a, b) #allow_tf32=False
        
        ### Advance our Offsets to the next chunk of K ###
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    ### Cast accumulator to correct dtype ###
    c = accumulator.to(A_ptr.dtype.element_ty)

    ### Identify which block in our output C we will save this in ###
    c_offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    
    ### Identify any invalid positions we dont want to save in ###
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    ### Save it! ###
    tl.store(C_ptr + c_offsets, c, mask=c_mask)

def get_cuda_autotune_config():

    if AUTOTUNE_MODE == "none":
        return [triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8)]

    if AUTOTUNE_MODE == "max":
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,num_warps=2),
        ]

@triton.autotune(configs = get_cuda_autotune_config(), key=['M', 'N', 'K'], do_bench=do_bench_cudagraph)
@triton.jit
def grouped_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE_FLAG: tl.constexpr # 0 for float32, 1 for float16, 2 for bfloat16
):
    
    ### Grouping logic as described above! ###
    pid = tl.program_id(axis=0)

    ### Cast our Pointers to the Correct DTYPE ###
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)
    A_ptr = tl.cast(A_ptr, tl.pointer_type(pointer_dtype))
    B_ptr = tl.cast(B_ptr, tl.pointer_type(pointer_dtype))
    C_ptr = tl.cast(C_ptr, tl.pointer_type(pointer_dtype))

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    ### EVERYTHING ELSE IS THE SAME AS BEFORE! ###
    ### Compute Offsets to grab full block of data ###
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    ### Initializer Accumulator - use float32 for accumulation ###
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    ### Loop over K in chunks of BLOCK_SIZE_K ###
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        
        ### Load Blocks of A and B with proper masking ###
        # Mask for A: check both M and K dimensions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Mask for B: check both K and N dimensions
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        ### Accumulate Result ###
        ### investigate later: There is a discrepancy between pytorch and triton in float32
        ### if we dont include allow_fp32=False we wil have a higher error, but also, 
        ### we then hurt performance on float32. Leaving as is for now! 

        ### This is a well documented issue
        ### https://github.com/triton-lang/triton/issues/4574
        ### https://github.com/triton-lang/triton/issues/5204
        ### https://github.com/triton-lang/triton/issues/2843
        accumulator += tl.dot(a, b)
        
        ### Advance our Offsets to the next chunk of K ###
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    ### Cast accumulator to correct dtype ###
    c = accumulator.to(A_ptr.dtype.element_ty)

    ### Identify which block in our output C we will save this in ###
    c_offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    
    ### Identify any invalid positions we dont want to save in ###
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    ### Save it! ###
    tl.store(C_ptr + c_offsets, c, mask=c_mask)

def fused_grouped_matmul(a, b, use_dlpack=True):

    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    get_dtype_flag(a.dtype)  # raises on unsupported dtypes
    assert a.dtype == b.dtype, f"a and b have different precision types {a.dtype} and {b.dtype}"
    
    # Initialize output tensor with same dtype as input
    with cp.cuda.Device(a.device.id):
        c = cp.empty(
            (M, N), dtype=a.dtype
        )

    # Define grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    # Launch kernel
    grouped_matmul_kernel[grid](
        TritonPtr(a), 
        TritonPtr(b), 
        TritonPtr(c),
        M, N, K,
        a.strides[0]//a.itemsize, 
        a.strides[1]//a.itemsize,
        b.strides[0]//b.itemsize, 
        b.strides[1]//b.itemsize,
        c.strides[0]//c.itemsize, 
        c.strides[1]//c.itemsize,
        DTYPE_FLAG=get_dtype_flag(a.dtype)
    )
    
    return c