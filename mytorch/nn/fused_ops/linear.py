"""
The linear layer is identical to our `fused_ops/matmul.py` with the inclusion of a fused bias in the 
forward pass!

### Forward Pass ###
y = x@W.T + b

x: [B, I]
W: [O, I]
b: [O,]

#### Backward Pass ###

In our operation y = x@W.T + b, lets just pretend W.T = K
So our operation was really just y = x@K + b 

By our standard formula for derivative of a matmul we will have:

dK = x.T @ d_output
dX = d_output @ K.T
db = d_output.sum(axis=0)


The issue is that when we pass our data to our kernel, we are passing in W, not W.T=K.
So two ways to handle this:

1) Just transpose our weights before passing to the kernel
2) adjust our formula for this

Option 2 is cleaner so lets go with that. 

if dK = x.T @ d_output then dK.T = dW.T.T = dW = (x.T @ d_output).T = d_output.T @ x
if dX = d_output @ K.T = d_output @ W.T.T = d_output @ W
db doesnt change!



"""

import torch
import triton
import triton.language as tl
from .matmul import grouped_matmul_kernel

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,num_warps=2),
    ]

@triton.autotune(configs = get_cuda_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def fused_linear_forward_kernel(
    input_ptr, # input in the shape [B x I]
    weight_ptr, # weights in the shape [O, I] (needs to be transposed)
    out_ptr, # output in the shape [B x O]
    bias_ptr, # bias in the shape [B,]
    M,  
    N, 
    K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE_FLAG: tl.constexpr # 0 for float32, 1 for float16
):
    
    ### Grouping logic as described above! ###
    pid = tl.program_id(axis=0)

    ### Cast our Pointers to the Correct DTYPE ###
    if DTYPE_FLAG == 0:  # float32
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float32))
        weight_ptr = tl.cast(weight_ptr, tl.pointer_type(tl.float32))
        out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float32))
        if bias_ptr is not None:
            bias_ptr = tl.cast(bias_ptr, tl.pointer_type(tl.float32))
    elif DTYPE_FLAG == 1:  # float16
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float16))
        weight_ptr = tl.cast(weight_ptr, tl.pointer_type(tl.float16))
        out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float16))
        if bias_ptr is not None:
            bias_ptr = tl.cast(bias_ptr, tl.pointer_type(tl.float16))

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
    
    input_ptrs = input_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)

    ### This is how we do our transpose! The commented code is what we had 
    ### in our original matmul, but by flipping the stride we transpose our data 
    ### without explicitly doing another tl.trans!
    ### b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    weight_ptrs = weight_ptr + (offs_k[:, None] * stride_bn + offs_n[None, :] * stride_bk)
    
    ### Initializer Accumulator - use float32 for accumulation ###
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    ### Loop over K in chunks of BLOCK_SIZE_K ###
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        
        ### Load Blocks of A and B with proper masking ###
        # Mask for A: check both M and K dimensions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(input_ptrs, mask=a_mask, other=0.0)
        
        # Mask for B: check both K and N dimensions
        b_mask = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        b = tl.load(weight_ptrs, mask=b_mask, other=0.0)
        
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
        input_ptrs += BLOCK_SIZE_K * stride_ak
        weight_ptrs += BLOCK_SIZE_K * stride_bk
    
    ### Cast accumulator to correct dtype ###
    c = accumulator.to(input_ptr.dtype.element_ty)

    ### Add bias is not none ###
    ### Now remember that our output matrix will be [B x O], and for
    ### each output value we have a bais we can add. But we are computing
    ### our final [B x O] in blocks. offs_n from earlier tells us which 
    ### n (columns) we are currently processing. In our case we want to 
    ### grab the same indexes of our bias vector so we can add it right here!
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N))
        bias = bias.to(input_ptr.dtype.element_ty)
        c += bias[None, :]

    ### Identify which block in our output C we will save this in ###
    c_offsets = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    
    ### Identify any invalid positions we dont want to save in ###
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    ### Save it! ###
    tl.store(out_ptr + c_offsets, c, mask=c_mask)


import torch

M, N, K = 64,33,16
# Initialize output tensor with same dtype as input
a = torch.randn(M,K, device="cuda", dtype=torch.float16)
b = torch.randn(N,K, device="cuda", dtype=torch.float16)
c = torch.empty((M, N), device=a.device, dtype=a.dtype)
bias = torch.randn((N,), device="cuda", dtype=a.dtype)

# Define grid
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

fused_linear_forward_kernel[grid](
    input_ptr=a, # input in the shape [B x I]
    weight_ptr=b, # weights in the shape [O, I] (needs to be transposed)
    out_ptr=c, # output in the shape [B x O]
    bias_ptr=bias, # bias in the shape [B,]
    M=M,  
    N=N, 
    K=K,
    stride_am=a.stride(0), 
    stride_ak=a.stride(1),
    stride_bk=b.stride(0), 
    stride_bn=b.stride(1),
    stride_cm=c.stride(0), 
    stride_cn=c.stride(1),
    DTYPE_FLAG= 1 #for float32, 1 for float16
)
print(bias.shape)
print((a@b.T + bias.unsqueeze(0)).shape)
print(a@b.T + bias.unsqueeze(0) - c)
