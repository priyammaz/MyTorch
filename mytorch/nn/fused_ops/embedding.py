"""
This is an Embedding Matrix Implementation to replace the .add.at that we use 
in the backward indexing op (__getitem__)

Code inspiration came from Liger Kernels: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/experimental/embedding.py
"""

import torch
import cupy as cp
import triton
import triton.language as tl

@triton.jit
def embedding_forward_kernel(
    embeddings_ptr, 
    indices_ptr, 
    output_ptr, 
    n_elements, 
    embedding_dim: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, # How many embeds to load at a time 
    BLOCK_SIZE_N: tl.constexpr,  # Block along the embed vector we want to load
    DTYPE_FLAG: tl.constexpr
):
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    ### Cast Pointers ###
    indices_ptr = tl.cast(indices_ptr, tl.pointer_type(tl.int32))
    embeddings_ptr = tl.cast(embeddings_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))

    ### Get the indexes of what embeddings we are loading ###
    start_m = pid_m * BLOCK_SIZE_M
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = start_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

    ### Get the indexes of what part of those embeddings we are loading ###
    start_n = pid_n * BLOCK_SIZE_N
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    ### Get the offsets now (BLOCK_SIZE_M, 1) + (1 x BLOCK_SIZE_N)###
    ### each embedding has embed_dim number of values, so we advance by that ###
    ### much to get the correct starting pointer ###
    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(embeddings_ptr + embedding_offsets, 
                         mask=mask_m[:, None] & mask_n[None, :], 
                         other=0.0)
    
    ### This is just a copy op, we grabbed from our embedding matrix and place into our output matrix ###
    output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    tl.store(output_ptr + output_offsets, embeddings, mask=mask_m[:, None] & mask_n[None, :])

@triton.jit
def embedding_backward_kernel(
    grad_output_ptr, 
    grad_weight_ptr, 
    indices_ptr, 
    n_elements, 
    embedding_dim: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16
):
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    ### Cast Pointers ###
    indices_ptr = tl.cast(indices_ptr, tl.pointer_type(tl.int32))
    grad_weight_ptr = tl.cast(grad_weight_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    grad_output_ptr = tl.cast(grad_output_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    
    ### Get the indexes of what embeddings we are loading ###
    start_m = pid_m * BLOCK_SIZE_M
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

    ### Get the indexes of what part of those embeddings we are loading ###
    start_n = pid_n * BLOCK_SIZE_N
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    ### Get the grad cooresponding to this block ###
    grad_output = tl.load(
        grad_output_ptr + offsets_m[:, None] * embedding_dim + offsets_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :], 
        other=0.0
    )

    ### Now this is an accumulate op. We want to add the grad to our embeddings grad ###
    ### at their cooresponding indices ###
    grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]

    ### Atomic add to handle race conditions ###
    tl.atomic_add(
        grad_weight_ptr + grad_weight_offsets,
        grad_output,
        mask=mask_m[:, None] & mask_n[None, :],
    )

def fused_embedding_forward(embeddings, indices, use_dlpack=True):
    
    ### Only support int32 indexes ###
    indices = indices.astype("int32")

    if use_dlpack:

        ### Get the original shape ###
        original_indices_shape = indices.shape

        ### Flatten ###
        indices = indices.reshape(-1)

        ### Convert to torch ###
        embeddings = torch.utils.dlpack.from_dlpack(embeddings)
        indices = torch.utils.dlpack.from_dlpack(indices)
        
        ### Check Contiguous ###
        if not embeddings.is_contiguous():
            embeddings = embeddings.contiguous()
        if not indices.is_contiguous():
            indices = indices.contiguous()

        ### Create an empty output to copy embeds into ###
        output = torch.empty(
            indices.shape[0], # Number of tokens in total  
            embeddings.shape[1], # Embed dim per token
            device=indices.device, 
            dtype=embeddings.dtype
        )

        ### Get our dimensions ###
        n_elements = indices.shape[0]
        embed_dim = embeddings.shape[1]

        ### Vocab size is typically much larger than the embed_dim so lets set our ###
        ### block size based on the embed dim! ###
        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embed_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embed_dim))
        
        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embed_dim, BLOCK_SIZE_N)
        )

        embedding_forward_kernel[grid](
            embeddings_ptr=embeddings, 
            indices_ptr=indices, 
            output_ptr=output, 
            n_elements=n_elements, 
            embedding_dim=embed_dim, 
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            DTYPE_FLAG=0 if embeddings.dtype == torch.float32 else 1
        )

        return cp.from_dlpack(output.reshape(*original_indices_shape, -1))
    
    else:

        ### Get the original shape ###
        original_indices_shape = indices.shape

        ### Flatten ###
        indices = indices.reshape(-1)

        ### Check Contiguous ###
        if not embeddings.flags.c_contiguous:
            embeddings = cp.ascontiguousarray(embeddings)
        if not indices.flags.c_contiguous:
            indices = cp.ascontiguousarray(indices)

        ### Create an empty output to copy embeds into ###
        with cp.cuda.Device(indices.device.id):
            output = cp.empty(
                (indices.shape[0], embeddings.shape[1]),
                dtype=embeddings.dtype
            )

        ### Get our dimensions ###
        n_elements = indices.shape[0]
        embed_dim = embeddings.shape[1]

        ### Vocab size is typically much larger than the embed_dim so lets set our ###
        ### block size based on the embed dim! ###
        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embed_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embed_dim))
        
        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embed_dim, BLOCK_SIZE_N)
        )

        embedding_forward_kernel[grid](
            embeddings_ptr=embeddings.data.ptr, 
            indices_ptr=indices.data.ptr, 
            output_ptr=output.data.ptr, 
            n_elements=n_elements, 
            embedding_dim=embed_dim, 
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            DTYPE_FLAG=0 if embeddings.dtype == torch.float32 else 1
        )

        return cp.from_dlpack(output.reshape(*original_indices_shape, -1))

def fused_embedding_backward(grad_output, embeddings, indices, use_dlpack=True):

    ### Only support int32 indexes ###
    indices = indices.astype("int32")

    if use_dlpack:

        ### Flatten ###
        indices = indices.reshape(-1)
        
        grad_output = torch.utils.dlpack.from_dlpack(grad_output)
        embeddings = torch.utils.dlpack.from_dlpack(embeddings)
        indices = torch.utils.dlpack.from_dlpack(indices)

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        if not embeddings.is_contiguous():
            embeddings = embeddings.contiguous()
        if not indices.is_contiguous():
            indices = indices.contiguous()

        grad_weight = torch.zeros_like(embeddings)

        n_elements = indices.shape[0]
        embed_dim = embeddings.shape[1]

        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embed_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embed_dim))
        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embed_dim, BLOCK_SIZE_N),
        )

        embedding_backward_kernel[grid](
            grad_output,
            grad_weight,
            indices,
            n_elements,
            embedding_dim=embed_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            DTYPE_FLAG=0 if embeddings.dtype == torch.float32 else 1
        )

        return cp.from_dlpack(grad_weight)
    
    else:

        ### Flatten ###
        indices = indices.reshape(-1)

        if not grad_output.flags.c_contiguous:
            grad_output = cp.ascontiguousarray(grad_output)
        if not embeddings.flags.c_contiguous:
            embeddings = cp.ascontiguousarray(embeddings)
        if not indices.flags.c_contiguous:
            indices = cp.ascontiguousarray(indices)

        grad_weight = cp.zeros_like(embeddings)

        n_elements = indices.shape[0]
        embed_dim = embeddings.shape[1]

        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embed_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embed_dim))

        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embed_dim, BLOCK_SIZE_N),
        )

        embedding_backward_kernel[grid](
            grad_output.data.ptr,
            grad_weight.data.ptr,
            indices.data.ptr,
            n_elements,
            embedding_dim=embed_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            DTYPE_FLAG=0 if embeddings.dtype == cp.float32 else 1
        )

        return grad_weight
        
if __name__ == "__main__":

    from torch.nn import Embedding

    def test_fused_embedding():
        torch.manual_seed(0)

        # --- Config ---
        vocab_size = 10
        embed_dim = 16
        batch_size = 4
        seq_len = 3

        embedding = Embedding(vocab_size, embed_dim, _weight=torch.randn(vocab_size, embed_dim, device="cuda", dtype=torch.float16))
        indices = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.int32)
        embeddings_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(embedding.weight))
        indices_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices))
        out_cp = fused_embedding_forward(embeddings_cp, indices_cp, use_dlpack=False)
        out_torch = embedding(indices)
        torch.testing.assert_close(torch.tensor(out_cp.get(), device="cuda", dtype=out_torch.dtype), out_torch, atol=1e-3, rtol=1e-3)
        print("Forward pass matches PyTorch embedding output")

        grad_output = torch.randn_like(out_torch)
        grad_output_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(grad_output))
        grad_weight_cp = fused_embedding_backward(grad_output_cp, embeddings_cp, indices_cp, use_dlpack=True)
        grad_weight_torch = torch.zeros_like(embedding.weight)
        grad_weight_torch.index_add_(0, indices.view(-1), grad_output.view(-1, embed_dim))
        torch.testing.assert_close(torch.tensor(grad_weight_cp.get(), device="cuda", dtype=grad_weight_torch.dtype), grad_weight_torch, atol=1e-3, rtol=1e-3)
        print("Backward pass matches PyTorch index_add_ gradient")

    test_fused_embedding()

