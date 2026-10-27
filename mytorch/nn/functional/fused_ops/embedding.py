"""
This is an Embedding Matrix Implementation to replace the .add.at that we use 
in the backward indexing op (__getitem__)

Code inspiration came from Liger Kernels: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/experimental/embedding.py
"""

import os
import cupy as cp
import triton
import triton.language as tl
from .utils import get_dtype_flag

### Set MYTORCH_CHECK_EMBEDDING_INDICES=true to validate indices (costs a GPU sync per call) ###
CHECK_INDICES = os.environ.get("MYTORCH_CHECK_EMBEDDING_INDICES", "false").lower() == "true"

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
    embeddings_ptr = tl.cast(embeddings_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))
    output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))

    ### Get the indexes of what embeddings we are loading ###
    start_m = pid_m * BLOCK_SIZE_M
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

    ### Get the indexes of what part of those embeddings we are loading ###
    start_n = pid_n * BLOCK_SIZE_N
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    ### Get the combined mask where our indexing is valid ###
    combined_mask = mask_m[:, None] & mask_n[None, :]

    ### Get the offsets now (BLOCK_SIZE_M, 1) + (1 x BLOCK_SIZE_N)###
    ### each embedding has embed_dim number of values, so we advance by that ###
    ### much to get the correct starting pointer ###
    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(embeddings_ptr + embedding_offsets, 
                         mask=combined_mask, 
                         other=0.0)
    
    ### This is just a copy op, we grabbed from our embedding matrix and place into our output matrix ###
    output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    tl.store(output_ptr + output_offsets, embeddings, mask=combined_mask)

@triton.jit
def embedding_backward_kernel(
    grad_output_ptr, 
    grad_weight_ptr, 
    indices_ptr, 
    n_elements, 
    embedding_dim: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr,
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16, 2 for bfloat16
):
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    ### Cast Pointers ###
    indices_ptr = tl.cast(indices_ptr, tl.pointer_type(tl.int32))
    grad_weight_ptr = tl.cast(grad_weight_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))
    grad_output_ptr = tl.cast(grad_output_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))
    
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
    indices = indices.astype("int32", copy=False)

    ### Bounds checks force a GPU sync every call, so they are opt in ###
    if CHECK_INDICES:
        assert (indices >= 0).all(), f"Negative indices found: {indices.min()}"
        assert (indices < embeddings.shape[0]).all(), f"Index out of bounds: max={indices.max()}, vocab={embeddings.shape[0]}"

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
        DTYPE_FLAG=get_dtype_flag(embeddings.dtype)
    )

    embeds = output.reshape(*original_indices_shape, -1)

    return embeds

def fused_embedding_backward(grad_output, embeddings, indices, use_dlpack=True):

    ### tl.atomic_add is extremely slow for bfloat16, so accumulate in float32 and cast back ###
    if str(embeddings.dtype) == "bfloat16":
        grad_weight = fused_embedding_backward(grad_output.astype(cp.float32),
                                               cp.empty(embeddings.shape, dtype=cp.float32),
                                               indices, use_dlpack=use_dlpack)
        return grad_weight.astype("bfloat16")

    ### Only support int32 indexes ###
    indices = indices.astype("int32", copy=False)

    ### Bounds checks force a GPU sync every call, so they are opt in ###
    if CHECK_INDICES:
        assert (indices >= 0).all(), f"Negative indices found: {indices.min()}"
        assert (indices < embeddings.shape[0]).all(), f"Index out of bounds: max={indices.max()}, vocab={embeddings.shape[0]}"
    
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
        DTYPE_FLAG=get_dtype_flag(embeddings.dtype)
    )

    return grad_weight
