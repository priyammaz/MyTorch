"""
LayerNorm fused kernel inspired by https://github.com/lucidrains/triton-transformer/blob/main/triton_transformer/layernorm.py
"""

import cupy as cp
import torch
import triton
import triton.language as tl
from .utils import calc_num_warps
from .flags import DLPACK_DISABLE

def layernorm_naive(x, weight, bias=None, eps=1e-5):

    # Original shape
    *dims, embed_dim = x.shape

    # Flatten if necessary (e.g. [B, T, D] -> [-1, D])
    if len(dims) > 1:
        x_reshaped = x.reshape(-1, embed_dim)
    else:
        x_reshaped = x

    # Mean and variance along last dim
    mean = x_reshaped.mean(axis=-1, keepdims=True)
    var = x_reshaped.var(axis=-1, keepdims=True)

    # Normalize
    normed = (x_reshaped - mean) / cp.sqrt(var + eps)

    # Affine transform
    out = normed * weight.reshape(1, -1)
    if bias is not None:
        out = out + bias.reshape(1, -1)

    # Restore shape
    if len(dims) > 1:
        out = out.reshape(*dims, embed_dim)

    return out

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def layernorm_kernel_forward_training(
    output_ptr, 
    inv_var_ptr, # Need for Backward Pass (N,)
    x_hat_ptr, # Need for Backward Pass (N x E)
    input_ptr, 
    gamma_ptr,  # 1D vector shared across all samples (E, )
    beta_ptr,   # 1D vector shared across all samples (E, )
    input_row_stride, 
    output_row_stride, 
    x_hat_row_stride, 
    dtype_flag: tl.constexpr, # Flag for if our data is float32 or float16
    eps: tl.constexpr,
    n_cols: tl.constexpr, # Dimensionality of our embeddings
    BLOCK_SIZE: tl.constexpr # closest power of 2 to our dim of embeddings 
):

    """
    Operation:

    gamma * ((x - mu)/(var + eps)) + beta

    This kernel expects an (N x E) dimension matrix and so we can write an easy
    kernel that has every block normalize a single row! A more optimal method can be found
    in the Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py

    But we are trying to find a balance between giving our modules some teeth, but also easily 
    understandable! Im sorry to all the hardcore kernel engineers! 

    For backprop we need our Mean centered data and normed data! So we store these here
    """

    ### Which row are we normalizing? ###
    row_idx = tl.program_id(0)

    ### Map ptrs to correct dtype ###
    if dtype_flag == 0:  # float32
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32))
        inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(tl.float32))
        x_hat_ptr = tl.cast(x_hat_ptr, tl.pointer_type(tl.float32))
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float32))
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(tl.float32))
        beta_ptr = tl.cast(beta_ptr, tl.pointer_type(tl.float32))
    elif dtype_flag == 1:  # float16
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float16))
        inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(tl.float16))
        x_hat_ptr = tl.cast(x_hat_ptr, tl.pointer_type(tl.float16))
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float16))
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(tl.float16))
        beta_ptr = tl.cast(beta_ptr, tl.pointer_type(tl.float16))

    ### Get the start idx of data we want to norm (remember in memory its one long flat vector) ###
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    ### Get offsets for the full block ###
    col_offsets = tl.arange(0,BLOCK_SIZE)

    ### Mask for invalid regions of block ###
    mask = col_offsets < n_cols
    
    ### Get All Indexes ###
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_ptr + col_offsets
    beta_ptrs = beta_ptr + col_offsets

    ### Load Row and Gamma and Beta ###
    row = tl.load(input_ptrs, mask=mask, other=0.) # Invalid row values can just be 0
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.) # We multiply by gamma, so 0 invalid is fine has no effect
    betas = tl.load(beta_ptrs, mask=mask, other=0.) # We add betas so 0 has no effect 

    ### Compute row mean and var w/ reduction ops ###
    row_mean = tl.sum(row, axis=0) / n_cols

    ### Subtract mean from row where mask is valid, otherwise just 0 ###
    row_mean_centered = tl.where(mask, row-row_mean, 0.)
    
    ### Compute variance (E((x-mu)**2))
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = 1. / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    ### Compute final output ###
    output = normed * gammas + betas

    ### Write outputs ###
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

    # store x_hat (the normalized input row)
    x_hat_row_start_ptr = x_hat_ptr + row_idx * x_hat_row_stride
    x_hat_ptrs = x_hat_row_start_ptr + col_offsets
    tl.store(x_hat_ptrs, normed, mask=mask)

    # store inv_var (scalar for this row)
    inv_var_ptrs = inv_var_ptr + row_idx
    tl.store(inv_var_ptrs, inv_var)

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def layernorm_kernel_forward_inference(
    output_ptr, 
    input_ptr, 
    gamma_ptr,  # 1D vector shared across all samples (E, )
    beta_ptr,   # 1D vector shared across all samples (E, )
    input_row_stride, 
    output_row_stride, 
    dtype_flag: tl.constexpr, # Flag for if our data is float32 or float16
    eps: tl.constexpr,
    n_cols: tl.constexpr, # Dimensionality of our embeddings
    BLOCK_SIZE: tl.constexpr # closest power of 2 to our dim of embeddings 
):

    """
    Identical to training, just dont need to store extra things like inv_var and x_hat
    """

    ### Which row are we normalizing? ###
    row_idx = tl.program_id(0)

    ### Map ptrs to correct dtype ###
    if dtype_flag == 0:  # float32
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32))
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float32))
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(tl.float32))
        beta_ptr = tl.cast(beta_ptr, tl.pointer_type(tl.float32))
    elif dtype_flag == 1:  # float16
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float16))
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float16))
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(tl.float16))
        beta_ptr = tl.cast(beta_ptr, tl.pointer_type(tl.float16))

    ### Get the start idx of data we want to norm (remember in memory its one long flat vector) ###
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    ### Get offsets for the full block ###
    col_offsets = tl.arange(0,BLOCK_SIZE)

    ### Mask for invalid regions of block ###
    mask = col_offsets < n_cols
    
    ### Get All Indexes ###
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_ptr + col_offsets
    beta_ptrs = beta_ptr + col_offsets

    ### Load Row and Gamma and Beta ###
    row = tl.load(input_ptrs, mask=mask, other=0.) # Invalid row values can just be 0
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.) # We multiply by gamma, so 0 invalid is fine has no effect
    betas = tl.load(beta_ptrs, mask=mask, other=0.) # We add betas so 0 has no effect 

    ### Compute row mean and var w/ reduction ops ###
    row_mean = tl.sum(row, axis=0) / n_cols

    ### Subtract mean from row where mask is valid, otherwise just 0 ###
    row_mean_centered = tl.where(mask, row-row_mean, 0.)
    
    ### Compute variance (E((x-mu)**2))
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = 1. / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    ### Compute final output ###
    output = normed * gammas + betas

    ### Write outputs ###
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def layernorm_kernel_forward_training_no_bias(
    output_ptr, 
    inv_var_ptr, # Need for Backward Pass (N,)
    x_hat_ptr, # Need for Backward Pass (N x E)
    input_ptr, 
    gamma_ptr,  # 1D vector shared across all samples (E, )
    input_row_stride, 
    output_row_stride, 
    x_hat_row_stride, 
    dtype_flag: tl.constexpr, # Flag for if our data is float32 or float16
    eps: tl.constexpr,
    n_cols: tl.constexpr, # Dimensionality of our embeddings
    BLOCK_SIZE: tl.constexpr # closest power of 2 to our dim of embeddings 
):

    """
    Same as layernorm_kernel_forward_training, just without a bias
    """

    ### Which row are we normalizing? ###
    row_idx = tl.program_id(0)

    ### Map ptrs to correct dtype ###
    if dtype_flag == 0:  # float32
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32))
        inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(tl.float32))
        x_hat_ptr = tl.cast(x_hat_ptr, tl.pointer_type(tl.float32))
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float32))
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(tl.float32))
    elif dtype_flag == 1:  # float16
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float16))
        inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(tl.float16))
        x_hat_ptr = tl.cast(x_hat_ptr, tl.pointer_type(tl.float16))
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float16))
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(tl.float16))

    ### Get the start idx of data we want to norm (remember in memory its one long flat vector) ###
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    ### Get offsets for the full block ###
    col_offsets = tl.arange(0,BLOCK_SIZE)

    ### Mask for invalid regions of block ###
    mask = col_offsets < n_cols
    
    ### Get All Indexes ###
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_ptr + col_offsets

    ### Load Row and Gamma and Beta ###
    row = tl.load(input_ptrs, mask=mask, other=0.) # Invalid row values can just be 0
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.) # We multiply by gamma, so 0 invalid is fine has no effect

    ### Compute row mean and var w/ reduction ops ###
    row_mean = tl.sum(row, axis=0) / n_cols

    ### Subtract mean from row where mask is valid, otherwise just 0 ###
    row_mean_centered = tl.where(mask, row-row_mean, 0.)
    
    ### Compute variance (E((x-mu)**2))
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = 1. / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    ### Compute final output ###
    output = normed * gammas

    ### Write outputs ###
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

    # store x_hat (the normalized input row)
    x_hat_row_start_ptr = x_hat_ptr + row_idx * x_hat_row_stride
    x_hat_ptrs = x_hat_row_start_ptr + col_offsets
    tl.store(x_hat_ptrs, normed, mask=mask)

    # store inv_var (scalar for this row)
    inv_var_ptrs = inv_var_ptr + row_idx
    tl.store(inv_var_ptrs, inv_var)


@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def layernorm_kernel_forward_inference_no_bias(
    output_ptr, 
    input_ptr, 
    gamma_ptr,  # 1D vector shared across all samples (E, )
    input_row_stride, 
    output_row_stride, 
    dtype_flag: tl.constexpr, # Flag for if our data is float32 or float16
    eps: tl.constexpr,
    n_cols: tl.constexpr, # Dimensionality of our embeddings
    BLOCK_SIZE: tl.constexpr # closest power of 2 to our dim of embeddings 
):

    """
    Identical to training no_bias, just dont need to store extra things like inv_var and x_hat
    """

    ### Which row are we normalizing? ###
    row_idx = tl.program_id(0)

    ### Map ptrs to correct dtype ###
    if dtype_flag == 0:  # float32
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32))
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float32))
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(tl.float32))
    elif dtype_flag == 1:  # float16
        output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float16))
        input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float16))
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(tl.float16))

    ### Get the start idx of data we want to norm (remember in memory its one long flat vector) ###
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    ### Get offsets for the full block ###
    col_offsets = tl.arange(0,BLOCK_SIZE)

    ### Mask for invalid regions of block ###
    mask = col_offsets < n_cols
    
    ### Get All Indexes ###
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_ptr + col_offsets

    ### Load Row and Gamma and Beta ###
    row = tl.load(input_ptrs, mask=mask, other=0.) # Invalid row values can just be 0
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.) # We multiply by gamma, so 0 invalid is fine has no effect

    ### Compute row mean and var w/ reduction ops ###
    row_mean = tl.sum(row, axis=0) / n_cols

    ### Subtract mean from row where mask is valid, otherwise just 0 ###
    row_mean_centered = tl.where(mask, row-row_mean, 0.)
    
    ### Compute variance (E((x-mu)**2))
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = 1. / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    ### Compute final output ###
    output = normed * gammas

    ### Write outputs ###
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"]*args["ROW_BLOCK_SIZE"])})
@triton.jit
def layernorm_gamma_kernel_backward(
    dgamma_ptr, # (N//row_tile_size, E)
    norm_ptr,   # (N x E)
    dy_ptr,     # (N x E)
    norm_stride, 
    dy_stride, 
    d_gamma_row_stride, 
    dtype_flag: tl.constexpr,
    n_rows: tl.constexpr, 
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,     ### Number of columns in our tile
    ROW_BLOCK_SIZE: tl.constexpr  ### Number of rows in our tile
):
    
    """
    Derivative w.r.t gamma in layernorm is just:

    # y = x_hat * gamma + beta
    # dL/dgamma = dL/dy * dy/dgamma = sum(grad_output * x_hat) where sum is over batch dim
    # sum up grads over the batch dim

    dL/dy (grad_output) will be in the shape of (* x embed_dim)
    x_hat will be in the shape of (* x embed_dim)
    
    So this is basically just an element wise multiplication then (haramard product). But its a little
    more complicated than that. Here is the issue. Lets say our batch is something reasonable:

    Batch: (32 x 512 x 768) -> Batch x Seq Len x Embed Dim

    Now each of these vectors are normalized, and we flatted our dimensions to do this so we now have
 
    Batch: (32*512 x 768) -> (16384 x 768)

    And that means both our grad_output and x_hat are in this shape! So we have two matricies in the shape
    and we want to do an element wise multiplication, and then a sum ALONG THE BATCH DIM!

    sum((16384 x 768) * (16384 x 768))

    Now if we do this the naive way, we want to reduce along the batch dim so we can set up our blocks
    to each process this dimension. Except, there is no way to easily parallelize that. If we set our 
    BLOCK_SIZE to be 16384, but the max threads we can launch at a time is 1024 (32 warps and 32 threads per warp)
    triton would have to internally for loop essentially to fit that whole block. There must be a better way!

    Yes! Tiling!

    Lets take a look at this with a simple example. Lets pretend we have a "giant" 8 x 6 matrix. And doing a
    reduction over the full 8 dimension is too large. What if we instead focused on doing our matmul and sum on 
    small tiles of our data. This has two benefits. The GPU can be well used because each threadblock can do 
    its process on its small area and do its reduction. At the end we will have a much smaller array to accumulate. 
    
    Lets say we have a 2 x 2 tile. Then on this matrix:

    [x00 x01 x02 x03 x04 x05]       [y00 y01 y02 y03 y04 y05]
    [x10 x11 x12 x13 x14 x15]       [y10 y11 y12 y13 y14 y15]
    [x20 x21 x22 x23 x24 x25]       [y20 y21 y22 y23 y24 y25]
    [x30 x31 x32 x33 x34 x35]       [y30 y31 y32 y33 y34 y35]
    [x40 x41 x42 x43 x44 x45]   x   [y40 y41 y42 y43 y44 y45]
    [x50 x51 x52 x53 x54 x55]       [y50 y51 y52 y53 y54 y55]
    [x60 x61 x62 x63 x64 x65]       [y60 y61 y62 y63 y64 y65]
    [x70 x71 x72 x73 x74 x75]       [y70 y71 y72 y73 y74 y75]

    We can do for our first tile:

    [x00 x01]   x   [y00 y01]   =   [x10*y00 x01*y01]
    [x10 x11]       [y10 y11]       [x10*y10 x11*y11]

    And then we sum along the first dimension:

    [x10*y00+x10*y10 x01*y01+x11*y11] 

    This has then done two things. We have simultaneously done our 
    product and done some reduction along the batch dimension! We repeat
    this for every tile. And we will get a final matrix like:

    [a00 a01 a02 a03 a04 a05]       
    [a10 a11 a12 a13 a14 a15]       
    [a20 a21 a22 a23 a24 a25]       
    [a30 a31 a32 a33 a34 a35]      

    Where we still have the same number of columns but we have halved the 
    number of rows. Then we can do a final sum at the end along the rows
    and its a much smaller matrix to do this on!

    [s00 s01 s02 s03 s04 s05]


    TLDR: Do our hadamard product on each tile, and while we are there anyway 
    (as its loaded in memory) might as well do some of the sum we need right 
    there. How big of a tile you pick is totally a parameter that needs to be 
    tuned!
  
    """

    ### We now have 2 indexes (rows index and col idx) that define the top left ###
    ### corner of our tiles. we can use that to get the remaining indexes ###
    ### These indexes will be strided. For example, if our matrix is 8 x 6 ### 
    ### and the tiles we are using are 2 x 2. Then we will have 4 tiles in each dim ###

    ### (0, 0) represents (0, 0)
    ### (0, 1) represents (0, 2) -> 1 shift in our col_idx is a shift of 2 in our tile col index
    ### (0, 3) represents (0, 6) -> 2 shift in our col_idx is a 4 shift in our tile col index
    ### (1, 0) represents (2, 0) -> 1 shift in our row index is a 2 shift in our tile row index 

    ### So just imagine that we have spread our indexes around to represent the top left corner of
    ### every tile in our data (of course tiles are not overlapping)
    col_idx = tl.program_id(0)
    row_idx = tl.program_id(1)

    ### DTYPE MAP ###
    if dtype_flag == 0:  # float32
        dgamma_ptr = tl.cast(dgamma_ptr, tl.pointer_type(tl.float32))
        norm_ptr = tl.cast(norm_ptr, tl.pointer_type(tl.float32))
        dy_ptr = tl.cast(dy_ptr, tl.pointer_type(tl.float32))
    elif dtype_flag == 1:  # float16
        dgamma_ptr = tl.cast(dgamma_ptr, tl.pointer_type(tl.float16))
        norm_ptr = tl.cast(norm_ptr, tl.pointer_type(tl.float16))
        dy_ptr = tl.cast(dy_ptr, tl.pointer_type(tl.float16))

    ### Lets just pretend we are at the top left. This means index (0,0) for row/col idx ###
    ### Lets first get our offsets (our tile size in each dimension) ###
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = tl.arange(0, ROW_BLOCK_SIZE)

    ### Now apply these offsets to our starting col and row index ###
    ### Remember, we need to multiply our row and col idx by the tile size in each ###
    ### Dimension to get the actual dimension in the data ###
    ### And then add offsets to get all indexes

    ### In our example, if our tile is of size 2x2 and lets say we had 
    ### col_idx = 1, row_idx = 2, that means in the matrix we are really ###
    ### at col_idx = 1 * 2 = 2 and row_idx = 2 * 2 = 4.

    ### So the topleft corner of this tile is at row=4, col=2

    # [x00 x01 x02 x03 x04 x05]      
    # [x10 x11 x12 x13 x14 x15]   
    # [x20 x21 x22 x23 x24 x25]     
    # [x30 x31 x32 x33 x34 x35]    
    # [x40 x41 x42 x43 x44 x45]     
    # [x50 x51 x52 x53 x54 x55]   
    # [x60 x61 x62 x63 x64 x65]
    # [x70 x71 x72 x73 x74 x75]       

    # So the top left of our tile is at x42. And of course we want the indexes of our 
    # whole tile, so we add our col and row offsets giving us the row_range of 4 + [0,1] = 4,5
    # and the col_range of 2 + [0,1] = 2,3 

    # So this tile covers the entries:

    # [(4,2), (4,3),
    # (5,2), (5,3)]

    col_range = col_idx * BLOCK_SIZE + col_offsets
    row_range = row_idx * ROW_BLOCK_SIZE + row_offsets

    ### Of course your number of rows/cols may not be divisible by our tile, so we need to ###
    ### create a mask for invalid spots.  (like what if we had 9 columns, but tile of size 2)

    ### First check columns inside our bounds ###
    col_mask = col_range < n_cols # Vector of [True, True] (or False or whatever it is)

    ### Next our overall mask is going to be where both our row range AND our col_range are inside ###
    row_mask = row_range < n_rows # Vector of [True, True] (or False or whatever it is)

    ### LEts say our col_mask was [True, False] and our row_mask was [True, True]. This means we are ###
    ### Valid along the rows, but the second column is out of bounds. More specifically:

    ### [A, B]
    ### [C, D]

    ### B and D are out of bounds (spilled over the right edge of our matrix)
    ### A and C are fine though. So we want a mask like:

    ### [T, F]
    ### [T, F]

    ### We can easily make this w/ broadcasting
    ### col_mask -> [True False] -> (2,) dim vector. Lets make it 1 x 2:
    ### [True False].unsqueeze(0) -> [[True, True]] -> 1 x 2

    ### Similarly, row_mask -> [True True] -> (2, ) dim vector. Lets make it (2 x 1):
    ### [True True].unsqueeze(-1) -> [[True]
    #                                 [True]] -> 2 x 1

    ### And now we do a comparison:
    # [[True]    & [[True False]] compares a (2 x 1) with a (1 x 2), which with broadcasting we get a final (2,2) matrix
    #  [True]]

    # [True & True True & False] -> [True False]
    # [True & True True & False]    [True False]

    ### Which is exactly the mask we wanted! Now triton doesnt have an unsqueeze but we can easily add a dimension with None
    mask = row_mask[:, None] & col_mask[None, :]

    ### Now lets get our pointers to the actual data
    ### Remember our data is a matrix, but internally its a vector.  
    ### We may have an 8x6 matrix, but its really a 48 length vector, thus we get to do pointer arithmetic again!
    

    # [x00 x01 x02 x03 x04 x05]      
    # [x10 x11 x12 x13 x14 x15]   
    # [x20 x21 x22 x23 x24 x25]     
    # [x30 x31 x32 x33 x34 x35]    
    # [x40 x41 x42 x43 x44 x45]     
    # [x50 x51 x52 x53 x54 x55]   
    # [x60 x61 x62 x63 x64 x65]
    # [x70 x71 x72 x73 x74 x75]   

    ## Notice something nice though, to get the index of x_23 for example, all we need to do is 
    ## ptr(x00) + 2 * (# of elements per row) + 3 = 2 * (stride) + 3

    ### In our earlier example we want elements x43, x44, x53, x54 (as that is the tile)
    ### So in the same way:

    # x43 = ptr(x00) + 4 * (stride) + 3
    # x44 = ptr(x00) + 4 * (stride) + 4
    # x53 = ptr(x00) + 5 * (stride) + 3
    # x54 = ptr(x00) + 5 * (stride) + 4

    ### And just like before we can do this with broadcasting
    ### row_range = 4,5
    ### col_range = 3,4

    ### ptr(x00) += [[4], * (stride) + [[3,4]]
    #                [5]]

    ### Which gives with broadcasting a 2d tile of indexes like:

    ### [ptr(x00) + 4 * (stride) + 3    ptr(x00) + 4 * (stride) + 4]
    ### [ptr(x00) + 5 * (stride) + 3    ptr(x00) + 5 * (stride) + 4]
    
    ### Which is exactly what we want!

    dy_ptr += row_range[:, None] * dy_stride + col_range[None, :]
    norm_ptr += row_range[:, None] * norm_stride + col_range[None, :]

    ### Now actually get the data ! ###
    dy = tl.load(dy_ptr, mask=mask, other=0.)
    norm = tl.load(norm_ptr, mask=mask, other=0.)

    ### Multiply and then sum the tile across the rows to do a tile level reduction! ####
    dgamma = tl.sum(dy * norm, axis=0)

    ### Get pointer and store this ###
    ### In our output, our row  we save in is the actual row_idx of our kernel, so NO NEED to 
    ### do that offset like we did earlier (row_range = row_idx * ROW_BLOCK_SIZE + row_offsets)
    ### our dgamma is already presized, and the dgamma_row_stride tells me how much to move over
    ### on the other hand we need to offset to the correct column
    dgamma_offsets = row_idx * d_gamma_row_stride + col_range

    # We are storing a row. The only invalid thing can be along the col
    tl.store(dgamma_ptr + dgamma_offsets, dgamma, mask=col_mask)

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def layernorm_kernel_backward(
    dx_ptr,              # (N, E) output gradient wrt input
    dx_hat_ptr,          # (N, E) upstream gradient (dL/dy * dy/dx_hat)
    x_hat_ptr,           # (N, E) normalized inputs from forward
    inv_var_ptr,         # (N,) per-row inverse std
    dx_row_stride,
    dx_hat_row_stride,
    x_hat_row_stride,
    dtype_flag: tl.constexpr, 
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    
    """
    We need to compute the gradients w.r.t to the input x. This formula is as follows:

    y = x_hat * gamma + beta
    where x_hat = (x - mu) / (var + eps)
    dL/dx = dL/dy * dy/dx_hat * dx_hat / dx

    = 1/(var + eps) * [dL/dy * gamma - mean(dL/dy*gamma) - x_hat * mean(dL/dy * gamma * x_hat)]
    = 1/(var + eps) * [dx_hat - mean(dx_hat) - x_hat * mean(dx_hat* x_hat)]
    = 1/(var + eps) * (1 / N) * (N*dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
    sum up grads over the batch dim



    This is pretty simple as we are just doing this op per vector, so lets just set our Block size to cover that full embed dim. But
    as you can see we need some stuff from our forward pass like x_hat and 1/(var + eps)
    """

    row_idx = tl.program_id(0)

    ### Map Pointers To Correct Dtype ###
    if dtype_flag == 0:  # float32
        dx_ptr = tl.cast(dx_ptr, tl.pointer_type(tl.float32))
        dx_hat_ptr = tl.cast(dx_hat_ptr, tl.pointer_type(tl.float32))
        x_hat_ptr = tl.cast(x_hat_ptr, tl.pointer_type(tl.float32))
        inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(tl.float32))
    elif dtype_flag == 1:  # float16
        dx_ptr = tl.cast(dx_ptr, tl.pointer_type(tl.float16))
        dx_hat_ptr = tl.cast(dx_hat_ptr, tl.pointer_type(tl.float16))
        x_hat_ptr = tl.cast(x_hat_ptr, tl.pointer_type(tl.float16))
        inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(tl.float16))
    
    ### Get correct row of upstream grad and normalized data x_hat ###
    dx_hat_row_start_ptr = dx_hat_ptr + row_idx * dx_hat_row_stride
    x_hat_row_start_ptr = x_hat_ptr + row_idx * x_hat_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    dx_hat_ptrs = dx_hat_row_start_ptr + col_offsets
    x_hat_ptrs = x_hat_row_start_ptr + col_offsets

    ### Mask out invalid position ###
    mask = col_offsets < n_cols
    
    ### Grab Data ###
    dx_hat = tl.load(dx_hat_ptrs, mask=mask, other=0.)
    x_hat = tl.load(x_hat_ptrs, mask=mask, other=0.)
    inv_var = tl.load(inv_var_ptr + row_idx)  

    # core backward formula
    # dX = (1/n) * inv_var * ( n*dxhat - sum(dxhat) - xhat * sum(dxhat*x_hat))
    sum_dxhat = tl.sum(dx_hat, axis=0)
    sum_dxhat_xhat = tl.sum(dx_hat * x_hat, axis=0)
    dx = (1.0 / n_cols) * inv_var * (n_cols * dx_hat - sum_dxhat - x_hat * sum_dxhat_xhat)

    ### Store it ###
    dx_row_start_ptr = dx_ptr + row_idx * dx_row_stride
    dx_ptrs = dx_row_start_ptr + col_offsets
    tl.store(dx_ptrs, dx, mask=mask)

def fused_layernorm_forward(x, gamma, beta, eps=1e-5, training=True, use_dlpack=True):

    """
    x: Input (N, E)
    gamma: scale parameter (E,)
    beta: shift parameter (E,)

    Returns:
        y: Ouptut (N, E)
        x_hat: Normed x before shift/scale (N,E)
        inv_var: Inverse variance per sample in batch (N,)
    """
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if not DLPACK_DISABLE and use_dlpack:

        x = torch.utils.dlpack.from_dlpack(x)
        gamma = torch.utils.dlpack.from_dlpack(gamma)
        if beta is not None:
            beta = torch.utils.dlpack.from_dlpack(beta)

        # Allocate outputs
        y = torch.empty_like(x)
        x_hat = torch.empty_like(x)
        inv_var = torch.empty(n_rows, dtype=x.dtype, device=x.device)
        
        # Compute strides in elements for each array
        x_row_stride = x.stride(0)
        y_row_stride = y.stride(0)
        x_hat_row_stride = x_hat.stride(0)
        
        # Map dtype to Triton flag
        dtype_flag = 0 if x.dtype == torch.float32 else 1  # 0=float32, 1=float16
        
        ### Set Grid ###
        grid = (n_rows,)
        
        ### When in training mode
        if training:
            ### if we have a beta 
            if beta is not None:
                layernorm_kernel_forward_training[grid](
                    y,                   # output_ptr
                    inv_var,             # inv_var_ptr
                    x_hat,               # x_hat_ptr
                    x,                   # input_ptr
                    gamma,               # gamma_ptr
                    beta,                # beta_ptr
                    x_row_stride,        # input_row_stride
                    y_row_stride,        # output_row_stride
                    x_hat_row_stride,    # x_hat_row_stride
                    dtype_flag,          # dtype_flag (constexpr)
                    eps,                 # eps (constexpr)
                    n_cols,              # n_cols (constexpr)
                    BLOCK_SIZE           # BLOCK_SIZE (constexpr)
                )
            else:
                layernorm_kernel_forward_training_no_bias[grid](
                    y,                   # output_ptr
                    inv_var,             # inv_var_ptr
                    x_hat,               # x_hat_ptr
                    x,                   # input_ptr
                    gamma,               # gamma_ptr
                    x_row_stride,        # input_row_stride
                    y_row_stride,        # output_row_stride
                    x_hat_row_stride,    # x_hat_row_stride
                    dtype_flag,          # dtype_flag (constexpr)
                    eps,                 # eps (constexpr)
                    n_cols,              # n_cols (constexpr)
                    BLOCK_SIZE           # BLOCK_SIZE (constexpr)
                )
            
            # Convert back to CuPy if needed
            y = cp.from_dlpack(y)
            x_hat = cp.from_dlpack(x_hat)
            inv_var = cp.from_dlpack(inv_var)
            
            return y, x_hat, inv_var
        
        else:
            if beta is not None:
                layernorm_kernel_forward_inference[grid](
                    y,                   # output_ptr
                    inv_var,             # inv_var_ptr
                    x_hat,               # x_hat_ptr
                    x,                   # input_ptr
                    gamma,               # gamma_ptr
                    beta,                # beta_ptr
                    x_row_stride,        # input_row_stride
                    y_row_stride,        # output_row_stride
                    x_hat_row_stride,    # x_hat_row_stride
                    dtype_flag,          # dtype_flag (constexpr)
                    eps,                 # eps (constexpr)
                    n_cols,              # n_cols (constexpr)
                    BLOCK_SIZE           # BLOCK_SIZE (constexpr)
                )
            else:
                layernorm_kernel_forward_inference_no_bias[grid](
                    y,                   # output_ptr
                    inv_var,             # inv_var_ptr
                    x_hat,               # x_hat_ptr
                    x,                   # input_ptr
                    gamma,               # gamma_ptr
                    x_row_stride,        # input_row_stride
                    y_row_stride,        # output_row_stride
                    x_hat_row_stride,    # x_hat_row_stride
                    dtype_flag,          # dtype_flag (constexpr)
                    eps,                 # eps (constexpr)
                    n_cols,              # n_cols (constexpr)
                    BLOCK_SIZE           # BLOCK_SIZE (constexpr)
                )
            
            # Convert back to CuPy if needed
            y = cp.from_dlpack(y)
            return y

    else:
        # Allocate outputs
        y = cp.empty_like(x)
        x_hat = cp.empty_like(x)
        inv_var = cp.empty((n_rows,), dtype=x.dtype)

        # Compute strides in elements for each array
        x_row_stride = x.strides[0] // x.itemsize
        y_row_stride = y.strides[0] // y.itemsize
        x_hat_row_stride = x_hat.strides[0] // x_hat.itemsize

        # Map dtype to Triton flag
        dtype_flag = 0 if x.dtype == cp.float32 else 1  # 0=float32, 1=float16

        ### Set Grid ###
        grid = (n_rows,)

        ### When in training mode
        if training:
            ### if we have a beta 
            if beta is not None:
                layernorm_kernel_forward_training[grid](
                    y.data.ptr,          # output_ptr
                    inv_var.data.ptr,    # inv_var_ptr
                    x_hat.data.ptr,      # x_hat_ptr
                    x.data.ptr,          # input_ptr
                    gamma.data.ptr,      # gamma_ptr
                    beta.data.ptr,       # beta_ptr
                    x_row_stride,        # input_row_stride
                    y_row_stride,        # output_row_stride
                    x_hat_row_stride,    # x_hat_row_stride
                    dtype_flag,          # dtype_flag (constexpr)
                    eps,                 # eps (constexpr)
                    n_cols,              # n_cols (constexpr)
                    BLOCK_SIZE           # BLOCK_SIZE (constexpr)
                )

            else:
                layernorm_kernel_forward_training_no_bias[grid](
                    y.data.ptr,          # output_ptr
                    inv_var.data.ptr,    # inv_var_ptr
                    x_hat.data.ptr,      # x_hat_ptr
                    x.data.ptr,          # input_ptr
                    gamma.data.ptr,      # gamma_ptr
                    x_row_stride,        # input_row_stride
                    y_row_stride,        # output_row_stride
                    x_hat_row_stride,    # x_hat_row_stride
                    dtype_flag,          # dtype_flag (constexpr)
                    eps,                 # eps (constexpr)
                    n_cols,              # n_cols (constexpr)
                    BLOCK_SIZE           # BLOCK_SIZE (constexpr)
                )

            return y, x_hat, inv_var
        
        else:
            if beta is not None:
                layernorm_kernel_forward_inference[grid](
                    y.data.ptr,          # output_ptr
                    inv_var.data.ptr,    # inv_var_ptr
                    x_hat.data.ptr,      # x_hat_ptr
                    x.data.ptr,          # input_ptr
                    gamma.data.ptr,      # gamma_ptr
                    beta.data.ptr,       # beta_ptr
                    x_row_stride,        # input_row_stride
                    y_row_stride,        # output_row_stride
                    x_hat_row_stride,    # x_hat_row_stride
                    dtype_flag,          # dtype_flag (constexpr)
                    eps,                 # eps (constexpr)
                    n_cols,              # n_cols (constexpr)
                    BLOCK_SIZE           # BLOCK_SIZE (constexpr)
                )

            else:
                layernorm_kernel_forward_inference_no_bias[grid](
                    y.data.ptr,          # output_ptr
                    inv_var.data.ptr,    # inv_var_ptr
                    x_hat.data.ptr,      # x_hat_ptr
                    x.data.ptr,          # input_ptr
                    gamma.data.ptr,      # gamma_ptr
                    x_row_stride,        # input_row_stride
                    y_row_stride,        # output_row_stride
                    x_hat_row_stride,    # x_hat_row_stride
                    dtype_flag,          # dtype_flag (constexpr)
                    eps,                 # eps (constexpr)
                    n_cols,              # n_cols (constexpr)
                    BLOCK_SIZE           # BLOCK_SIZE (constexpr)
                )

            return y

def fused_layernorm_backward(x_hat, inv_var, dy, gamma, bias=True, use_dlpack=True):

    """
    x_hat: normalized input from forward (N, E)
    inv_var: 1/sqrt(var + eps) per row (N,)
    dy: upstream gradient (N, E)
    gamma: scale parameter (E,)
    
    Returns:
        dx: gradient w.r.t input (N, E)
        dgamma: gradient w.r.t gamma (E,)
        dbeta: gradient w.r.t beta (E,)
    """

    ### Lets just hardcode our tile size with something reasonable ###
    ### We should technically autotune this, but our output size for dgamma ###
    ### and our grid size change depending on this. We can solve this with an ###
    ### atomic add, but i dont want to make this more complicated than it needs to be! ###
    ### We want to give our model some teeth, not fangs! ###
    GAMMA_BLOCK_SIZE = 64
    GAMMA_ROW_BLOCK_SIZE = 64
    n_rows, n_cols = dy.shape

    if not DLPACK_DISABLE and use_dlpack:

        x_hat = torch.utils.dlpack.from_dlpack(x_hat)
        inv_var = torch.utils.dlpack.from_dlpack(inv_var)
        dy = torch.utils.dlpack.from_dlpack(dy)
        gamma = torch.utils.dlpack.from_dlpack(gamma)
    
        ### Lets just hardcode our tile size with something reasonable ###
        GAMMA_BLOCK_SIZE = 64
        GAMMA_ROW_BLOCK_SIZE = 64
        n_rows, n_cols = dy.shape
        
        ### COMPUTE DX ###
        dx = torch.empty_like(dy)
        dx_row_stride = dx.stride(0)
        dy_row_stride = dy.stride(0)
        x_hat_row_stride = x_hat.stride(0)
        dtype_flag = 0 if dy.dtype == torch.float32 else 1
        dx_BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)
        
        layernorm_kernel_backward[grid](
            dx,
            dy * gamma,  # dx_hat = dy * gamma
            x_hat,
            inv_var,
            dx_row_stride,
            dy_row_stride,
            x_hat_row_stride,
            dtype_flag,
            n_cols,
            dx_BLOCK_SIZE
        )
        
        ### COMPUTE DGAMMA ###
        num_col_programs = triton.cdiv(n_cols, GAMMA_BLOCK_SIZE)
        num_row_programs = triton.cdiv(n_rows, GAMMA_ROW_BLOCK_SIZE)
        dgamma = torch.empty(num_row_programs, n_cols, dtype=dy.dtype, device=dy.device)
        grid = (num_col_programs, num_row_programs)
        
        layernorm_gamma_kernel_backward[grid](
            dgamma,
            x_hat,
            dy,  # for dgamma, upstream is dy*gamma
            x_hat.stride(0),
            dy.stride(0),
            dgamma.stride(0),
            dtype_flag,
            n_rows,
            n_cols,
            GAMMA_BLOCK_SIZE,
            GAMMA_ROW_BLOCK_SIZE, 
        )

        ### Convert back to cupy ###
        dx = cp.from_dlpack(dx)
        dgamma = cp.from_dlpack(dgamma)
        dy = cp.from_dlpack(dy)
        
    else:

        ### COMPUTE DX ###
        dx = cp.empty_like(dy)
        dx_row_stride = dx.strides[0] // dx.itemsize
        dy_row_stride = dy.strides[0] // dy.itemsize
        x_hat_row_stride = x_hat.strides[0] // x_hat.itemsize
        dtype_flag = 0 if dy.dtype == cp.float32 else 1
        dx_BLOCK_SIZE = triton.next_power_of_2(n_cols)

        grid = (n_rows,)
        layernorm_kernel_backward[grid](
            dx.data.ptr,
            (dy * gamma).data.ptr,  # dx_hat = dy * gamma
            x_hat.data.ptr,
            inv_var.data.ptr,
            dx_row_stride,
            dy_row_stride,
            x_hat_row_stride,
            dtype_flag,
            n_cols,
            dx_BLOCK_SIZE
        )

        ### COMPUTE DGAMMA ###
        num_col_programs = triton.cdiv(n_cols, GAMMA_BLOCK_SIZE)
        num_row_programs = triton.cdiv(n_rows, GAMMA_ROW_BLOCK_SIZE)
        dgamma = cp.empty((num_row_programs, n_cols), dtype=dy.dtype)

        grid = (num_col_programs, num_row_programs)
        layernorm_gamma_kernel_backward[grid](
            dgamma.data.ptr,
            x_hat.data.ptr,
            dy.data.ptr,  # for dgamma, upstream is dy*gamma
            x_hat.strides[0] // x_hat.itemsize,
            dy.strides[0] // dy.itemsize,
            dgamma.strides[0] // dgamma.itemsize,
            dtype_flag,
            n_rows,
            n_cols,
            GAMMA_BLOCK_SIZE,
            GAMMA_ROW_BLOCK_SIZE, 
        )

    ### Sum up all contributions to Gamma ###
    dgamma = dgamma.sum(axis=0)

    ### COMPUTE DBETA ###
    if bias:
        dbeta = dy.sum(axis=0)
        return dx, dgamma, dbeta
    else:
        return dx, dgamma