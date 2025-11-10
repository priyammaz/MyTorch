"""
Groupnorm is nearly identical to LayerNorm! But instead of computing the 
normalization across an embedding dimension, we will chunk it up. 

Typically we apply groupnorm to vision problems, so we will have
the expected shape to be [B x C x *], and we will apply the 
grouping to the C dimension!

Full code credit got to the awesome set of LigerKernels!
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/group_norm.py
"""

import cupy as cp
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.math import rsqrt
from utils import calc_num_warps

def groupnorm_naive(x, weight, bias=None, num_groups=32, eps=1e-5):
    """
    Naive GroupNorm implementation for testing
    x: (N, C, *spatial) 
    weight: (C,)
    bias: (C,)
    """
    N, C = x.shape[0], x.shape[1]
    spatial_shape = x.shape[2:]
    G = num_groups
    
    # Reshape to (N, G, C//G, *spatial)
    x_reshaped = x.reshape(N, G, C // G, *spatial_shape)
    
    # Compute mean and var across (C//G, *spatial) dims
    norm_axes = tuple(range(2, x_reshaped.ndim))
    
    mean = x_reshaped.mean(axis=norm_axes, keepdims=True)
    var = x_reshaped.var(axis=norm_axes, keepdims=True)
    
    # Normalize
    normed = (x_reshaped - mean) / cp.sqrt(var + eps)
    
    # Reshape back to (N, C, *spatial)
    normed = normed.reshape(N, C, *spatial_shape)
    
    # Apply affine transform (per-channel)
    weight_shape = [1, C] + [1] * len(spatial_shape)
    out = normed * weight.reshape(*weight_shape)
    
    if bias is not None:
        out = out + bias.reshape(*weight_shape)
    
    return out

@triton.jit
def groupnorm_kernel_forward_training(
    Y_ptr,  # pointer to output, shape (n_rows, n_groups, hidden_size)
    Y_row_stride,  # stride of each row in output
    Y_col_stride,  # stride of each column in output
    X_ptr,  # pointer to input, shape (n_rows, n_groups, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_row_stride,  # stride of each row in mean
    Mean_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    RSTD_row_stride,  # stride of each row in rstd
    RSTD_col_stride,  # stride of each column in rstd
    W_ptr,  # pointer to W
    B_ptr,  # pointer to B
    hidden_size,  # hidden size of X
    channels_per_group,  # the number of channels per group
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Our forward kernel will process data in the shape of: B x G x hidden_dimension

    The hidden dimension will be of size (C//G * spatial_dimensions) So this can be 
    relatively large! Thus when computing mean/std we can employ an online algorithm!

    """

    ### Unlike layernorm where each pid went with a single embedding ###
    ### that we wanted to norm, now we have two that refer to the ###
    ### sample and group we want to norm ###
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    ### Move pointers to this batch and group
    X_ptr += batch_idx * X_row_stride + group_idx * X_col_stride
    Y_ptr += batch_idx * Y_row_stride + group_idx * Y_col_stride
    block_range = tl.arange(0, BLOCK_SIZE)

    ### Compute Mean and Variance w/ an online algorithm ###
    ### The issue here is that before in layernorm, our mean/var was just ###
    ### computed over the embedding dim, but here we have a mean/var over the ###
    ### (C//G * spatial_dims) which can be very large! So instead of loading everything ###
    ### to memory, we will compute this dimension in chunks ###

    ### We have sum(xi), sum(xi**2) assuming N elements in that sum
    ### then mean = sum(xi) / N
    ### and var = sum(xi**2)/N - (sum(xi)/N)**2 -> E[X**2] - E[X]**2
    ### https://math.stackexchange.com/questions/2148877/iterative-calculation-of-mean-and-standard-deviation
    s = 0.0
    squared_sum = 0.0
    for i in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_size_offsets = i + block_range
        mask = hidden_size_offsets < hidden_size
        X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=0.0)
        s += tl.sum(X)
        # X**2
        squared_sum += tl.sum(X * X)

    m = s / hidden_size
    variance = (squared_sum / hidden_size) - (m * m)
    rstd = rsqrt(variance + eps)

    ### Now we can normalize ###
    ### Remember that each thread will process a specific slice of channels cooresponding to that group ###
    ### our data (per sample in the batch) looks like -> X shape: [n_groups, hidden_size]
    ### but this hidden_size contains the (C//G * spatial_dimensions) = (channels_per_group * spatial_dimensions)
    ### What I want is what are the number of elements in each channel_per_group
    hidden_size_per_channel = hidden_size // channels_per_group

    ### Now this thread only processes this specific group, so we need to make sure as wel loop through our channel ###
    ### we are only looking at the channel cooresponding to this group ###
    for channel_idx in tl.range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):

        ### Load our W/B that correspond to this channel ###
        W = tl.load(W_ptr + channel_idx)
        B = tl.load(B_ptr + channel_idx)

        ### For all other dimensions (the spatial dims inside the channel), loop through and normalize the data ###
        for i in range(0, hidden_size_per_channel, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size_per_channel
            X = tl.load(X_ptr + hidden_size_offsets, mask=mask, other=m)
            Y = (X - m) * rstd * W + B
            tl.store(Y_ptr + hidden_size_offsets, Y, mask=mask)

        ### Advance our pointer to the next channel ###
        X_ptr += hidden_size_per_channel
        Y_ptr += hidden_size_per_channel

    tl.store(Mean_ptr + batch_idx * Mean_row_stride + group_idx * Mean_col_stride, m)
    tl.store(RSTD_ptr + batch_idx * RSTD_row_stride + group_idx * RSTD_col_stride, rstd)

def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    X = X.view(batch_size, num_groups, -1).contiguous()
    hidden_size = X.shape[-1]
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    Y = torch.empty((batch_size, num_groups, hidden_size), dtype=X.dtype, device=X.device)
    Mean = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)
    RSTD = torch.zeros((batch_size, num_groups), dtype=X.dtype, device=X.device)

    groupnorm_kernel_forward_training[(batch_size, num_groups)](
        Y,
        Y.stride(0),
        Y.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        Mean,
        Mean.stride(0),
        Mean.stride(1),
        RSTD,
        RSTD.stride(0),
        RSTD.stride(1),
        W,
        B,
        hidden_size,
        channels_per_group,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Return tensors in the original shape
    return Y.view(*shape).contiguous(), X.view(*shape).contiguous(), Mean, RSTD, BLOCK_SIZE

@triton.jit
def group_norm_backward(
    X_ptr,  # pointer to input, shape (n_rows, n_channels, hidden_size)
    X_row_stride,  # stride of each row in input
    X_col_stride,  # stride of each column in input
    W_ptr,  # pointer to weights, shape (n_channels)
    Mean_ptr,  # pointer to mean, shape (n_rows, n_groups)
    Mean_ptr_row_stride,  # stride of each column in mean
    Mean_ptr_col_stride,  # stride of each column in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows, n_groups)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_groups, hidden_size)
    DW_ptr,  # pointer to weights grad, shape (n_channels)
    DB_ptr,  # pointer to bias grad, shape (n_channels)
    UPSTREAM_ptr,  # pointer to output grad, shape (n_rows, n_channels, hidden_size)
    hidden_size: tl.constexpr,  # hidden size
    channels_per_group: tl.constexpr,  # number of groups in group norm
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    
    ### Again our backward pass will have every pid process a specific sample and group ###
    batch_idx = tl.program_id(0)
    group_idx = tl.program_id(1)

    ### Advance pointers to the correct batch ###
    X_ptr += batch_idx * X_row_stride
    DX_ptr += batch_idx * X_row_stride
    UPSTREAM_ptr += batch_idx * X_row_stride

    ### Grad Mean/Std for this specific group ###
    mean = tl.load(Mean_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)
    rstd = tl.load(RSTD_ptr + batch_idx * Mean_ptr_row_stride + group_idx * Mean_ptr_col_stride)

    ### We need to sum across the entire group, as gamma and beta contributed to the whole group ###
    ### therefore we need to accumulate up all the gradient contributions from each channel in this group ###
    c1 = 0.0
    c2 = 0.0
    block_range = tl.arange(0, BLOCK_SIZE)

    ### Now we loop through our channels for this group to get all our grad contributions per channel ### 
    for channel_idx in range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):

        ### What is our dW and dB for this channel ###
        dW = 0.0
        dB = 0.0

        ### Grab W ###
        W = tl.load(W_ptr + channel_idx)
        
        ### Loop through our hidden size (the spatial dims) ###
        for i in tl.range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size

            ### Grab our data for this specific channel in this specific group ###
            X = tl.load(
                X_ptr, channel_idx * X_col_stride + hidden_size_offsets, 
                mask=mask, 
                other=0.0
            )

            ### The upstream grad has the same shape, so grab that too ###
            UPSTREAM_grad = tl.load(
                UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets, 
                mask=mask, 
                other=0.0
            )

            ### recompute x_hat ###
            x_hat = (X - mean) / rstd

            ### Compute dW = sum(upstream_grad * x_hat) ###
            dW += tl.sum(UPSTREAM_grad * x_hat)

            ### Compute dB = sum(upstream_grad) ###
            dB += tl.sum(UPSTREAM_grad)

            ### Compute dX ###
            ### Remember that dX = (1/n) * inv_var * (n*dxhat - sum(dxhat) - xhat * sum(dxhat*x_hat))
            ### where dx_hat = dy * gamma
            dx_hat = UPSTREAM_grad * W
            c1 += tl.sum(dx_hat * x_hat)
            c2 += tl.sum(dx_hat)

        ### Now remember, each thread processes a batch and group. This means there are race conditions. ###
        ### we may be processing the same group in two different threads because they are on different samples ###
        ### but each sample contributes to the grad of W and B, so when we do our sum, we need to make sure we do ###
        ### and atomic_add so all other sums have to wait for the one currently running to finish before continuting ###
        tl.atomic_add(DW_ptr + channel_idx, dW.to(dtype))
        tl.atomic_add(DB_ptr + channel_idx, dB.to(dtype))

    ### Now we just need our grad w.r.t the input according to the formula ealier! we have computed ###
    ### our sum(dxhat) and sum(dxhat*x_hat), already accumulated across the channel! ###
    ### We can first make them means ###
    N = hidden_size * channels_per_group
    c1 = c1 / N
    c2 = c2 / N

    ### Now again loop through our group of channels ##
    for channel_idx in tl.range(group_idx * channels_per_group, (group_idx + 1) * channels_per_group):
        
        ### Load the weights ###
        W = tl.load(W_ptr + channel_idx)

        ### Grab X and our upstream grad ###
        for i in range(0, hidden_size, BLOCK_SIZE):
            hidden_size_offsets = i + block_range
            mask = hidden_size_offsets < hidden_size
            
            ### Load X ###
            X = tl.load(
                X_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )

            ### Load Upstream Grad ###
            UPSTREAM_grad = tl.load(
                UPSTREAM_ptr + channel_idx * X_col_stride + hidden_size_offsets,
                mask=mask,
                other=0.0,
            )

            ### Normalize Again ###
            x_hat = (X - mean) * rstd
            wdy = W * UPSTREAM_grad
            dx = (wdy - (x_hat * c1 + c2)) * rstd
            tl.store(DX_ptr + channel_idx * X_col_stride + hidden_size_offsets, dx, mask=mask)
