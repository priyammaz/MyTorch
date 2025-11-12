"""
LayerNorm fused kernel inspired by https://github.com/lucidrains/triton-transformer/blob/main/triton_transformer/layernorm.py
As well as from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py
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
    inv_var_ptr, # Need for Backward Pass (N,) # can be None during inference
    mean_ptr, # Need fo Backward Pass (N,) # Can be None during inference
    input_ptr, 
    gamma_ptr,  # 1D vector shared across all samples (E, ) can be None if no weight
    beta_ptr,   # 1D vector shared across all samples (E, ) can be None if no bias
    input_row_stride, 
    output_row_stride,  
    DTYPE_FLAG: tl.constexpr, # Flag for if our data is float32 or float16
    eps: tl.constexpr,
    n_cols: tl.constexpr, # Dimensionality of our embeddings
    BLOCK_SIZE: tl.constexpr, # closest power of 2 to our dim of embeddings 
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
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    output_ptr = tl.cast(output_ptr, tl.pointer_type(pointer_dtype))
    input_ptr = tl.cast(input_ptr, tl.pointer_type(pointer_dtype))
    if gamma_ptr is not None:
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(pointer_dtype))
    if mean_ptr is not None:
        mean_ptr = tl.cast(mean_ptr, tl.pointer_type(pointer_dtype))
    if inv_var_ptr is not None:
        inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(pointer_dtype))
    if beta_ptr is not None:
        beta_ptr = tl.cast(beta_ptr, tl.pointer_type(pointer_dtype))

    ### Get the start idx of data we want to norm (remember in memory its one long flat vector) ###
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    ### Get offsets for the full block ###
    col_offsets = tl.arange(0,BLOCK_SIZE)

    ### Mask for invalid regions of block ###
    mask = col_offsets < n_cols
    
    ### Get All Indexes ###
    input_ptrs = row_start_ptr + col_offsets

    ### Load Weights if they exist ###
    if gamma_ptr is not None:
        gamma_ptrs = gamma_ptr + col_offsets
        gammas = tl.load(gamma_ptrs, mask=mask, other=0.) # We multiply by gamma, so 0 invalid is fine has no effect

    if beta_ptr is not None:
        beta_ptrs = beta_ptr + col_offsets
        betas = tl.load(beta_ptrs, mask=mask, other=0.)

    ### Load Row and Gamma ###
    row = tl.load(input_ptrs, mask=mask, other=0.) # Invalid row values can just be 0
    
    ### Compute row mean and var w/ reduction ops ###
    row_mean = tl.sum(row, axis=0) / n_cols

    ### Subtract mean from row where mask is valid, otherwise just 0 ###
    row_mean_centered = tl.where(mask, row-row_mean, 0.)
    
    ### Compute variance (E((x-mu)**2))
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = tl.rsqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    ### Compute final output ###
    if gamma_ptr is not None:
        output = normed * gammas
    else:
        output = normed

    if beta_ptr is not None:
        output += betas

    ### Write outputs ###
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

    ### No need to waste time writing outputs if we dont need it! ###
    if mean_ptr is not None and inv_var_ptr is not None:

        # store mean (scalar for this row)
        mean_row_ptrs = mean_ptr + row_idx
        tl.store(mean_row_ptrs, row_mean)

        # store inv_var (scalar for this row)
        inv_var_ptrs = inv_var_ptr + row_idx
        tl.store(inv_var_ptrs, inv_var)

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def layernorm_kernel_backward(
    x_ptr, 
    gamma_ptr, 
    mean_ptr, 
    inv_var_ptr, 
    dX_ptr, 
    dGamma_ptr, 
    dB_ptr, 
    dY_ptr,
    stride_x, 
    stride_dx, 
    stride_dy, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr, 
    DTYPE_FLAG: tl.constexpr
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
    pointer_type = tl.float32 if DTYPE_FLAG == 0 else tl.float16
    x_ptr = tl.cast(x_ptr, tl.pointer_type(pointer_type))
    mean_ptr = tl.cast(mean_ptr, tl.pointer_type(pointer_type))
    inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(pointer_type))
    dX_ptr = tl.cast(dX_ptr, tl.pointer_type(pointer_type))
    dY_ptr = tl.cast(dY_ptr, tl.pointer_type(pointer_type))
    if gamma_ptr is not None:
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(pointer_type))
    if dGamma_ptr is not None:
        dGamma_ptr = tl.cast(dGamma_ptr, tl.pointer_type(tl.float32))   
    if dB_ptr is not None:
        dB_ptr = tl.cast(dB_ptr, tl.pointer_type(tl.float32))   
    
    ### Indexes for this row we are processing ###
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    ### Load Weights ###
    if gamma_ptr is not None:
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.).to(tl.float32)
    else:
        gamma = 1.0

    ### Get all other pointers for this row ###
    row_x_ptr = x_ptr + row_idx * stride_x
    row_dX_ptr = dX_ptr + row_idx * stride_dx
    row_dY_ptr = dY_ptr + row_idx * stride_dy
    row_mean_ptr = mean_ptr + row_idx
    row_inv_var_ptr = inv_var_ptr + row_idx

    ### Load all the data ###
    x_row = tl.load(row_x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy_row = tl.load(row_dY_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    row_mean = tl.load(row_mean_ptr).to(tl.float32)
    row_inv_std = tl.load(row_inv_var_ptr).to(tl.float32)

    ### Compute backward pass ###
    x_hat = (x_row - row_mean) * row_inv_std
    wdy = gamma * dy_row # <- upstream grad (dL/dy)(dy/dx_hat)
    mean1 = tl.sum(x_hat*wdy, axis=0) / n_cols
    mean2 = tl.sum(wdy, axis=0) / n_cols
    dx = row_inv_std * (wdy - (x_hat * mean1 + mean2))

    ### Store dx ###
    tl.store(row_dX_ptr + cols, dx.to(pointer_type), mask=mask)

    ### Accumulate grads for Weights and Biases ###
    if dGamma_ptr is not None:
        dW = dy_row * x_hat
        tl.atomic_add(dGamma_ptr + cols, dW, mask=mask)  # Keep in FP32!

    if dB_ptr is not None:
        db = dy_row
        tl.atomic_add(dB_ptr + cols, db, mask=mask)


def fused_layernorm_forward(x, gamma=None, beta=None, eps=1e-5, training=True, use_dlpack=True):

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
        if gamma is not None:
            gamma = torch.utils.dlpack.from_dlpack(gamma)
        if beta is not None:
            beta = torch.utils.dlpack.from_dlpack(beta)

        # Allocate outputs
        y = torch.empty_like(x)
        inv_var = torch.empty(n_rows, dtype=x.dtype, device=x.device) if training else None
        mean = torch.empty(n_rows, dtype=x.dtype, device=x.device) if training else None
        
        # Compute strides in elements for each array
        x_row_stride = x.stride(0)
        y_row_stride = y.stride(0)
        
        # Map dtype to Triton flag
        dtype_flag = 0 if x.dtype == torch.float32 else 1  # 0=float32, 1=float16
        
        ### Set Grid ###
        grid = (n_rows,)


        layernorm_kernel_forward_training[grid](
            y,                   # output_ptr
            inv_var,             # inv_var_ptr
            mean,                # mean_ptr
            x,                   # input_ptr
            gamma,               # gamma_ptr
            beta,                # beta_ptr
            x_row_stride,        # input_row_stride
            y_row_stride,        # output_row_stride
            dtype_flag,          # dtype_flag (constexpr)
            eps,                 # eps (constexpr)
            n_cols,              # n_cols (constexpr)
            BLOCK_SIZE,          # BLOCK_SIZE (constexpr),
        )
         
        # Convert back to CuPy if needed
        y = cp.from_dlpack(y)
        if training:
            mean = cp.from_dlpack(mean)
            inv_var = cp.from_dlpack(inv_var)
            return y, mean, inv_var
        return y
        
    else:
        # Allocate outputs
        y = cp.empty_like(x)
        mean = cp.empty((n_rows,), dtype=x.dtype) if training else None
        inv_var = cp.empty((n_rows,), dtype=x.dtype) if training else None

        # Compute strides in elements for each array
        x_row_stride = x.strides[0] // x.itemsize
        y_row_stride = y.strides[0] // y.itemsize

        # Map dtype to Triton flag
        dtype_flag = 0 if x.dtype == cp.float32 else 1  # 0=float32, 1=float16

        ### Set Grid ###
        grid = (n_rows,)

        layernorm_kernel_forward_training[grid](
            y.data.ptr,                                           # output_ptr
            inv_var.data.ptr if inv_var is not None else None,    # inv_var_ptr
            mean.data.ptr if mean is not None else None,
            x.data.ptr,                                           # input_ptr
            gamma.data.ptr if gamma is not None else None,        # gamma_ptr
            beta.data.ptr if beta is not None else None,          # beta_ptr
            x_row_stride,                                         # input_row_stride
            y_row_stride,                                         # output_row_stride
            dtype_flag,                                           # dtype_flag (constexpr)
            eps,                                                  # eps (constexpr)
            n_cols,                                               # n_cols (constexpr)
            BLOCK_SIZE,                                           # BLOCK_SIZE (constexpr)
        )

        if training:
            return y, mean, inv_var
        return y
        
def fused_layernorm_backward(x, mean, inv_var, dy, gamma=None, beta=None, use_dlpack=True):
    n_rows, n_cols = dy.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if not DLPACK_DISABLE and use_dlpack:
        x = torch.utils.dlpack.from_dlpack(x)
        dy = torch.utils.dlpack.from_dlpack(dy)
        mean = torch.utils.dlpack.from_dlpack(mean)
        inv_var = torch.utils.dlpack.from_dlpack(inv_var)
        gamma = torch.utils.dlpack.from_dlpack(gamma) if gamma is not None else None
        beta = torch.utils.dlpack.from_dlpack(beta) if beta is not None else None

        grad_dtype = torch.float32        
        dx = torch.empty_like(dy)
        dgamma = torch.zeros(n_cols, dtype=grad_dtype, device=dy.device) if gamma is not None else None
        dbeta = torch.zeros(n_cols, dtype=grad_dtype, device=dy.device) if beta is not None else None

        stride_x = x.stride(0)
        stride_dx = dx.stride(0)
        stride_dy = dy.stride(0)
        dtype_flag = 0 if dy.dtype == torch.float32 else 1

        grid = (n_rows,)
        layernorm_kernel_backward[grid](
            x, gamma, mean, inv_var, dx, dgamma, dbeta, dy,
            stride_x, stride_dx, stride_dy, n_cols, BLOCK_SIZE, dtype_flag,
        )

        dx = cp.from_dlpack(dx)
        
        # Convert back to original dtype
        if gamma is not None:
            dgamma = cp.from_dlpack(dgamma.to(gamma.dtype))
        if beta is not None:
            dbeta = cp.from_dlpack(dbeta.to(beta.dtype))

    else:

        dx = cp.empty_like(dy)
        dgamma = cp.zeros((n_cols,), dtype=grad_dtype) if gamma is not None else None
        dbeta = cp.zeros((n_cols,), dtype=grad_dtype) if beta is not None else None

        stride_x = x.strides[0] // x.itemsize
        stride_dx = dx.strides[0] // dx.itemsize
        stride_dy = dy.strides[0] // dy.itemsize
        dtype_flag = 0 if dy.dtype == cp.float32 else 1
        grid = (n_rows,)

        layernorm_kernel_backward[grid](
            x.data.ptr,
            gamma.data.ptr if gamma is not None else None,
            mean.data.ptr, inv_var.data.ptr, dx.data.ptr,
            dgamma.data.ptr if gamma is not None else None,
            dbeta.data.ptr if beta is not None else None,
            dy.data.ptr,
            stride_x, stride_dx, stride_dy, n_cols, BLOCK_SIZE, dtype_flag,
        )
        
        # Convert back to original dtype
        if gamma is not None:
            dgamma = dgamma.astype(gamma.dtype)
        if beta is not None:
            dbeta = dbeta.astype(beta.dtype)

    return dx, dgamma, dbeta

if __name__ == "__main__":

    import torch
    import triton
    import numpy as np
    import cupy as cp

    torch.manual_seed(42)
    np.random.seed(42)

    def test():
        # Test configuration
        batch_size = 4
        hidden_size = 8
        eps = 1e-5
        dtype = np.float16
            
        # Create input and weights
        x_np = np.random.randn(batch_size, hidden_size).astype(dtype)
        gamma_np = np.random.randn(hidden_size).astype(dtype)
        beta_np = np.random.randn(hidden_size).astype(dtype)
        grad_output_np = np.random.randn(batch_size, hidden_size).astype(dtype)

        # CuPy (for your Triton kernels)
        x_cp = cp.array(x_np)
        gamma_cp = cp.array(gamma_np)
        beta_cp = cp.array(beta_np)
        
        # PyTorch
        x_torch = torch.tensor(x_np, device='cuda', requires_grad=True)
        gamma_torch = torch.tensor(gamma_np, device='cuda', requires_grad=True)
        beta_torch = torch.tensor(beta_np, device='cuda', requires_grad=True)
        
        y_triton, mean_triton, inv_var_triton = fused_layernorm_forward(
            x_cp, gamma_cp, beta_cp, eps=eps, training=True, use_dlpack=True
        )
        
        # PyTorch forward
        torch_ln = torch.nn.LayerNorm(hidden_size, eps=eps).cuda()
        torch_ln.weight.data = gamma_torch.clone()
        torch_ln.bias.data = beta_torch.clone()
        y_torch = torch_ln(x_torch)
        
        # Compare
        y_triton_np = cp.asnumpy(y_triton)
        y_torch_np = y_torch.detach().cpu().numpy()
        
        forward_diff = np.abs(y_triton_np - y_torch_np)
        forward_match = np.allclose(y_triton_np, y_torch_np, rtol=1e-2, atol=1e-2)
        
        print(f"Output max diff: {forward_diff.max():.2e}")
        
        # FIX: Create upstream gradient with SAME dtype as input!
        grad_output_cp = cp.array(grad_output_np)
        grad_output_torch = torch.tensor(grad_output_np, device='cuda')
        
        # Your Triton backward
        dx_triton, dgamma_triton, dbeta_triton = fused_layernorm_backward(
            x_cp, mean_triton, inv_var_triton, grad_output_cp, 
            gamma_cp, beta_cp, use_dlpack=True
        )
        
        # PyTorch backward
        y_torch.backward(grad_output_torch)
        dx_torch = x_torch.grad
        dgamma_torch = torch_ln.weight.grad
        dbeta_torch = torch_ln.bias.grad
        
        # Compare dx
        dx_triton_np = cp.asnumpy(dx_triton)
        dx_torch_np = dx_torch.detach().cpu().numpy()
        dx_diff = np.abs(dx_triton_np - dx_torch_np)
        dx_match = np.allclose(dx_triton_np, dx_torch_np, rtol=1e-2, atol=1e-2)
        
        print(f"\ndx max diff: {dx_diff.max():.2e}")
        print(f"dx relative error: {(dx_diff / (np.abs(dx_torch_np) + 1e-8)).mean():.2e}")
        
        # Compare dgamma
        dgamma_triton_np = cp.asnumpy(dgamma_triton)
        dgamma_torch_np = dgamma_torch.detach().cpu().numpy()
        dgamma_diff = np.abs(dgamma_triton_np - dgamma_torch_np)
        dgamma_match = np.allclose(dgamma_triton_np, dgamma_torch_np, rtol=1e-2, atol=1e-2)
        
        print(f"\ndgamma max diff: {dgamma_diff.max():.2e}")
        print(f"dgamma relative error: {(dgamma_diff / (np.abs(dgamma_torch_np) + 1e-8)).mean():.2e}")
        
        # Compare dbeta
        dbeta_triton_np = cp.asnumpy(dbeta_triton)
        dbeta_torch_np = dbeta_torch.detach().cpu().numpy()
        dbeta_diff = np.abs(dbeta_triton_np - dbeta_torch_np)
        dbeta_match = np.allclose(dbeta_triton_np, dbeta_torch_np, rtol=1e-2, atol=1e-2)
        
        print(f"\ndbeta max diff: {dbeta_diff.max():.2e}")
        print(f"dbeta relative error: {(dbeta_diff / (np.abs(dbeta_torch_np) + 1e-8)).mean():.2e}")
        
        print(f"\nForward:  {'PASS' if forward_match else 'FAIL'}")
        print(f"dx:       {'PASS' if dx_match else 'FAIL'}")
        print(f"dgamma:   {'PASS' if dgamma_match else 'FAIL'}")
        print(f"dbeta:    {'PASS' if dbeta_match else 'FAIL'}")
        print(f"\nOverall:  {'ALL TESTS PASSED' if all([forward_match, dx_match, dgamma_match, dbeta_match]) else 'SOME TESTS FAILED'}")
        
    test()