"""
LayerNorm fused kernel inspired by https://github.com/lucidrains/triton-transformer/blob/main/triton_transformer/layernorm.py
As well as from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py
"""

import cupy as cp
import torch
import triton
import triton.language as tl
from .utils import calc_num_warps
from .utils import TritonPtr
from .flags import DLPACK_DISABLE
from .utils import get_dtype_flag

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
    inv_var_ptr, # Need for Backward Pass (N,) float32 # can be None during inference
    mean_ptr, # Need fo Backward Pass (N,) float32 # Can be None during inference
    input_ptr, 
    gamma_ptr,  # 1D vector shared across all samples (E, ) can be None if no weight
    beta_ptr,   # 1D vector shared across all samples (E, ) can be None if no bias
    input_row_stride, 
    output_row_stride,  
    DTYPE_FLAG: tl.constexpr, # Flag for if our data is float32, float16 or bfloat16
    eps: tl.constexpr,
    n_cols: tl.constexpr, # Dimensionality of our embeddings
    BLOCK_SIZE: tl.constexpr, # closest power of 2 to our dim of embeddings 
):

    """
    Operation:

    gamma * ((x - mu)/sqrt(var + eps)) + beta

    This kernel expects an (N x E) dimension matrix and so we can write an easy
    kernel that has every block normalize a single row! A more optimal method can be found
    in the Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py

    But we are trying to find a balance between giving our modules some teeth, but also easily 
    understandable! Im sorry to all the hardcore kernel engineers! 

    The mean and variance are precision sensitive (a sum over the whole row), so just like torch
    we compute them in float32 (upcasting each element inside the kernel, no float32 copies of the
    tensor) and save mean and 1/std as float32 for the backward pass
    """

    ### Which row are we normalizing? ###
    row_idx = tl.program_id(0)

    ### Map ptrs to correct dtype ###
    pointer_dtype = tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)
    output_ptr = tl.cast(output_ptr, tl.pointer_type(pointer_dtype))
    input_ptr = tl.cast(input_ptr, tl.pointer_type(pointer_dtype))
    if gamma_ptr is not None:
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(pointer_dtype))
    if mean_ptr is not None:
        mean_ptr = tl.cast(mean_ptr, tl.pointer_type(tl.float32))
    if inv_var_ptr is not None:
        inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(tl.float32))
    if beta_ptr is not None:
        beta_ptr = tl.cast(beta_ptr, tl.pointer_type(pointer_dtype))

    ### Get offsets for the full block and mask for invalid regions ###
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    ### Load Row (upcast to float32 for the statistics) ###
    row = tl.load(input_ptr + row_idx * input_row_stride + col_offsets, mask=mask, other=0.).to(tl.float32)
    
    ### Compute row mean and var w/ reduction ops ###
    row_mean = tl.sum(row, axis=0) / n_cols
    row_mean_centered = tl.where(mask, row - row_mean, 0.)
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = tl.rsqrt(row_var + eps)
    output = row_mean_centered * inv_var

    ### Scale and shift ###
    if gamma_ptr is not None:
        output = output * tl.load(gamma_ptr + col_offsets, mask=mask, other=0.).to(tl.float32)
    if beta_ptr is not None:
        output = output + tl.load(beta_ptr + col_offsets, mask=mask, other=0.).to(tl.float32)

    ### Write outputs (back in the input dtype) ###
    tl.store(output_ptr + row_idx * output_row_stride + col_offsets, output, mask=mask)

    ### No need to waste time writing outputs if we dont need it! ###
    if mean_ptr is not None and inv_var_ptr is not None:
        tl.store(mean_ptr + row_idx, row_mean)
        tl.store(inv_var_ptr + row_idx, inv_var)

@triton.jit
def layernorm_kernel_backward(
    x_ptr, 
    gamma_ptr, 
    mean_ptr, 
    inv_var_ptr, 
    dX_ptr, 
    dGamma_ptr, # (n_blocks, E) float32 partial sums, can be None
    dB_ptr,     # (n_blocks, E) float32 partial sums, can be None
    dY_ptr,
    stride_x, 
    stride_dx, 
    stride_dy, 
    n_rows,
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr, 
    ROWS_PER_BLOCK: tl.constexpr,
    DTYPE_FLAG: tl.constexpr
):
    
    """
    We need to compute the gradients w.r.t to the input x. This formula is as follows:

    y = x_hat * gamma + beta
    where x_hat = (x - mu) / sqrt(var + eps)
    dL/dx = dL/dy * dy/dx_hat * dx_hat / dx

    = 1/std * [dL/dy * gamma - mean(dL/dy*gamma) - x_hat * mean(dL/dy * gamma * x_hat)]
    = 1/std * [dx_hat - mean(dx_hat) - x_hat * mean(dx_hat* x_hat)]

    dL/dgamma = sum over rows of dL/dy * x_hat
    dL/dbeta  = sum over rows of dL/dy

    Every program processes ROWS_PER_BLOCK rows at once (a 2D block). The gamma/beta grads are a sum over
    ALL rows. Accumulating them with tl.atomic_add from every program into the same (E,) buffer was over
    half the runtime (thousands of programs contending on the same E cells), so instead each program
    writes the partial sum for its own block of rows into its own row of an (n_blocks, E) buffer, and the
    caller does a single .sum(0) at the end.
    """

    block_idx = tl.program_id(0)

    ### Map Pointers To Correct Dtype ###
    pointer_type = tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)
    x_ptr = tl.cast(x_ptr, tl.pointer_type(pointer_type))
    mean_ptr = tl.cast(mean_ptr, tl.pointer_type(tl.float32))
    inv_var_ptr = tl.cast(inv_var_ptr, tl.pointer_type(tl.float32))
    dX_ptr = tl.cast(dX_ptr, tl.pointer_type(pointer_type))
    dY_ptr = tl.cast(dY_ptr, tl.pointer_type(pointer_type))
    if gamma_ptr is not None:
        gamma_ptr = tl.cast(gamma_ptr, tl.pointer_type(pointer_type))
    if dGamma_ptr is not None:
        dGamma_ptr = tl.cast(dGamma_ptr, tl.pointer_type(tl.float32))   
    if dB_ptr is not None:
        dB_ptr = tl.cast(dB_ptr, tl.pointer_type(tl.float32))   
    
    ### Rows (R,) and columns (E,) this program handles ###
    rows = block_idx * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = rows < n_rows
    cols = tl.arange(0, BLOCK_SIZE)
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    ### Load all the data (R, E) in float32 ###
    x = tl.load(x_ptr + rows[:, None] * stride_x + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dY_ptr + rows[:, None] * stride_dy + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    row_mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0)[:, None]
    row_inv_std = tl.load(inv_var_ptr + rows, mask=row_mask, other=0.0)[:, None]

    if gamma_ptr is not None:
        gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.).to(tl.float32)[None, :]
    else:
        gamma = 1.0

    ### Compute backward pass ###
    x_hat = (x - row_mean) * row_inv_std
    wdy = gamma * dy # <- upstream grad (dL/dy)(dy/dx_hat)
    mean1 = tl.sum(x_hat * wdy, axis=1)[:, None] / n_cols
    mean2 = tl.sum(wdy, axis=1)[:, None] / n_cols
    dx = row_inv_std * (wdy - (x_hat * mean1 + mean2))

    ### Store dx ###
    tl.store(dX_ptr + rows[:, None] * stride_dx + cols[None, :], dx, mask=mask)

    ### Partial sums of the weight/bias grads for this block of rows (kept in float32) ###
    if dGamma_ptr is not None:
        tl.store(dGamma_ptr + block_idx * n_cols + cols, tl.sum(dy * x_hat, axis=0), mask=col_mask)
    if dB_ptr is not None:
        tl.store(dB_ptr + block_idx * n_cols + cols, tl.sum(dy, axis=0), mask=col_mask)


def _rows_per_block(block_size):
    ### ~8192 elements per program was the fastest in our sweeps (8 rows for a 768 dim model) ###
    return max(1, 8192 // block_size)


def fused_layernorm_forward(x, gamma=None, beta=None, eps=1e-5, training=True, use_dlpack=True):

    """
    x: Input (N, E)
    gamma: scale parameter (E,)
    beta: shift parameter (E,)

    Returns:
        y: Ouptut (N, E)
        mean: per row mean (N,) float32
        inv_var: per row 1/std (N,) float32
    """
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    dtype_flag = get_dtype_flag(x.dtype)  # 0=float32, 1=float16, 2=bfloat16
    grid = (n_rows,)

    if not DLPACK_DISABLE and use_dlpack:

        x = torch.utils.dlpack.from_dlpack(x)
        if gamma is not None:
            gamma = torch.utils.dlpack.from_dlpack(gamma)
        if beta is not None:
            beta = torch.utils.dlpack.from_dlpack(beta)

        # Allocate outputs (saved statistics are always float32)
        y = torch.empty_like(x)
        inv_var = torch.empty(n_rows, dtype=torch.float32, device=x.device) if training else None
        mean = torch.empty(n_rows, dtype=torch.float32, device=x.device) if training else None

        layernorm_kernel_forward_training[grid](
            y, inv_var, mean, x, gamma, beta,
            x.stride(0), y.stride(0),
            dtype_flag, eps, n_cols, BLOCK_SIZE,
        )
         
        y = cp.from_dlpack(y)
        if training:
            return y, cp.from_dlpack(mean), cp.from_dlpack(inv_var)
        return y
        
    else:
        # Allocate outputs (saved statistics are always float32)
        y = cp.empty_like(x)
        mean = cp.empty((n_rows,), dtype=cp.float32) if training else None
        inv_var = cp.empty((n_rows,), dtype=cp.float32) if training else None

        layernorm_kernel_forward_training[grid](
            TritonPtr(y),
            TritonPtr(inv_var) if inv_var is not None else None,
            TritonPtr(mean) if mean is not None else None,
            TritonPtr(x),
            TritonPtr(gamma) if gamma is not None else None,
            TritonPtr(beta) if beta is not None else None,
            x.strides[0] // x.itemsize, y.strides[0] // y.itemsize,
            dtype_flag, eps, n_cols, BLOCK_SIZE,
        )

        if training:
            return y, mean, inv_var
        return y
        
def fused_layernorm_backward(x, mean, inv_var, dy, gamma=None, beta=None, use_dlpack=True):
    n_rows, n_cols = dy.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    ROWS_PER_BLOCK = _rows_per_block(BLOCK_SIZE)
    n_blocks = triton.cdiv(n_rows, ROWS_PER_BLOCK)
    dtype_flag = get_dtype_flag(dy.dtype)
    grid = (n_blocks,)

    gamma_dtype = gamma.dtype if gamma is not None else None
    beta_dtype = beta.dtype if beta is not None else None

    if not DLPACK_DISABLE and use_dlpack:
        x = torch.utils.dlpack.from_dlpack(x)
        dy = torch.utils.dlpack.from_dlpack(dy)
        mean = torch.utils.dlpack.from_dlpack(mean)
        inv_var = torch.utils.dlpack.from_dlpack(inv_var)
        gamma = torch.utils.dlpack.from_dlpack(gamma) if gamma is not None else None

        dx = torch.empty_like(dy)
        dgamma_blocks = torch.empty((n_blocks, n_cols), dtype=torch.float32, device=dy.device) if gamma is not None else None
        dbeta_blocks = torch.empty((n_blocks, n_cols), dtype=torch.float32, device=dy.device) if beta is not None else None

        layernorm_kernel_backward[grid](
            x, gamma, mean, inv_var, dx, dgamma_blocks, dbeta_blocks, dy,
            x.stride(0), dx.stride(0), dy.stride(0), n_rows, n_cols, BLOCK_SIZE, ROWS_PER_BLOCK, dtype_flag,
        )

        dx = cp.from_dlpack(dx)
        dgamma = cp.from_dlpack(dgamma_blocks.sum(0)).astype(gamma_dtype, copy=False) if gamma is not None else None
        dbeta = cp.from_dlpack(dbeta_blocks.sum(0)).astype(beta_dtype, copy=False) if beta is not None else None

    else:
        dx = cp.empty_like(dy)
        dgamma_blocks = cp.empty((n_blocks, n_cols), dtype=cp.float32) if gamma is not None else None
        dbeta_blocks = cp.empty((n_blocks, n_cols), dtype=cp.float32) if beta is not None else None

        layernorm_kernel_backward[grid](
            TritonPtr(x),
            TritonPtr(gamma) if gamma is not None else None,
            TritonPtr(mean), TritonPtr(inv_var), TritonPtr(dx),
            TritonPtr(dgamma_blocks) if gamma is not None else None,
            TritonPtr(dbeta_blocks) if beta is not None else None,
            TritonPtr(dy),
            x.strides[0] // x.itemsize, dx.strides[0] // dx.itemsize, dy.strides[0] // dy.itemsize,
            n_rows, n_cols, BLOCK_SIZE, ROWS_PER_BLOCK, dtype_flag,
        )

        dgamma = dgamma_blocks.sum(0).astype(gamma_dtype, copy=False) if gamma is not None else None
        dbeta = dbeta_blocks.sum(0).astype(beta_dtype, copy=False) if beta is not None else None

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