"""
SwiGLU activation functions follow the pattern of:
swiglu(x,y) = x * silu(y) = x * y * sigmoid(y) 
Fixed to match Liger Kernel implementation exactly
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""
import cupy as cp
import torch
import triton
import triton.language as tl
from .activations import silu_forward_kernel, silu_backward_kernel
from .utils import calc_num_warps

@triton.jit
def silu(x):
    """Inline SiLU function"""
    return x * tl.sigmoid(x)

@triton.jit
def swiglu_forward_kernel(
    a_ptr, 
    b_ptr, 
    out_ptr, 
    stride, 
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)
    
    a_ptr = tl.cast(a_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    b_ptr = tl.cast(b_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    out_ptr = tl.cast(out_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))

    # Locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    out_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data - sigmoid requires float32
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
    stride, 
    n_cols: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    
    a_ptr = tl.cast(a_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    b_ptr = tl.cast(b_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dout_ptr = tl.cast(dout_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))

    # Locate start index
    dout_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    
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

def fused_swiglu_forward(a, b):

    ori_shape = a.shape
    n_cols = ori_shape[-1]
    
    a = a.reshape(-1, n_cols)
    b = b.reshape(-1, n_cols)

    n_rows = a.shape[0]
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = calc_num_warps(BLOCK_SIZE)

    a = torch.utils.dlpack.from_dlpack(a)
    b = torch.utils.dlpack.from_dlpack(b)

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    c = torch.empty_like(a)

    swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        DTYPE_FLAG=0 if a.dtype == torch.float32 else 1
    )
    
    # Return a and b as well so we don't have to reshape again in backward
    return cp.from_dlpack(a), cp.from_dlpack(b), cp.from_dlpack(c.view(*ori_shape))

def fused_swiglu_backward(output_grad, a, b):

    ori_shape = output_grad.shape
    n_cols = ori_shape[-1]
    
    dc = output_grad.reshape(-1, n_cols)
    
    n_rows = dc.shape[0]
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = calc_num_warps(BLOCK_SIZE)
    
    a = torch.utils.dlpack.from_dlpack(a)
    b = torch.utils.dlpack.from_dlpack(b)
    dc = torch.utils.dlpack.from_dlpack(dc)

    swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        DTYPE_FLAG=0 if a.dtype == torch.float32 else 1
    )
    
    # a and b now contain gradients
    return cp.from_dlpack(a.view(*ori_shape)), cp.from_dlpack(b.view(*ori_shape))


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    def pytorch_swiglu_forward(a, b):
        return F.silu(a) * b

    def test_forward(shapes, dtype=cp.float32, device='cuda', atol=1e-2, rtol=1e-2):
        
        for shape in shapes:
            print(f"\nShape: {shape}")
            
            # Create random inputs
            a = cp.random.normal(size=shape).astype(dtype)
            b = cp.random.normal(size=shape).astype(dtype)

            a_torch = torch.tensor(a.get())
            b_torch = torch.tensor(b.get())
            
            # PyTorch reference
            pytorch_out = pytorch_swiglu_forward(a_torch.clone(), b_torch.clone())
            
            # Triton implementation
            _, _, triton_out = fused_swiglu_forward(a, b)
    
            is_close = cp.allclose(cp.asarray(pytorch_out.detach().cpu().numpy()), triton_out, atol=atol, rtol=rtol)

            if not is_close:
                print("FAILED")
                print(f"PyTorch sample: {pytorch_out.flatten()[:5]}")
                print(f"Triton sample:  {triton_out.flatten()[:5]}")
            
            print("FORWARD SUCCESS")

    def test_backward(shapes, dtype=cp.float32, device='cuda', atol=1e-2, rtol=1e-2):
        
        for shape in shapes:
            print(f"\nShape: {shape}")
            
            # Create random inputs
            a = cp.random.normal(size=shape).astype(dtype)
            b = cp.random.normal(size=shape).astype(dtype)
            grad_output = cp.random.normal(size=shape).astype(dtype)
            
            
            # PyTorch reference (using autograd)
            a_torch = torch.tensor(a.get(), requires_grad=True)
            b_torch = torch.tensor(b.get(), requires_grad=True)
            out_pt = pytorch_swiglu_forward(a_torch, b_torch)
            out_pt.backward(torch.tensor(grad_output.get()))
            
            # Triton implementation
            a_triton, b_triton, out_triton = fused_swiglu_forward(a, b)
            grad_a_triton, grad_b_triton = fused_swiglu_backward(grad_output, a_triton, b_triton)
            
            # Compare gradients
            is_close_a = cp.allclose(cp.asarray(a_torch.grad.detach().cpu().numpy()), grad_a_triton, atol=atol, rtol=rtol)
            is_close_b = cp.allclose(cp.asarray(b_torch.grad.detach().cpu().numpy()), grad_b_triton, atol=atol, rtol=rtol)

            if not is_close_a or not is_close_b:
                print("FAILED")
                if not is_close_a:
                    print(f"    PyTorch grad_a sample: {a_torch.grad.flatten()[:5]}")
                    print(f"    Triton grad_a sample:  {grad_a_triton.flatten()[:5]}")
                if not is_close_b:
                    print(f"    PyTorch grad_b sample: {b_torch.grad.flatten()[:5]}")
                    print(f"    Triton grad_b sample:  {grad_b_triton.flatten()[:5]}")

            print("BACKWARD SUCCESS")

    # Test shapes
    test_shapes = [
        (10,4,128),
        (2,12,1424)
    ]
    
    # Test forward pass
    test_forward(test_shapes, dtype=cp.float16)
    
    # Test backward pass
    test_backward(test_shapes, dtype=cp.float16)
    
