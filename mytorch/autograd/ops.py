import numpy as np
from .function import Function
from .ops_registry import register_op


def _broadcasted_grad_accumulate(x_shape, x_grad):

    grad_shape = x_grad.shape

    assert len(x_shape) == len(grad_shape), "Gradient and tensor shapes must be the same length! Only different by broadcasting"

    sum_axes = [idx for idx, (x_dim, grad_dim) in enumerate(zip(x_shape, grad_shape)) if x_dim == 1 and grad_dim != 1]
    if sum_axes:
        x_grad = np.sum(x_grad, axis=tuple(sum_axes), keepdims=True)

    return x_grad

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):

        ctx.save_for_backward(a.shape, b.shape)
        
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        a_req_grad, b_req_grad = ctx.needs_input_grad
        
        a_grad = _broadcasted_grad_accumulate(a_shape, grad_output) if a_req_grad else None
        b_grad = _broadcasted_grad_accumulate(b_shape, grad_output) if b_req_grad else None

        ctx = None

        return a_grad, b_grad
    
class Sub(Function):

    """
    Same as __add__ but now subtraction (with accumulation for broadcasting)
    O = A - B
    dO/dA = 1
    dO/dB = -1
    """
    
    @staticmethod
    def forward(ctx, a, b):

        ctx.save_for_backward(a.shape, b.shape)

        return a - b

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        a_req_grad, b_req_grad = ctx.needs_input_grad
        
        a_grad = _broadcasted_grad_accumulate(a_shape, grad_output) if a_req_grad else None
        b_grad = _broadcasted_grad_accumulate(b_shape, -grad_output) if b_req_grad else None

        ctx = None

        return a_grad, b_grad
    

class Mul(Function):
    """
    Element-wise multiplication of two tensors (with broadcasting).

    Forward: O = A * B
    Backward:
      dO/dA = grad_output * B
      dO/dB = grad_output * A
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):

        a, b = ctx.saved_tensors
        a_shape, b_shape = a.shape, b.shape
        a_req_grad, b_req_grad = ctx.needs_input_grad

        grad_a = grad_b = None

        if a_req_grad:
            grad_a = grad_output * b
            grad_a = _broadcasted_grad_accumulate(a_shape, grad_a)

        if b_req_grad:
            grad_b = grad_output * a
            grad_b = _broadcasted_grad_accumulate(b_shape, grad_b)

        ctx = None

        return grad_a, grad_b
        
class MatMul(Function):
    """
    Matrix multiplication (supports batch dimensions).

    Forward: O = A @ B
    Backward:
        dO/dA = grad_output @ B^T
        dO/dB = A^T @ grad_output
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        a_shape, b_shape = a.shape, b.shape
        a_req_grad, b_req_grad = ctx.needs_input_grad

        grad_a = grad_b = None

        if a_req_grad:
            grad_a = grad_output @ b.swapaxes(-1, -2)
            grad_a = _broadcasted_grad_accumulate(a_shape, grad_a)

        if b_req_grad:
            grad_b = a.swapaxes(-1, -2) @ grad_output
            grad_b = _broadcasted_grad_accumulate(b_shape, grad_b)
        
        ctx = None

        return grad_a, grad_b
    
class Div(Function):
    """
    Element-wise division with broadcasting support.

    Forward: O = A / B
    Backward:
        dO/dA = 1 / B
        dO/dB = -A / B^2
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        a_shape, b_shape = a.shape, b.shape
        a_req_grad, b_req_grad = ctx.needs_input_grad

        grad_a = grad_b = None

        if a_req_grad:
            grad_a = grad_output / b
            grad_a = _broadcasted_grad_accumulate(a_shape, grad_a)

        if b_req_grad:
            grad_b = grad_output * (-a / (b ** 2))
            grad_b = _broadcasted_grad_accumulate(b_shape, grad_b)

        ctx = None

        return grad_a, grad_b
    
class Pow(Function):
    """
    Element-wise power with a constant exponent.

    Forward: O = A^k
    Backward: dO/dA = k * A^(k-1)
    """

    @staticmethod
    def forward(ctx, a, exponent):
        ctx.save_for_backward(a, exponent)
        return a ** exponent

    @staticmethod
    def backward(ctx, grad_output):
        a, exponent = ctx.saved_tensors
        a_req_grad, _ = ctx.needs_input_grad

        grad_a = None
        if a_req_grad:
            grad_a = grad_output * (exponent * (a ** (exponent - 1)))
            grad_a = _broadcasted_grad_accumulate(a.shape, grad_a)

        ctx = None

        return grad_a
    
class Index(Function):
    """
    Supports tensor indexing with slices, integers, and arrays.
    """

    @staticmethod
    def forward(ctx, a, idx):

        def _normalize_index(a, idx):
            
            """
            This is a sanity check as if indexes are passed in as a tensor
            it will break our Array, as Array expects other Arrays. So we
            make sure to always post process our indexes to make sure it all 
            matches! 
            """
            from ..tensor import Tensor


            if isinstance(idx, Tensor):
                return idx.data.astype(np.int32)
            elif isinstance(idx, (list, tuple)):
                return tuple(
                    (i.data.astype(np.int32) if isinstance(i, Tensor) else np.array(i, dtype=np.int32))
                    if isinstance(i, (list, Tensor)) else i
                    for i in idx
                )
            else:
                return idx
                    
        idx = _normalize_index(a, idx)

        ctx.save_for_backward(a, idx)
        return a[idx]

    @staticmethod
    def backward(ctx, grad_output):

        a, idx = ctx.saved_tensors
        a_req_grad, _ = ctx.needs_input_grad


        grad_a = None
        if a_req_grad:
            grad_a = np.zeros_like(a)
            grad_a[idx] += grad_output

        ctx = None

        return grad_a  

class Transpose(Function):

    """
    Swap two dimensions in forward pass, 
    Swap them back in the backward pass
    """

    @staticmethod
    def forward(ctx, a, dim1, dim2):
        ctx.save_for_backward(dim1, dim2)
        return a.swapaxes(dim1, dim2)

    @staticmethod
    def backward(ctx, grad_output):
        dim1, dim2 = ctx.saved_tensors

        ctx = None

        return grad_output.swapaxes(dim1, dim2)   
    
class Permute(Function):
    """
    Permute multiple dimensions in the forward pass
    Return back in the backward pass
    """
    @staticmethod
    def forward(ctx, a, dims):
        ctx.save_for_backward(dims)
        return np.transpose(a, axes=dims)

    @staticmethod
    def backward(ctx, grad_output):
        dims, = ctx.saved_tensors
        inv_dims = np.argsort(dims)

        ctx = None

        return grad_output.transpose(inv_dims), None  # None for dims

class Reshape(Function):
    """
    Reshape a tensor.
    Forward: output = A.reshape(*shape)
    Backward: reshape incoming gradient back to the original shape.
    """

    @staticmethod
    def forward(ctx, a, *shape):
        # Save original shape for backward
        ctx.save_for_backward(a.shape)
        out = a.reshape(*shape)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        original_shape, = ctx.saved_tensors
        a_req_grad = ctx.needs_input_grad

        grad_a = None
        if a_req_grad:
            # Reshape gradient back to input shape
            grad_a = grad_output.reshape(original_shape)

        ctx = None

        return grad_a
    
class Exp(Function):
    """
    Element-wise natural logarithm.
    Forward: O = e^x
    Backward: dO/dA = e^x
    """
    @staticmethod
    def forward(ctx, a):
        out = np.exp(a)
        ctx.save_for_backward(out)  # save forward result
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        ctx = None
        
        return grad_output * out
    
class Log(Function):
    """
    Element-wise natural logarithm.
    Forward: O = log(A)
    Backward: dO/dA = 1 / A
    """

    @staticmethod
    def forward(ctx, a):
        out = np.log(a)
        ctx.save_for_backward(a)  # save input for backward
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        a_req_grad = ctx.needs_input_grad

        grad_a = None
        if a_req_grad:
            grad_a = grad_output / a

        ctx = None

        return grad_a
    
class Sum(Function):
    """
    Sum across a dimension.
    Forward: output = A.sum(axis=dim, keepdims=keepdims)
    Backward: distribute incoming gradient to all elements along summed axes
    """

    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):
        ctx.save_for_backward(a.shape, dim, keepdims)
        out = np.sum(a, axis=dim, keepdims=keepdims)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, dim, keepdims = ctx.saved_tensors
        a_req_grad = ctx.needs_input_grad

        grad_a = None
        if a_req_grad:
            # Broadcast the incoming gradient back to the input shape
            if not keepdims and dim is not None:
                grad_output = grad_output.reshape(
                    [1 if i in (dim if isinstance(dim, tuple) else (dim,)) else s
                     for i, s in enumerate(a_shape)]
                )
            grad_a = np.broadcast_to(grad_output, a_shape)

        ctx = None

        return grad_a

class Mean(Function):
    """
    Compute mean across specified dimensions.
    Forward: output = A.mean(axis=dim, keepdims=keepdims)
    Backward: broadcast incoming gradient and divide by number of elements reduced
    """

    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):

        if dim is None:
            dim = tuple(range(a.ndim))

        ctx.save_for_backward(a.shape, dim, keepdims)

        out = np.mean(a, axis=dim, keepdims=keepdims)
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, dim, keepdims = ctx.saved_tensors
        a_req_grad = ctx.needs_input_grad
        grad_a = None
        if a_req_grad:
            dims = dim if isinstance(dim, tuple) else (dim,)
            
            # Number of elements reduces 
            num_vals = np.prod([a_shape[d] for d in dims])

            # Reshape grad_output if keepdims=False
            if not keepdims:
                grad_output = grad_output.reshape(
                    [1 if i in dims else s for i, s in enumerate(a_shape)]
                )

            # Broadcast and scale by 1 / num_vals
            grad_a = np.broadcast_to(grad_output, a_shape) / num_vals
        
        ctx = None

        return grad_a
    
class Var(Function):
    """
    Compute variance across specified dimensions.
    Forward: var = mean((A - mean(A, dim)) ** 2, axis=dim, keepdims=keepdims)
    Backward: dVar/dA = 2 * (A - mean(A)) / N * grad_output
    """

    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):
        if dim is None:
            dim = tuple(range(a.ndim))

        # Compute mean for variance
        mean_vals = np.mean(a, axis=dim, keepdims=True)
        var_vals = np.mean((a - mean_vals) ** 2, axis=dim, keepdims=keepdims)

        # Save needed info for backward
        ctx.save_for_backward(a, mean_vals, dim, keepdims)

        return var_vals

    @staticmethod
    def backward(ctx, grad_output):
        a, mean_vals, dim, keepdims = ctx.saved_tensors
        a_req_grad = ctx.needs_input_grad
        grad_a = None

        if a_req_grad:
            dims = dim if isinstance(dim, tuple) else (dim,)
            
            # Number of elements reduced
            num_vals = np.prod([a.shape[d] for d in dims])

            # Reshape grad_output if keepdims=False
            if not keepdims:
                grad_output = grad_output.reshape(
                    [1 if i in dims else s for i, s in enumerate(a.shape)]
                )

            # Broadcast and compute gradient
            grad_a = np.broadcast_to(grad_output, a.shape) * 2 * (a - mean_vals) / num_vals

        ctx = None

        return grad_a
    
class Max(Function):
    """
    Compute max along specified axis.
    Forward: output = A.max(axis=dim, keepdims=keepdims)
    Backward: gradient only flows to positions where the maximum occurred
    """

    @staticmethod
    def forward(ctx, a, dim=None, keepdims=False):
        if dim is None:
            dim = tuple(range(a.ndim))

        out = np.max(a, axis=dim, keepdims=keepdims)

        # Save input and axis info for backward
        ctx.save_for_backward(a, out, dim, keepdims)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, out, dim, keepdims = ctx.saved_tensors
        a_req_grad = ctx.needs_input_grad
        grad_a = None

        if a_req_grad:
            grad_a = a.xp.zeros_like(a)

            # Reshape grad_output if keepdims=False
            if dim is not None and not keepdims:
                grad_output = np.expand_dims(grad_output, axis=dim)

            # Broadcast grad_output to input shape
            grad_broadcast = np.broadcast_to(grad_output, a.shape)

            # Mask for positions where maximum occurred
            mask = (a == (out if keepdims else np.expand_dims(out, axis=dim)))

            grad_a = grad_broadcast * mask

        ctx = None

        return grad_a
    
class ArgMax(Function):
    """
    Compute indices of the maximum value along a specified axis.
    Forward: output = A.argmax(axis=dim)
    Backward: non-differentiable, so return zeros with the shape of input.
    """

    @staticmethod
    def forward(ctx, a, dim=-1):
        ctx.save_for_backward(a.xp, a.shape) 
        out = np.argmax(a, axis=dim)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        xp, a_shape = ctx.saved_tensors
        grad_a = xp.zeros(a_shape, dtype=grad_output.dtype)
        
        ctx = None

        return grad_a