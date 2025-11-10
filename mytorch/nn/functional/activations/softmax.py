import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_softmax_forward, fused_softmax_backward

def auto_softmax(x, dim=-1):
    max_x = x.max(dim=dim, keepdims=True)
    x_shifted = x - max_x
    exp_x = x_shifted.exp()
    sum_exp = exp_x.sum(dim=dim, keepdims=True)
    return exp_x / sum_exp

def manual_softmax(x, dim=-1):

    x_arr = get_inner_array(x)

    max_val = np.max(x_arr, axis=dim, keepdims=True)
    shifted = x_arr - max_val
    exp_x = np.exp(shifted)
    sum_exp = np.sum(exp_x, axis=dim, keepdims=True)
    out_data = exp_x / sum_exp

    # Define manual backward
    def _softmax_backward(grad_output):

        if x.requires_grad:
            # Softmax derivative: grad_input = s * (grad - sum(grad*s))
            # s = out_data
            sum_grad_s = np.sum(grad_output * out_data, axis=dim, keepdims=True)
            grad_input = out_data * (grad_output - sum_grad_s)
            
            if x.grad is None:
                x.grad = grad_input
            else:
                x.grad += grad_input
    
    requires_grad = x.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_softmax_backward if requires_grad else None,
        grad_fn_name="<SoftmaxBackward>" if requires_grad else None
    )

    # Add child to autograd graph
    if requires_grad:
        out._add_parents(x)

    return out


def fused_softmax(x, dim=-1):

    ### Fused Ops Need Access to Raw Arrays and they must be on CUDA ###
    array = get_inner_inner_array(x)
    
    orig_shape = array.shape
    ndim = len(orig_shape)

    # grabs the dim we want. If we have ndim=4 and we want to take softmax #
    # over dim=2, then 2%4 is just 2. But if we say -1, then -1 % 4 is 3. #
    dim = dim % ndim 

    ### Permute so target dim is last ###
    if dim != ndim - 1:

        ### Put all other dimensions first ###
        permute_axis = [i for i in range(ndim) if i != dim] + [dim]
        array_perm = array.transpose(permute_axis)
    
    else:
        array_perm = array

    ### Flatten to (*,I) ###
    n_rows = int(np.prod(array_perm.shape[:-1]))
    n_cols = array_perm.shape[-1]
    reshaped = array_perm.reshape(n_rows, n_cols)
    
    ### Fused Softmax ###
    out_flat = fused_softmax_forward(reshaped)

    ### Reshape Back ###
    out_perm = out_flat.reshape(array_perm.shape)
    
    if dim != ndim - 1:
        inv_permute = np.argsort(permute_axis)
        out_data = out_perm.transpose(inv_permute)
    else:
        out_data = out_perm

    def _softmax_backward(input_grad):

        # Extract raw array
        if hasattr(input_grad, "_array"):
            grad_array = input_grad._array
        else:
            grad_array = input_grad

        # Get shapes
        orig_shape = grad_array.shape
        ndim = len(orig_shape)
        dim_idx = dim % ndim  # support negative dims

        # Flatten our grad and out data ###
        if dim_idx != ndim - 1:
            permute_axes = [i for i in range(ndim) if i != dim_idx] + [dim_idx]
            grad_perm = grad_array.transpose(permute_axes)
            out_perm = out_data.transpose(permute_axes)  # permute softmax output similarly
        else:
            grad_perm = grad_array
            out_perm = out_data

        n_rows = int(np.prod(grad_perm.shape[:-1]))
        n_cols = grad_perm.shape[-1]
        grad_flat = grad_perm.reshape(n_rows, n_cols)
        out_flat = out_perm.reshape(n_rows, n_cols)

        ### Fused Backward Op ###
        grad_input_flat = fused_softmax_backward(grad_flat, out_flat)

        # Step 4: Reshape back
        grad_input_perm = grad_input_flat.reshape(grad_perm.shape)

        if dim_idx != ndim - 1:
            inv_permute = np.argsort(permute_axes)
            grad_input = grad_input_perm.transpose(inv_permute)
        else:
            grad_input = grad_input_perm

        if x.grad is None:
            x.grad = grad_input
        else:
            x.grad += grad_input

    requires_grad = x.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_softmax_backward if requires_grad else None,
        grad_fn_name="<SoftmaxBackward>" if requires_grad else None
    )

    # Add child to autograd graph
    if requires_grad:
        out._add_parents(x)

    return out

def softmax(x, dim=-1, auto=False, fused=False):
    if auto:
        return auto_softmax(x, dim=dim)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_softmax if _use_fused else manual_softmax
        if fused and op is manual_softmax:
            CHECKS.warn_triton_missing()
        return op(x, dim) 
