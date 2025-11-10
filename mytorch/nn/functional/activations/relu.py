import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_activation_forward, fused_activation_backward

def auto_relu(input):
    mask = Tensor(np.where(input.data < 0, 0, 1).astype(input.dtype, copy=False))
    return input * mask

def manual_relu(input):
    
    input_arr = get_inner_array(input)
    mask = (input_arr < 0)
    input_arr[mask] = 0 # <- inplace replacement of values

    def _relu_backward(output_grad):
        if input.requires_grad:
            # grad_input = input_grad * (input.data > 0)
            output_grad[mask] = 0

            if input.grad is None:
                input.grad = output_grad
            else:
                input.grad += output_grad
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        input_arr,
        requires_grad=requires_grad,
        grad_fn=_relu_backward if requires_grad else None,
        grad_fn_name="<ReLUBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def fused_relu(input):

    """
    This is a little wasteful. Technically all we need to store for the backward
    pass for autograd is the mask, but we store here the entire array! We need to 
    do this for most of our ops so I leave it like this for simplicity. 

    But at the penalty for about 5% more memory usage we get a 9% increase in 
    performance, so atleast we get something for it!
    """
    input_arr = get_inner_inner_array(input)
    output = fused_activation_forward(input_arr, act_func="relu")

    def _relu_backward(output_grad):
        output_grad = fused_activation_backward(input_arr, output_grad, act_func="relu")
        if input.grad is None:
                input.grad = output_grad
        else:
            input.grad += output_grad

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_relu_backward if requires_grad else None,
        grad_fn_name="<ReLUBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def relu(input, auto=False, fused=False):

    if auto:
        return auto_relu(input)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_relu if _use_fused else manual_relu
        if fused and op is manual_relu:
            CHECKS.warn_triton_missing()
        return op(input)
