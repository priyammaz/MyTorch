import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_activation_forward, fused_activation_backward

def auto_relu_squared(input):
    mask = Tensor(np.where(input.data < 0, 0, 1).astype(input.dtype, copy=False))
    relu = mask * input
    return relu ** 2

def manual_relu_squared(input):
    
    input_arr = get_inner_array(input)
    mask = (input_arr < 0)
    input_arr[mask] = 0 # <- inplace replacement of values

    def _relu_squared_backward(output_grad):
        if input.requires_grad:
            
            grad_input = output_grad * np.where(mask, 2 * input_arr, 0)

            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        input_arr**2,
        requires_grad=requires_grad,
        grad_fn=_relu_squared_backward if requires_grad else None,
        grad_fn_name="<ReLUSquaredBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def fused_relu_squared(input):

    input_arr = get_inner_inner_array(input)
    output = fused_activation_forward(input_arr, act_func="relu_squared")

    def _relu_squared_backward(output_grad):
        output_grad = fused_activation_backward(input_arr, output_grad, act_func="relu_squared")
        if input.grad is None:
                input.grad = output_grad
        else:
            input.grad += output_grad

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_relu_squared_backward if requires_grad else None,
        grad_fn_name="<ReLUSquaredBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def relu_squared(input, auto=False, fused=False):

    if auto:
        return auto_relu_squared(input)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_relu_squared if _use_fused else manual_relu_squared
        if fused and op is manual_relu_squared:
            CHECKS.warn_triton_missing()
        return op(input)
