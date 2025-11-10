import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_activation_forward, fused_activation_backward

def auto_leaky_relu(input, negative_slope=0.1):
    mask_pos = Tensor(np.where(input.data > 0, 1, 0).astype(input.dtype, copy=False))
    mask_neg = 1 - mask_pos
    return input * mask_pos + input * mask_neg * negative_slope

def manual_leaky_relu(input, negative_slope=0.1):
    
    input_arr = get_inner_array(input)
    
    # Compute mask only once and reuse it
    mask = input_arr > 0
    output = np.where(mask, input_arr, negative_slope * input_arr)

    def _leaky_relu_backward(input_grad):
        if input.requires_grad:
            # Reuse the precomputed mask from closure
            grad_input = input_grad * np.where(mask, 1, negative_slope)

            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_leaky_relu_backward if requires_grad else None,
        grad_fn_name="<LeakyReLUBackward>" if requires_grad else None,
        device=input.device,
        dtype=input.dtype,
    )

    if requires_grad:
        out._add_parents(input)

    return out

def fused_leaky_relu(input):

    input_arr = get_inner_inner_array(input)
    output = fused_activation_forward(input_arr, act_func="leaky_relu")

    def _leaky_relu_backward(output_grad):
        output_grad = fused_activation_backward(input_arr, output_grad, act_func="leaky_relu")
        if input.grad is None:
                input.grad = output_grad
        else:
            input.grad += output_grad

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_leaky_relu_backward if requires_grad else None,
        grad_fn_name="<LeakyReLUBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def leaky_relu(input, auto=False, fused=False):
    if auto:
        return auto_leaky_relu(input)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_leaky_relu if _use_fused else manual_leaky_relu
        if fused and op is manual_leaky_relu:
            CHECKS.warn_triton_missing()
        return op(input)
