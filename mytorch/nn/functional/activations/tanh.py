import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_activation_forward, fused_activation_backward

def manual_tanh(input):
    
    input_arr = get_inner_array(input)
    output = np.tanh(input_arr)

    def _tanh_backward(input_grad):
        if input.requires_grad:
            # derivative of tanh(x) = 1 - tanh(x)^2
            grad_input = input_grad * (1 - output ** 2)
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_tanh_backward if requires_grad else None,
        grad_fn_name="<TanhBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def fused_tanh(input):

    input_arr = get_inner_inner_array(input)
    output = fused_activation_forward(input_arr, act_func="tanh")

    def _tanh_backward(output_grad):
        output_grad = fused_activation_backward(input_arr, output_grad, act_func="tanh")
        if input.grad is None:
                input.grad = output_grad
        else:
            input.grad += output_grad

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_tanh_backward if requires_grad else None,
        grad_fn_name="<TanhBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def tanh(input, fused=False):
    _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
    op = fused_tanh if _use_fused else manual_tanh
    if fused and op is manual_tanh:
        CHECKS.warn_triton_missing()
    return op(input)
