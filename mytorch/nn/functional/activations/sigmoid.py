import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_activation_forward, fused_activation_backward

def auto_sigmoid(input):
    return 1 / (1 + (-input).exp())

def manual_sigmoid(input):
    
    input_arr = get_inner_array(input)
    output = 1 / (1 + np.exp(-input_arr))
    
    def _sigmoid_backward(input_grad):
        if input.requires_grad:
            # derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            grad_input = input_grad * output * (1 - output)
            
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_sigmoid_backward if requires_grad else None,
        grad_fn_name="<SigmoidBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )
    
    if requires_grad:
        out._add_parents(input)
    
    return out


def fused_sigmoid(input):
    
    input_arr = get_inner_inner_array(input)
    output = fused_activation_forward(input_arr, act_func="sigmoid")
    
    def _sigmoid_backward(output_grad):
        output_grad = fused_activation_backward(input_arr, output_grad, act_func="sigmoid")
        
        if input.grad is None:
            input.grad = output_grad
        else:
            input.grad += output_grad
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_sigmoid_backward if requires_grad else None,
        grad_fn_name="<SigmoidBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )
    
    if requires_grad:
        out._add_parents(input)
    
    return out


def sigmoid(input, auto=False, fused=False):
    if auto:
        return auto_sigmoid(input)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_sigmoid if _use_fused else manual_sigmoid
        if fused and op is manual_sigmoid:
            CHECKS.warn_triton_missing()
        return op(input)