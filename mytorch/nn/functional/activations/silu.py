from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_activation_forward, fused_activation_backward
from .sigmoid import auto_sigmoid, manual_sigmoid

def auto_silu(input):
    sigmoid_out = auto_sigmoid(input)
    return input * sigmoid_out

def manual_silu(input):
    
    input_arr = get_inner_array(input)
    
    # Compute sigmoid once and reuse it
    sigmoid_output = manual_sigmoid(input)
    
    # SiLU(x) = x * sigmoid(x)
    output = input_arr * sigmoid_output
    
    def _silu_backward(input_grad):
        if input.requires_grad:
            # derivative of silu(x) = sigmoid(x) * (x * (1 - sigmoid(x)) + 1)
            # This can be derived from: d/dx[x * sigmoid(x)]
            grad_input = input_grad * (sigmoid_output * (input_arr * (1 - sigmoid_output) + 1))
            
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_silu_backward if requires_grad else None,
        grad_fn_name="<SiLUBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )
    
    if requires_grad:
        out._add_parents(input)
    
    return out

def fused_silu(input):
    
    input_arr = get_inner_inner_array(input)
    output = fused_activation_forward(input_arr, act_func="silu")
    
    def _sigmoid_backward(output_grad):
        output_grad = fused_activation_backward(input_arr, output_grad, act_func="silu")
        
        if input.grad is None:
            input.grad = output_grad
        else:
            input.grad += output_grad
    
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_sigmoid_backward if requires_grad else None,
        grad_fn_name="<SiLUBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )
    
    if requires_grad:
        out._add_parents(input)
    
    return out

def silu(input, fused=False):
    _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
    op = fused_silu if _use_fused else manual_silu
    if fused and op is manual_silu:
        CHECKS.warn_triton_missing()
    return op(input)