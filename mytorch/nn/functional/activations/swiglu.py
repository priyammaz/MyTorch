from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_swiglu_forward, fused_swiglu_backward
from .silu import auto_silu
from .sigmoid import manual_sigmoid

def auto_swiglu(input_a, input_b):
    return input_b * auto_silu(input_a)

def manual_swiglu(input_a, input_b):

    a = get_inner_array(input_a)
    b = get_inner_array(input_b)

    sigmoid_a = manual_sigmoid(input_b)
    silu_a = a * sigmoid_a
    output = b * silu_a

    def _swiglu_backward(grad_out):

        # gradient wrt b:
        # d/db [b * SiLU(a)] = SiLU(a)
        if input_b.requires_grad:
            grad_b = grad_out * silu_a
            if input_b.grad is None:
                input_b.grad = grad_b
            else:
                input_b.grad += grad_b

        # gradient wrt a:
        # d/da [b * (a * sigmoid(a))]
        #
        # First derivative of SiLU(a):
        # SiLU'(a) = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
        if input_a.requires_grad:
            dsilu_a = sigmoid_a * (1 + a * (1 - sigmoid_a))
            grad_a = grad_out * dsilu_a * b

            if input_a.grad is None:
                input_a.grad = grad_a
            else:
                input_a.grad += grad_a

    requires_grad = (
        (input_a.requires_grad or input_b.requires_grad)
        and Tensor.build_graph_enabled()
    )

    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_swiglu_backward if requires_grad else None,
        grad_fn_name="<SwiGLUBackward>" if requires_grad else None,
        device=input_a.device,
        dtype=input_a.dtype,
    )

    if requires_grad:
        out._add_parents(input_a, input_b)

    return out

def fused_swiglu(input_a, input_b):

    a = get_inner_inner_array(input_a)
    b = get_inner_inner_array(input_b)
    a, b, output = fused_swiglu_forward(a, b)
    
    def _swiglu_backward(output_grad):
        grad_a, grad_b = fused_swiglu_backward(output_grad, a, b)
        
        if input_a.grad is None:
            input_a.grad = grad_a
        else:
            input_a.grad += grad_a

        if input_b.grad is None:
            input_b.grad = grad_b
        else:
            input_b.grad += grad_b

    requires_grad = (
        (input_a.requires_grad or input_b.requires_grad)
        and Tensor.build_graph_enabled()
    )
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_swiglu_backward if requires_grad else None,
        grad_fn_name="<SwiGLUBackward>" if requires_grad else None,
        device=input_a.device, 
        dtype=input_a.dtype
    )
    
    if requires_grad:
        out._add_parents(input_a, input_b)
    
    return out

def swiglu(input_a, input_b, auto=False, fused=False):
    if auto:
        return auto_swiglu(input)
    else:    
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_swiglu if _use_fused else manual_swiglu
        if fused and op is manual_swiglu:
            CHECKS.warn_triton_missing()
        return op(input_a, input_b)
