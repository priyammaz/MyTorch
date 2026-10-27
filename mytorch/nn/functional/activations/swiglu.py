import cupy as cp
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

def fused_swiglu_packed(input):
    """
    SwiGLU on a packed projection: input = [a, b] along the last dim, out = silu(a) * b.
    """
    x = get_inner_inner_array(input)
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"packed swiglu needs an even last dim, got {x.shape[-1]}")
    hidden = x.shape[-1] // 2

    ### Contiguous rows let the kernel read (and later overwrite) both halves in place ###
    buffer = x if x.flags.c_contiguous else cp.ascontiguousarray(x)
    a, b, output = fused_swiglu_forward(buffer[..., :hidden], buffer[..., hidden:])

    def _swiglu_packed_backward(output_grad):
        if not input.requires_grad:
            return
        ### Writes grad_a / grad_b into buffer's two halves, so buffer now holds input's full gradient ###
        fused_swiglu_backward(output_grad, a, b)
        grad = buffer if buffer.shape == tuple(input.shape) else buffer.reshape(input.shape)
        if input.grad is None:
            input.grad = grad
        else:
            input.grad += grad

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_swiglu_packed_backward if requires_grad else None,
        grad_fn_name="<SwiGLUBackward>" if requires_grad else None,
        device=input.device,
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def swiglu(input_a, input_b=None, auto=False, fused=False):
    """
    swiglu(a, b) = silu(a) * b. With input_b=None, input_a is the packed projection [a, b] along the last dim
    (the fused version then avoids the chunk copies entirely).
    """
    _use_fused = (not auto) and ((fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED)
    if input_b is None:
        if _use_fused:
            return fused_swiglu_packed(input_a)
        input_a, input_b = input_a.chunk(2, dim=-1)
    if auto:
        return auto_swiglu(input_a, input_b)
    op = fused_swiglu if _use_fused else manual_swiglu
    if fused and op is manual_swiglu:
        CHECKS.warn_triton_missing()
    return op(input_a, input_b)
