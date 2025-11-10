import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_inner_array
from ..fused_ops import fused_activation_forward, fused_activation_backward

def manual_gelu(x):
    
    """
    gelu as described in https://arxiv.org/pdf/2305.12073

    Forward method is Equation 24
    Backward methdod is Equation 42-43
    """

    data = x.data

    # Constants
    sqrt_2_over_pi = 0.7978845 # xp.sqrt(2 / xp.pi).astype(x.data.dtype)
    coeff = 0.44715

    #inner = sqrt_2_over_pi * (x + coeff * x^3)
    x_squared = np.power(data, 2)
    x_cubed = x_squared * data

    inner = sqrt_2_over_pi * (data + coeff * x_cubed)

    ### Tanh out = tanh(inner) ###
    tanh_out = np.tanh(inner)
    out_data = 0.5 * data * (1.0 + tanh_out)

    # Backward
    def _gelu_backward(grad_output):

        if x.requires_grad:
    
            inner_grad = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_squared)

            # derivative of GELU approximation (sech^2(x) = 1 - tanh^2(x))
            sech2 = 1 - np.power(tanh_out, 2)  # derivative of tanh

            grad_input = 0.5 * (1.0 + tanh_out + data * sech2 * inner_grad) * grad_output

            if x.grad is None:
                x.grad = grad_input
            else:
                x.grad += grad_input

    requires_grad = x.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_gelu_backward if requires_grad else None,
        grad_fn_name="<GELUBackward>" if requires_grad else None
    )

    if requires_grad:
        out._add_parents(x)

    return out

def fused_gelu(input):

    input_arr = get_inner_inner_array(input)
    output = fused_activation_forward(input_arr, act_func="gelu")

    def _gelu_backward(output_grad):
        output_grad = fused_activation_backward(input_arr, output_grad, act_func="gelu")
        if input.grad is None:
                input.grad = output_grad
        else:
            input.grad += output_grad

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_gelu_backward if requires_grad else None,
        grad_fn_name="<GELUBackward>" if requires_grad else None,
        device=input.device, 
        dtype=input.dtype
    )

    if requires_grad:
        out._add_parents(input)

    return out

def gelu(input, fused=False):
    _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
    op = fused_gelu if _use_fused else manual_gelu
    if fused and op is manual_gelu:
        CHECKS.warn_triton_missing()
    return op(input)
