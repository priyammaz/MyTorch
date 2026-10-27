import math
import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_inner_array
from ..fused_ops import fused_activation_forward, fused_activation_backward

### approximate="none" -> exact erf GELU, approximate="tanh" -> tanh approximation###
_GELU_APPROXIMATIONS = {"none": "gelu", "tanh": "gelu_tanh"}

def _check_approximate(approximate):
    if approximate not in _GELU_APPROXIMATIONS:
        raise ValueError(f"approximate argument must be either 'none' or 'tanh', got {approximate}")

def _erf(data):
    ### cupy has an elementwise erf kernel, numpy doesnt (without scipy) so vectorize math.erf on cpu ###
    if data.xp is np:
        return np.vectorize(math.erf, otypes=[np.float64])(data._array).astype(data.dtype)
    import cupyx.scipy.special
    return type(data)(cupyx.scipy.special.erf(data._array.astype("float32")).astype(data.dtype), device=data.device)

def manual_gelu_exact(x):

    """
    Exact gelu: x * Phi(x) where Phi is the standard normal CDF
    Backward: Phi(x) + x * phi(x) where phi is the standard normal PDF
    """

    data = x.data
    cdf = 0.5 * (1.0 + _erf(data * 0.7071067811865476))
    out_data = data * cdf

    def _gelu_backward(grad_output):

        if x.requires_grad:

            pdf = np.exp(-0.5 * data * data) * 0.3989422804014327
            grad_input = (cdf + data * pdf) * grad_output

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

def manual_gelu(x):
    
    """
    tanh approximation of gelu as described in https://arxiv.org/pdf/2305.12073

    Forward method is Equation 24
    Backward method is Equation 42-43
    """

    data = x.data

    # Constants
    sqrt_2_over_pi = 0.7978845608 # xp.sqrt(2 / xp.pi).astype(x.data.dtype)
    coeff = 0.044715

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

def fused_gelu(input, approximate="none"):

    act_func = _GELU_APPROXIMATIONS[approximate]
    input_arr = get_inner_inner_array(input)
    output = fused_activation_forward(input_arr, act_func=act_func)

    def _gelu_backward(output_grad):
        output_grad = fused_activation_backward(input_arr, output_grad, act_func=act_func)
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

def gelu(input, approximate="none", fused=False):
    """
    approximate: 'none' (exact, default) | 'tanh', same as torch.nn.functional.gelu
    """
    _check_approximate(approximate)
    _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
    if fused and not _use_fused:
        CHECKS.warn_triton_missing()
    if _use_fused:
        return fused_gelu(input, approximate)
    return manual_gelu_exact(input) if approximate == "none" else manual_gelu(input)
