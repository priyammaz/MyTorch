"""
RMSNorm formula: y = (x / RMS(x)) * weight
where RMS(x) = sqrt(mean(x^2) + eps)

Backward formula:
dx = (1/RMS) * [dy * w - (1/N) * (1/RMS^2) * ((dy * w) dot x) * x]
dw = sum(dy * (x / RMS)) over batch dimension
"""

import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_rmsnorm_forward, fused_rmsnorm_backward

def reshape_for_rmsnorm(x):
    reshaped = False
    *dims, embed_dim = x.shape

    ### If we have more than 1 dim, we have to flatten ###
    if len(dims) > 1:
        reshaped = True

    x = x.reshape(-1, embed_dim)

    return x, dims, reshaped

def auto_rmsnorm(input, weight, eps=1e-5, *args):
    
    input, dims, reshape_flag = reshape_for_rmsnorm(input)
    embed_dim = input.shape[-1]

    rms = ((input.pow(2)).mean(dim=-1, keepdims=True) + eps) ** 0.5
    norm_x = input / rms

    if weight is not None:
        scaled_x = norm_x * weight.reshape(1,-1)
    else:
        scaled_x = norm_x
    
    if reshape_flag:
        scaled_x = scaled_x.reshape(*dims, embed_dim)
    
    return scaled_x

def manual_rmsnorm(input, weight, eps=1e-5, *args):

    input, dims, reshaped_flag = reshape_for_rmsnorm(input)
    embed_dim = input.shape[-1]

    input_arr = get_inner_array(input)
    weight_arr = get_inner_array(weight) if weight is not None else None

    ### Compute RMS ###
    mean_square = np.mean(input_arr * input_arr, axis=-1, keepdims=True)
    rms = np.sqrt(mean_square + eps)
    rstd = np.reciprocal(rms)
    
    ### Normalize ###
    x_hat = input_arr * rstd

    ### Scale by Weight ###
    if weight_arr is not None:
        output = x_hat * weight_arr.reshape(1,-1)
    else:
        output = x_hat

    if reshaped_flag:
        output = output.reshape(*dims, embed_dim)

    def _rmsnorm_backward(grad_output):

        if reshaped_flag:
            grad_output = grad_output.reshape(-1, embed_dim)
        
        if weight is not None and weight.requires_grad:
            grad_weight = np.sum(grad_output * x_hat, axis=0)

            if weight.grad is None:
                weight.grad = grad_weight
            else:
                weight.grad += grad_weight

        if input.requires_grad:
            # Compute m = dy * w
            m = grad_output * weight_arr if weight_arr is not None else grad_output
            
            # dx = rstd * [m - (1/N) * rstd^2 * sum(m * x) * x]
            sum_m_x = np.sum(m * input_arr, axis=-1, keepdims=True)
            grad_input = rstd * (m - (1.0/embed_dim) * (rstd * rstd * sum_m_x) * input_arr)

            if reshaped_flag:
                grad_input = grad_input.reshape(*dims, embed_dim)
            
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input

    requires_grad = input.requires_grad or \
                    (weight is not None and weight.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    
    output = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_rmsnorm_backward if requires_grad else None,
        grad_fn_name="<RMSNormBackward>" if requires_grad else None
    )
    
    if requires_grad:
        output._add_parents(input, weight)
    
    return output

def fused_rmsnorm(input, weight, eps=1e-5, training=True):

    """
    RMSNorm with optional weight (gamma)
    """
  
    input, dims, reshaped_flag = reshape_for_rmsnorm(input)
    embed_dim = input.shape[-1]

    input_arr  = get_inner_inner_array(input)
    weight_arr = get_inner_inner_array(weight) if weight is not None else None

    outputs = fused_rmsnorm_forward(input_arr,
                                    gamma=weight_arr,
                                    eps=eps,
                                    training=training)

    if training:
        output, rstd = outputs
    else:
        output = outputs

    if reshaped_flag:
        output = output.reshape(*dims, -1)

    def _rmsnorm_backward(grad_output):

        if reshaped_flag:
            grad_output = grad_output.reshape(-1, embed_dim)

        dx, dgamma = fused_rmsnorm_backward(x=input_arr,
                                            rstd=rstd, 
                                            dy=grad_output,
                                            gamma=weight_arr)
        if reshaped_flag:
            dx = dx.reshape(*dims, -1)

        if input.grad is None:
            input.grad = dx
        else:
            input.grad += dx

        if weight is not None:
            if weight.grad is None:
                weight.grad = dgamma
            else:
                weight.grad += dgamma

    requires_grad = (input.requires_grad or
                     (weight is not None and weight.requires_grad))
    requires_grad = requires_grad and Tensor.build_graph_enabled()

    output_tensor = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_rmsnorm_backward if requires_grad else None,
        grad_fn_name="<RMSNormBackward>" if requires_grad else None
    )

    if requires_grad:
        output_tensor._add_parents(input, weight)

    return output_tensor

def rmsnorm(input, weight, eps=1e-5, auto=False, training=True, fused=False):
    if auto:
        return auto_rmsnorm(input, weight, eps)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_rmsnorm if _use_fused else manual_rmsnorm
        if fused and op is manual_rmsnorm:
            CHECKS.warn_triton_missing()
        return op(input, weight, eps, training)