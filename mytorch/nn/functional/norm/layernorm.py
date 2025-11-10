import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_layernorm_forward, fused_layernorm_backward

def reshape_for_layernorm(x):
    reshaped = False
    *dims, embed_dim = x.shape

    ### If we have more than 1 dim, we have to flatten ###
    if len(dims) > 1:
        reshaped = True

    x = x.reshape(-1, embed_dim)

    return x, dims, reshaped

def auto_layernorm(input, weight, bias, eps=1e-5, *args):

    input, dims, reshaped_flag = reshape_for_layernorm(input)
    embed_dim = input.shape[-1]

    var_x = (input.var(dim=-1, keepdims=True) + eps)
    norm_x = (input - input.mean(dim=-1, keepdims=True)) / var_x**0.5
    scale_shifted_x = norm_x * weight.reshape(1,-1) 
    
    if bias:
        scale_shifted_x = scale_shifted_x + bias.reshape(1,-1)

    if reshaped_flag:
        scale_shifted_x = scale_shifted_x.reshape(*dims, embed_dim)

    return scale_shifted_x

def manual_layernorm(input, weight, bias, eps=1e-5, *args):
    
    input, dims, reshaped_flag = reshape_for_layernorm(input)
    embed_dim = input.shape[-1]

    input_arr = get_inner_array(input)
    weight_arr = get_inner_array(weight)
    beta_arr = get_inner_array(bias) if bias is not None else None

    ### Compute Mean and Var Along Last Dimension ###
    mean = np.mean(input_arr, axis=-1, keepdims=True)
    var = np.var(input_arr, axis=-1, keepdims=True)
    inv_std = np.reciprocal(np.sqrt(var + eps))

    ### Store copy of x_hat for the input backward ###
    x_hat = (input_arr - mean) * inv_std
    
    output = np.multiply(x_hat, weight_arr.reshape(1,-1))
    if beta_arr is not None:
        output += beta_arr.reshape(1,-1)

    ### Reshape Back if Needed ###
    if reshaped_flag:
        output = output.reshape(*dims, embed_dim)

    def _layernorm_backward(grad_output):
        
        ### Reshape Grad Output as its currently (*, I) ###
        if reshaped_flag:
            grad_output = grad_output.reshape(-1, embed_dim)

        if weight.requires_grad:
            # y = x_hat * gamma + beta
            # dL/dgamma = dL/dy * dy/dgamma = grad_output * x_hat
            # sum up grads over the batch dim
            grad_gamma = np.sum(grad_output * x_hat, axis=0)

            if weight.grad is None:
                weight.grad = grad_gamma
            else:
                weight.grad += grad_gamma
        
        if bias is not None:
            if bias.requires_grad:
                # y = x_hat * gamma + beta
                # dL/dbeta = dL/dy * dy/dbeta = grad_output * 1
                # sum up grads over the batch dim
                grad_beta = np.sum(grad_output, axis=0)
                
                if bias.grad is None:
                    bias.grad = grad_beta
                else:
                    bias.grad += grad_beta

        if input.requires_grad:
            # y = x_hat * gamma + beta
            # where x_hat = (x - mu) / (var + eps)
            # dL/dx = dL/dy * dy/dx_hat * dx_hat / dx
            # = inv_std * (grad_output * gamma - mean(grad_output*gamma) - x_hat*mean(grad_output * weight_arr * x_hat))
            # sum up grads over the batch dim
            dx_hat = grad_output * weight_arr
            mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
            mean_mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
            grad_input = inv_std * (dx_hat - mean_dx_hat - x_hat * mean_mean_dx_hat_x_hat) 

            ### Put Back into Original Shape ###
            if reshaped_flag:
                grad_input = grad_input.reshape(*dims, embed_dim)

            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input

    requires_grad = input.requires_grad or weight.requires_grad or \
                            (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output, 
        requires_grad=requires_grad,
        grad_fn=_layernorm_backward if requires_grad else None, 
        grad_fn_name="<LayerNormBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)

    return output

def fused_layernorm(input, weight, bias, eps=1e-5, training=True):
    
    input, dims, reshaped_flag = reshape_for_layernorm(input)
    embed_dim = input.shape[-1]

    input_arr = get_inner_inner_array(input)
    weight_arr = get_inner_inner_array(weight)
    bias_arr = get_inner_inner_array(bias) if bias is not None else None       

    outputs = fused_layernorm_forward(input_arr,
                                      gamma=weight_arr, 
                                      beta=bias_arr, 
                                      eps=eps, 
                                      training=training)

    ### during training we return intermediate tensors ###
    if training:
        output, x_hat, inv_var = outputs
    else:
        output = outputs

    ### Return y back to (B x S x E) ###
    if reshaped_flag:
        output = output.reshape(*dims, -1)

    def _layernorm_backward(grad_output):
        
        ### Reshape grad back to (*xE) ###
        grad_flat = grad_output.reshape(-1, embed_dim)

        grads = fused_layernorm_backward(x_hat=x_hat,
                                         inv_var=inv_var,
                                         dy=grad_flat,
                                         gamma=weight_arr, 
                                         bias=True if bias_arr is not None else False)

        if bias is not None:
            dx, dgamma, dbeta = grads
        else:
            dx, dgamma = grads

        ### Reshape dx back to original shape ###
        dx = dx.reshape(*dims, -1)

        ### Accumulate Grads ####
        if input.grad is None:
            input.grad = dx
        else:
            input.grad += dx

        if weight.grad is None:
            weight.grad = dgamma
        else:
            weight.grad += dgamma
        
        if bias is not None:
            if bias.grad is None:
                bias.grad = dbeta
            else:
                bias.grad += dbeta

    requires_grad = input.requires_grad or weight.requires_grad or \
                        (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output, 
        requires_grad=requires_grad,
        grad_fn=_layernorm_backward if requires_grad else None, 
        grad_fn_name="<LayerNormBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)

    return output

def layernorm(input, weight, bias, eps=1e-5, auto=False, training=True, fused=False):

    if auto:
        return auto_layernorm(input, weight, bias, eps)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_layernorm if _use_fused else manual_layernorm
        if fused and op is manual_layernorm:
            CHECKS.warn_triton_missing()
        return op(input, weight, bias, eps, training)