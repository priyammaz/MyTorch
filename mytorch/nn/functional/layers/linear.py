"""
Forward:
    y = x @ W.T + b
    x: (B, I)        input
    W: (O, I)        weight
    b: (O,)          bias (broadcasted)
    y: (B, O)        output

Backward (given grad_y = ∂L/∂y ∈ (B, O)):

    dx = grad_y @ W
          (B, O) @ (O, I) -> (B, I)

    dW = (grad_y.T @ x).T
          (O, B) @ (B, I) -> (O, I)
        implemented as: np.matmul(grad_y.T, x).T
        or equivalently: np.matmul(x.T, grad_y).T

    db = grad_y.sum(axis=0)
          (B, O) -> (O,)
"""

import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_linear_forward, fused_grouped_matmul, fused_activation_backward

def reshape_for_linear(x):

    """
    Linear layers can pass in multidim tensors, for example
    if our data is:
    
    [A x B x C x I], we reshape to [A*B*C x I], perform the 
    Matmul, and return back to the original shape
    """
    reshaped = False
    *dims, in_features = x.shape

    if len(dims) > 1:
        reshaped = True

    if reshaped:
        x = x.reshape(np.prod(dims), in_features)
   
    return x, dims, reshaped

def auto_linear(input, weight, bias=None, *args):

    """
    auto_linear will leverage our autograd system
    to perform the operation
    """

    input, dims, reshaped_flag = reshape_for_linear(input)
    out_features = weight.shape[-1]

    output = input @ weight.transpose(-1,-2)
    if bias is not None:
        output = output + bias.reshape(1,-1)
    
    if reshaped_flag:
        output = output.reshape(*dims, out_features)

    return output

def manual_linear(input, weight, bias=None, *args):

    """
    manual_linear will manually pass the gradients 
    backward for this operation, so we no longer 
    are using our own autograd system
    """

    input, dims, reshaped_flag = reshape_for_linear(input)
    out_features, in_features = weight.shape

    ### Manual Grad Means we Unpack our Tensor to the underlying Array ###
    input_arr = get_inner_array(input)
    weight_arr = get_inner_array(weight).T
    if bias is not None:
        bias_arr = get_inner_array(bias)

    output = np.matmul(input_arr, weight_arr)
    if bias is not None:
        output += bias_arr.reshape(1,-1)

    if reshaped_flag:
        output = output.reshape(*dims, out_features)

    ### We will capture our variables as a closure ###
    def _linear_backward(grad_output):
            
        ### Our gradients are coming in the shape of (*, O) ###
        ### But our operation happened in the shape of (N x O) ###
        ### So change our grad_output shape to that by flattening ###
        if reshaped_flag:
            grad_output = grad_output.reshape(-1, out_features)

        ### Standard Weight Update formula ###
        if weight.requires_grad:
            grad_W = np.matmul(input_arr.T, grad_output)

            if weight.grad is None:
                weight.grad = grad_W.T
            else:
                weight.grad += grad_W.T
            grad_W = None

        ### Standard Bias Update Formula ###
        if bias is not None and bias.requires_grad:
            grad_b = grad_output.sum(axis=0)
            if bias.grad is None:
                bias.grad = grad_b
            else:
                bias.grad += grad_b
            grad_b = None
        
        ### Grad to Input ###
        if input.requires_grad:
            grad_input = np.matmul(grad_output, weight_arr.T)

            ### Reshape grad_input back to input feature shape (* x I) ###
            grad_input = grad_input.reshape(*dims, in_features)
            
            if input.grad is None:
                input.grad = grad_input
            else:   
                input.grad += grad_input
            grad_input = None
            
    requires_grad = input.requires_grad or weight.requires_grad or \
                            (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output, 
        requires_grad=requires_grad,
        grad_fn=_linear_backward if requires_grad else None,
        grad_fn_name="<LinearBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output

def fused_linear(input, weight, bias=None, act_func=None):

    """
    fused_linear will use triton kernels to do 
    the same operation as manual_linear
    """
    input, dims, reshaped_flag = reshape_for_linear(input)
    out_features, in_features = weight.shape

    ### Fused Means we need the inner inner array (the cupy array) ###
    input_arr = get_inner_inner_array(input)
    weight_arr = get_inner_inner_array(weight).T
    if bias is not None:
        bias_arr = get_inner_inner_array(bias)
    
    ### Perform the Fused Forward Pass ###
    outputs = fused_linear_forward(
        input_arr, weight_arr, bias_arr if bias is not None else None, act_func=act_func
    )

    ### If we use an activation function we get both the pre/post activation for backprop ###
    if act_func is not None:
        preact_output, output = outputs
    else:
        output, _ = outputs
 
    if reshaped_flag:
        output = output.reshape(*dims, out_features)

    ### We will capture our variables as a closure ###
    def _linear_backward(grad_output):

        ### If we had an activation func in the forward pass, we first need to backprop
        ### Through it in the backward pass ###
        if act_func is not None: 
            grad_output = fused_activation_backward(preact_output, grad_output, act_func=act_func)

        ### Our gradients are coming in the shape of (*, O) ###
        ### But our operation happened in the shape of (N x O) ###
        ### So change our grad_output shape to that by flattening ###
        if reshaped_flag:
            grad_output = grad_output.reshape(-1, out_features)

        ### Standard Weight Update formula ###
        if weight.requires_grad:
            grad_W = fused_grouped_matmul(input_arr.T, grad_output)
            if weight.grad is None:
                weight.grad = grad_W.T
            else:
                weight.grad += grad_W.T
            grad_W = None

        ### Standard Bias Update Formula ###
        if bias is not None and bias.requires_grad:
            grad_b = grad_output.sum(axis=0)
            if bias.grad is None:
                bias.grad = grad_b
            else:
                bias.grad += grad_b
            grad_b = None
        
        ### Grad to Input ###
        if input.requires_grad:
            grad_input = fused_grouped_matmul(grad_output, weight_arr.T)

            ### Reshape grad_input back to input feature shape (* x I) ###
            grad_input = grad_input.reshape(*dims, in_features)
            
            if input.grad is None:
                input.grad = grad_input
            else:   
                input.grad += grad_input
            grad_input = None

    requires_grad = input.requires_grad or weight.requires_grad or \
                            (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output, 
        requires_grad=requires_grad,
        grad_fn=_linear_backward if requires_grad else None,
        grad_fn_name="<LinearBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output 

def linear(input, weight, bias=None, auto=False, fused=False, act_func=None):

    """
    This toggles between the different methods implemented!
    """
    _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
    if not _use_fused and act_func is not None:
        raise Exception("linear layer act_func only supported for fused operations!")
    if auto:
        if fused:
            raise Exception("Auto methods cannot use fused activations")
        if act_func is not None:
            raise Exception("Auto methods cannot fuse activation functions")
        return auto_linear(input, weight, bias)
    else:
        op = fused_linear if _use_fused else manual_linear
        if fused and op is manual_linear:
            CHECKS.warn_triton_missing()
        return op(input, weight, bias, act_func) 