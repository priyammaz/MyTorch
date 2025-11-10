from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_conv1d_forward, fused_conv1d_backward

def manual_conv1d(input, weight, bias=None, stride=1, padding=0, **args):

    ### Get Backend ###
    xp = input.xp

    ### Get Input/Output Shapes ###
    B, C_in,  L_in = input.data.shape
    C_out, _, K = weight.data.shape
    S,P = stride, padding

    L_out = (L_in + 2*P - K)//S + 1

    ### Pad Data If Padding is set ###
    if P > 0:
        x_padded = xp.pad(input.data, ((0,0), (0,0), (P,P)), mode='constant')
    else:
        x_padded = input.data

    ### Use stride tricks for efficient im2col ###
    shape = (B, C_in, K, L_out)
    strides = (
        x_padded.strides[0], # Number of bits to move to get to next batch
        x_padded.strides[1], # Number of bits to move to get to next channel
        x_padded.strides[2], # Number of bits to move to get to next row in kernel
        S*x_padded.strides[2], # Number of bits to move to get to next col in kernel
    )

    ### Grab Strided View of our Data (no extra copy needed!) ###
    cols = xp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

    cols_flat = cols.reshape(B*L_out, -1)
    weights_flat = weight.data.reshape(C_out, -1).T

    ### Forward ###
    output = xp.matmul(cols_flat, weights_flat)
    if bias is not None:
        output += bias.data

    #### Reshape back to (B x C_out x H_out x W_out) ###
    output = output.reshape(B, L_out, C_out).transpose(0, 2, 1)

    def _conv1d_backward(grad_output):
        grad_output_flat = grad_output.transpose(0,2,1).reshape(B*L_out, C_out)

        # Gradient w.r.t. weight
        if weight.requires_grad:
            grad_W = xp.matmul(cols_flat.T, grad_output_flat)
            weight.grad = grad_W.T.reshape(C_out, C_in, K) if weight.grad is None else weight.grad + grad_W.T.reshape(C_out, C_in, K)

        # Gradient w.r.t. bias
        if bias is not None and bias.requires_grad:
            grad_b = grad_output_flat.sum(axis=0)
            bias.grad = grad_b if bias.grad is None else bias.grad + grad_b

        # Gradient w.r.t. input
        if input.requires_grad:
            grad_cols_flat = xp.matmul(grad_output_flat, weights_flat.T)  # (B*L_out, C_in*K)
            grad_cols = grad_cols_flat.reshape(B, L_out, C_in*K).transpose(0,2,1)  # (B, C_in*K, L_out)

            grad_input = xp.zeros_like(x_padded)
            # Compute indices for scatter-add
            i0 = xp.repeat(xp.arange(K), C_in)
            k = xp.tile(xp.arange(C_in), K)
            i1 = S * xp.repeat(xp.arange(L_out), 1)

            i = i0.reshape(-1,1) + i1.reshape(1,-1)
            kk = xp.tile(k.reshape(-1,1), (1,L_out))

            # Flatten for batch
            bb = xp.repeat(xp.arange(B), C_in*K*L_out)
            ii = xp.tile(i.flatten(), B)
            kk_flat = xp.tile(kk.flatten(), B)
            vals = grad_cols.flatten()

            xp.add.at(grad_input, (bb, kk_flat, ii), vals)

            if P > 0:
                grad_input = grad_input[:, :, P:-P]

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
        grad_fn=_conv1d_backward if requires_grad else None,
        grad_fn_name="<Conv1dBackward>" if requires_grad else None
    )

    
    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output 

def fused_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1):

    ### Fused ops need raw cupy arrays not Array ###
    input_arr = get_inner_inner_array(input)
    weight_arr = get_inner_inner_array(weight)
    bias_arr = get_inner_inner_array(bias) if bias is not None else None

    ### Get Input/Output Shapes ###
    B, C_in,  L_in = input.data.shape
    C_out, _, K = weight.data.shape
    S,P = stride, padding

    L_out = (L_in + 2*P - K)//S + 1

    output = fused_conv1d_forward(input_arr, weight_arr, bias_arr, 
                                  stride, padding, dilation=dilation)
    def _conv2d_backward(grad_output):
        
        grads = fused_conv1d_backward(grad_output, input_arr, weight_arr, 
                                      bias_arr,
                                      L_in, K, 
                                      stride, padding, dilation=dilation)
        
        if bias is not None:
            dinput, dweight, dbias = grads
        else:
            dinput, dweight = grads

        ### Accumulate Grads ####
        if input.grad is None:
            input.grad = dinput
        else:
            input.grad += dinput

        if weight.grad is None:
            weight.grad = dweight
        else:
            weight.grad += dweight
        
        if bias is not None:
            if bias.grad is None:
                bias.grad = dbias
            else:
                bias.grad += dbias

    requires_grad = input.requires_grad or weight.requires_grad or \
                        (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_conv2d_backward if requires_grad else None,
        grad_fn_name="<Conv2dBackward>" if requires_grad else None
    )
    
    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output   

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, fused=False):

    """
    This toggles between the different methods implemented!
    """

    _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
    op = fused_conv1d if _use_fused else manual_conv1d
    if not _use_fused and dilation > 1:
        raise Exception("Non-Fused Conv1d does not support Dilations Greater than 1!")
    if fused and op is manual_conv1d:
        CHECKS.warn_triton_missing()
    return op(input, weight, bias, stride=stride, padding=padding, dilation=dilation) 