from mytorch import Tensor

def batchnorm(input, weight, bias, 
              running_mean, running_var, momentum=0.1, 
              eps=1e-5, training=True):
    
    """
    BatchNorm for input of shape (N, C, *), normalizing per-channel.

    gamma: (C,)
    beta: (C,)
    running_mean: (C,)
    running_var: (C,)
    """
    
    ###################
    ### MANUAL GRAD ###
    ###################

    ### Get Backend ###
    xp = input.xp 

    N, C, *dims = input.shape
    reshaped = len(dims) > 0

    input_xp = input.data 
    weight_xp = weight.data.reshape(1,C,1)
    bias_xp = bias.data.reshape(1,C,1)

    ### Flatten Spatal Dims ###
    x = input_xp.reshape(N,C,-1)

    if training:
        mean = xp.mean(x, axis=(0, 2), keepdims=True)
        var = xp.var(x, axis=(0, 2), keepdims=True)
        running_mean.data[:] = (1 - momentum) * running_mean.data + momentum * mean.squeeze()
        running_var.data[:] = (1 - momentum) * running_var.data + momentum * var.squeeze()

    else:
        mean = running_mean.data.reshape(1, C, 1)
        var = running_var.data.reshape(1, C, 1)

    inv_std = xp.reciprocal(xp.sqrt(var + eps))
    norm_x = (x - mean) * inv_std
    out_data = norm_x * weight_xp + bias_xp

    if reshaped:
        out_data = out_data.reshape(N,C,*dims)

    def _batchnorm_backward(grad_output):

        ### Reshape Grad from (N,C,*) to (N,C,-1) ###
        grad_output = grad_output.reshape(N,C,-1)

        if weight.requires_grad:
            grad_gamma = xp.sum(grad_output * norm_x, axis=(0, 2))
            if weight.grad is None:
                weight.grad = grad_gamma
            else:
                weight.grad += grad_gamma

        if bias.requires_grad:
            grad_beta = xp.sum(grad_output, axis=(0, 2))
            if bias.grad is None:
                bias.grad = grad_beta
            else:
                bias.grad += grad_beta

        if input.requires_grad:
            grad_norm = grad_output * weight_xp

            mean_grad = xp.mean(grad_norm, axis=(0,2), keepdims=True)
            mean_norm_grad = xp.mean(grad_norm * norm_x, axis=(0,2), keepdims=True)
            grad_input = (grad_norm - mean_grad - norm_x * mean_norm_grad) * inv_std

            ### Put Back into Original Shape ###
            if reshaped:
                grad_input = grad_input.reshape(N,C,*dims)
            else:
                grad_input = grad_input.reshape(N,C)
            
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input

    requires_grad = input.requires_grad or weight.requires_grad or bias.requires_grad
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_batchnorm_backward if requires_grad else None,
        grad_fn_name="<BatchNormBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)
    
    return output