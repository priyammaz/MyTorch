from mytorch import Tensor

def auto_dropout(input, dropout_p, training=True):
    if not training or dropout_p == 0.0:
        return input

    mask = (input.xp.random.random_sample(input.data.shape) >= dropout_p).astype(input.dtype, copy=False)
    ratio = 1 / (1 - dropout_p)
    mask *= ratio

    return input * mask

def manual_dropout(input, dropout_p, training=True):

    if not training or dropout_p == 0.0:
        return input

    ### Sample Mask ###
    mask = (input.xp.random.random_sample(input.data.shape) >= dropout_p).astype(input.dtype, copy=False)
    ratio = 1 / (1 - dropout_p)
    mask *= ratio

    out_data = input.data * mask
    
    # Backward function only needs the mask (not full input_tensor)
    def _dropout_backward(input_grad):
        if input.requires_grad:
            self_grad = input_grad * mask
            if input.grad is None:
                input.grad = self_grad
            else:
                input.grad += self_grad

    # Attach to computation graph if needed
    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_dropout_backward if requires_grad else None,
        grad_fn_name="<DropoutBackward>" if requires_grad else None,
    )

    if requires_grad:
        out._add_parents(input)

    return out

def dropout(input, dropout_p, training=True, auto=False):

    if auto:
        return auto_dropout(input, dropout_p, training)
    else:
        return manual_dropout(input, dropout_p, training)

    