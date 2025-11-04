from .tensor import Tensor

### Functional Access to Non-Dunder Methods ###
def transpose(input, dim1, dim2):
    return input.transpose(dim1, dim2)

def permute(input, *dims):
    return input.permute(dims)

def reshape(input, *shape):
    return input.reshape(shape)

def exp(input):
    return input.exp()

def log(input):
    return input.log()

def sum(input, dim=None, keepdims=False):
    return input.sum(dim, keepdims)

def cumsum(input, dim=None):
    return input.cumsum(dim)

def mean(input, dim=None, keepdims=False):
    return input.mean(dim, keepdims)

def var(input, dim=None, keepdims=False):
    return input.var(dim, keepdims)

def max(input, dim=None, keepdims=False):
    return input.max(dim, keepdims)

def argmax(input, dim=None):
    return input.argmax(dim)

def masked_fill(input, mask, value):
    return input.masked_fill(mask, value)

def abs(input):
    return input.abs()

def clamp(input, min, max):
    return input.clamp(min, max)

def sqrt(input):
    return input.sqrt()

def sin(input):
    return input.sin()

def cos(input):
    return input.cos()

def tan(input):
    return input.tan()

def chunk(input, chunks, dim=0):
    return input.chunk(chunks=chunks, dim=dim)

### MultiTensor Ops ###
def concatenate(tensors, dim=0):

    if len(tensors) == 0:
        raise ValueError("concatenate() expects a non-empty list of Tensors")
    
    xp = tensors[0].xp
    device = tensors[0].device
    requires_grad = any(t.requires_grad for t in tensors) and Tensor.build_graph_enabled()

    tensor_list = [t.data for t in tensors]
    out_data = xp.concatenate(tensor_list, axis=dim)

    ### For backward pass we need the sizes along the concat dim ###
    sizes = [t.shape[dim] for t in tensors]

    def _concat_backward(out_grad):
        
        ### Loop through each slice ###
        offset = 0
        for t, size in zip(tensors, sizes):
            
            ### Check if that slice had grads ###
            if t.requires_grad:

                ### No slice on all dimension in our tensor ###
                grad_idx = [slice(None)] * out_grad.ndim

                ### Get slice for the one dim we did concat on ###
                grad_idx[dim] = slice(offset, offset + size)
                grad_chunk = out_grad[tuple(grad_idx)]

                if t.grad is None:
                    t.grad = grad_chunk
                else:
                    t.grad += grad_chunk

            offset += size

    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_concat_backward if requires_grad else None,
        grad_fn_name="<ConcatBackward>" if requires_grad else None,
        device=device,
    )

    if requires_grad:
        out._add_parents(*tensors)

    return out

def stack(tensors, dim=0):
    """
    Basically like concat, but adds a new dimension
    """
    if len(tensors) == 0:
        raise ValueError("stack() expects a non-empty list of tensors")

    # Ensure consistency
    xp = tensors[0].xp
    device = tensors[0].device
    requires_grad = any(t.requires_grad for t in tensors) and Tensor.build_graph_enabled()

    # Extract raw arrays
    data_list = [t.data for t in tensors]
    out_data = xp.stack(data_list, axis=dim)

    def _stack_backward(out_grad):

        ### if our inputs were (4,30), (4,30), (4,30), etc...
        ### then when we stack (if we stacked on the first dim )
        ### we would have (N x 4 x 30). So when doing backwards, 
        ### we just need to index that specific dim and everything 
        ### else is the same!

        for i, t in enumerate(tensors):
            if t.requires_grad:
                grad_idx = [slice(None)] * out_grad.ndim
                grad_idx[dim] = i

                grad_chunk = out_grad[tuple(grad_idx)]

                if t.grad is None:
                    t.grad = grad_chunk
                else:
                    t.grad += grad_chunk

    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_stack_backward if requires_grad else None,
        grad_fn_name="<StackBackward>" if requires_grad else None,
        device=device,
    )

    if requires_grad:
        out._add_parents(*tensors)

    return out
