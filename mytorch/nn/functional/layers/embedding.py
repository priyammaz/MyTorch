import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_inner_array
from ..fused_ops import fused_embedding_forward, fused_embedding_backward

def auto_embedding(indices, weight):
    return weight[indices]

def fused_embedding(indices, weight):

    indices_arr = get_inner_inner_array(indices)
    weight_arr = get_inner_inner_array(weight)
    
    output = fused_embedding_forward(weight_arr, indices_arr)

    def _embedding_backward(grad_output):
        
        grads = fused_embedding_backward(grad_output, 
                                         weight_arr, 
                                         indices_arr)

        ### Accumulate Grads ####
        if weight.grad is None:
            weight.grad = grads
        else:
            weight.grad += grads
        
        grads = None

    requires_grad = weight.requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output, 
        requires_grad=requires_grad,
        grad_fn=_embedding_backward if requires_grad else None, 
        grad_fn_name="<EmbeddingBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(weight)

    return output

def embedding(indices, weight, fused=False):

    """
    This toggles between the different methods implemented!
    """
    _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
    op = fused_embedding if _use_fused else auto_embedding
    if fused and op is auto_embedding:
        CHECKS.warn_triton_missing()
    return op(indices, weight) 
