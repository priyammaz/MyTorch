import cupy as cp

def clip_grad_norm_(params, max_norm):
    """
    The norm is computed over the norms of the individual gradients of all parameters, 
    as if the norms of the individual gradients were concatenated into a 
    single vector. Gradients are modified in-place.
    """

    params = list(params)

    ### Compute Total Norm across
    total_norm = 0.0
    for param in params:
        if hasattr(param, "grad") and param.grad is not None:
            total_norm += float(cp.linalg.norm(param.grad.reshape(-1), ord=2.0)) ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for param in params:
            if hasattr(param, "grad") and param.grad is not None:
                param.grad *= clip_coef