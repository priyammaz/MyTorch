def embedding(indices, weight, fused=False):
    """
    An embedding lookup is just weight[indices], so we reuse Tensor.__getitem__
    """
    return weight[indices]
