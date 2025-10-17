from .ops import sum, cumsum
from .tensor import rand

def multinomial(probs, num_samples=1):
    """
    Sample indices from categorical distributions.
    probs: shape (..., n_classes)
    returns: shape (..., num_samples)
    """
    probs = probs / sum(probs, dim=-1, keepdims=True)
    cdf = cumsum(probs, dim=-1)
    u = rand((*probs.shape[:-1], num_samples), device=probs.device).unsqueeze(-1)
    return sum(u > cdf.unsqueeze(-2), dim=-1)