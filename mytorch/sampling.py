from .ops import sum, cumsum
from .tensor import rand

def multinomial(probs, num_samples=1):
    """
    Sample indices from categorical distributions.
    probs: shape (..., n_classes)
    returns: shape (..., num_samples)
    """
    
    ### Cast to float32 otherwise there will be precision errors! ###
    probs = probs.astype("float32")
    probs = probs / sum(probs, dim=-1, keepdims=True)
    cdf = cumsum(probs, dim=-1)

    ### CDF must add to 1, otherwise it doesnt make sense. The reason for the 
    ### float32 cast was if in float16 it can give values like 0.99995 which is 
    ### undesireable and leads to incorrect indexes !
    assert cdf[..., -1] == 1.0
    
    u = rand((*probs.shape[:-1], num_samples), device=probs.device).unsqueeze(-1)
    sampled = sum(u > cdf.unsqueeze(-2), dim=-1)
    return sampled