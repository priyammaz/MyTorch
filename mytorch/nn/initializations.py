import math
import numpy as np
from .. import zeros_like, ones_like, randn_like, rand_like
from ..tensor import Tensor

### RETURN INITS ###
def zeros(tensor):
    return zeros_like(tensor)

def ones(tensor):
    return ones_like(tensor)

def normal(tensor, mean=0.0, std=1.0):
    arr = randn_like(tensor).data * std + mean
    return Tensor(arr, device=tensor.device, dtype=tensor.dtype, requires_grad=tensor.requires_grad)

def uniform(tensor, low=0.0, high=1.0):
    arr = rand_like(tensor).data * (high - low) + low
    return Tensor(arr, device=tensor.device, dtype=tensor.dtype, requires_grad=tensor.requires_grad)

def xavier_uniform(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_out(tensor.shape)
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform(tensor, -limit, limit)

def xavier_normal(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_out(tensor.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal(tensor, 0.0, std)

def kaiming_uniform(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_fan(tensor.shape, mode)
    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(fan)
    return uniform(tensor, -bound, bound)

def kaiming_normal(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_fan(tensor.shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal(tensor, 0.0, std)

### INPLACE INITS ###
def zeros_(tensor):
    tensor.data = zeros_like(tensor).data

def ones_(tensor):
    tensor.data = ones_like(tensor).data

def normal_(tensor, mean=0.0, std=1.0):
    arr = randn_like(tensor).data * std + mean
    tensor.data = arr

def uniform_(tensor, low=0.0, high=1.0):
    arr = rand_like(tensor).data * (high - low) + low
    tensor.data = arr

def xavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_out(tensor.shape)
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    uniform_(tensor, -limit, limit)

def xavier_normal_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_out(tensor.shape)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    normal_(tensor, 0.0, std)

def kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"):
    fan = _calculate_fan(tensor.shape, mode)
    gain = calculate_gain(nonlinearity, a)
    bound = math.sqrt(3.0) * gain / math.sqrt(fan)
    uniform_(tensor, -bound, bound)

def _calculate_fan_in_out(shape):
    if len(shape) == 2:  # linear layer weight
        fan_in, fan_out = shape[1], shape[0]
    elif len(shape) in {3, 4, 5}:  # conv weights
        receptive_field_size = np.prod(shape[2:])
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    else:
        fan_in = fan_out = 1
    return fan_in, fan_out

def _calculate_fan(shape, mode="fan_in"):
    fan_in, fan_out = _calculate_fan_in_out(shape)
    return fan_in if mode == "fan_in" else fan_out

def calculate_gain(nonlinearity, param=None):
    if nonlinearity == "linear" or nonlinearity == "conv1d":
        return 1.0
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + param**2)) if param is not None else math.sqrt(2.0)
    return 1.0