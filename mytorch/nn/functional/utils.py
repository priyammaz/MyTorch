
"""
We have a few layers of abstractions. 

Tensor wraps Array in .data
Array wraps numpy/cupy in ._array

This method (given an Array) will return the underlying 
ndarray (cupy or numpy). We mainly care about this for 
Cupy as that is what our triton kernels will use!
"""

from ..._array import Array
from ...tensor import Tensor

def get_inner_array(arr):
    if isinstance(arr, Array):
        return arr
    elif isinstance(arr, Tensor):
        if hasattr(arr, "data"):
            return arr.data
        else:
            raise Exception("This Tensor object doesn't have a .data attribute!")

def get_inner_inner_array(arr):
    if isinstance(arr, Tensor):
        arr = get_inner_array(arr)
    if hasattr(arr, "_array"):
        return arr._array
    else:
        raise Exception("This Array object doesn't have a ._array attribute!")
