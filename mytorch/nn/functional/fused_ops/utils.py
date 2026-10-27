MAX_FUSED_SIZE = 2**16
def calc_num_warps(block_size):
    if block_size > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since block_size = {block_size} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   block_size >= 32768: num_warps = 32
    elif block_size >=  8192: num_warps = 16
    elif block_size >=  2048: num_warps = 8
    return num_warps

### DTYPE FLAGS ###
### Our kernels cast raw pointers to the right element type with a constexpr flag ###
### 0 = float32, 1 = float16, 2 = bfloat16. Works for both torch and cupy dtypes ###
_DTYPE_FLAGS = {"float32": 0, "float16": 1, "bfloat16": 2}

def get_dtype_flag(dtype):
    name = str(dtype).replace("torch.", "")
    if name not in _DTYPE_FLAGS:
        raise TypeError(f"Fused kernels only support {list(_DTYPE_FLAGS)}, got {name}")
    return _DTYPE_FLAGS[name]

def to_torch_dtype(dtype):
    import torch
    get_dtype_flag(dtype)
    return getattr(torch, str(dtype).replace("torch.", ""))

class TritonPtr:
    """
    Pass a cupy array to a Triton kernel as a typed pointer. Triton treats anything with .data_ptr() and .dtype
    like a torch tensor: it compiles a typed pointer, this allows us to avoid the entire torch dependency and
    torch DLPack when using Triton
    """
    __slots__ = ("_ptr", "dtype")

    def __init__(self, arr):
        self._ptr = arr.data.ptr
        self.dtype = arr.dtype

    def data_ptr(self):
        return self._ptr
