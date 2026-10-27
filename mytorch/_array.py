"""
Our goal here is to homogenize Numpy and Cupy. 
Cupy is awesome but doesnt have CPU support. 
Numpy is awesome but doesnt have GPU support. 

Instead of manually toggling between the two, 
we will create a new object Array, that can 
sit on the CPU or GPU, and all numpy methods 
will work on it!
"""

import warnings
import numpy as np
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    NUM_AVAIL_GPUS = cp.cuda.runtime.getDeviceCount()
except ImportError:
    cp = None
    CUDA_AVAILABLE = False
    NUM_AVAIL_GPUS = 0
    warnings.warn("Cupy not installed!!!")

### Registers the bfloat16 dtype with numpy so cupy (>=14) can use "bfloat16" ###
try:
    import ml_dtypes
    BFLOAT16_AVAILABLE = True
except ImportError:
    BFLOAT16_AVAILABLE = False

def _is_bf16(x):
    return str(getattr(x, "dtype", "")) == "bfloat16"

def _keep_bf16(result, inputs_bf16):
    ### cupy promotes bfloat16 to float32/float64 in some functions (np.var, np.mean w/ scalars...) ###
    ### so if our inputs were bfloat16 we cast floating results back ###
    if inputs_bf16 and str(result.dtype) in ("float32", "float64"):
        return result.astype("bfloat16")
    return result

def _bf16_batched_matmul(a, b):
    """
    cupy (14.x) can do 2D bfloat16 matmul through cuBLAS but its batched path treats bfloat16
    as an integer type and fails, so we call cuBLAS gemmStridedBatchedEx ourselves.
    """
    from cupy.cuda import cublas
    a = a if a.ndim > 1 else a[None, :]
    b = b if b.ndim > 1 else b[:, None]
    n, k = a.shape[-2:]
    k2, m = b.shape[-2:]
    if k != k2:
        raise ValueError(f"matmul shape mismatch {a.shape} and {b.shape}")

    ### Broadcast batch dims and flatten them to (B, *, *) contiguous buffers ###
    batch_shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = cp.ascontiguousarray(cp.broadcast_to(a, batch_shape + (n, k))).reshape(-1, n, k)
    b = cp.ascontiguousarray(cp.broadcast_to(b, batch_shape + (k, m))).reshape(-1, k, m)
    batch = a.shape[0]
    out = cp.empty((batch, n, m), dtype="bfloat16")

    ### cuBLAS is column-major, so row-major C = A @ B is computed as C^T = B^T @ A^T ###
    alpha, beta = np.array(1.0, np.float32), np.array(0.0, np.float32)
    cublas.gemmStridedBatchedEx(
        cp.cuda.device.get_cublas_handle(), cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
        m, n, k, alpha.ctypes.data,
        b.data.ptr, _CUDA_R_16BF, m, k * m,
        a.data.ptr, _CUDA_R_16BF, k, n * k,
        beta.ctypes.data,
        out.data.ptr, _CUDA_R_16BF, m, n * m,
        batch, cublas.CUBLAS_COMPUTE_32F, cublas.CUBLAS_GEMM_DEFAULT,
    )
    return out.reshape(batch_shape + (n, m))

_CUDA_R_16BF = 14

def _matmul_needs_bf16_path(x, y):
    return (_is_bf16(x) or _is_bf16(y)) and isinstance(x, cp.ndarray) and isinstance(y, cp.ndarray) \
        and (x.ndim > 2 or y.ndim > 2)

_SCATTER_DTYPES = ("float32", "float16", "bfloat16")
_sorted_segment_add = None

def _get_sorted_segment_add():
    ### Lazy: fused_ops imports torch/triton, and _array is imported before everything else ###
    global _sorted_segment_add
    if _sorted_segment_add is None:
        try:
            from mytorch.nn.functional.fused_ops.scatter import sorted_segment_add
            _sorted_segment_add = sorted_segment_add
        except ImportError:
            _sorted_segment_add = False
    return _sorted_segment_add

def _is_basic_index(i):
    return i is None or i is Ellipsis or isinstance(i, (int, np.integer, slice))

def _num_leading_int_arrays(indices):
    """k if indices look like (int_array x k, then only full slices), else 0"""
    k = 0
    for i in indices:
        if isinstance(i, cp.ndarray) and i.dtype.kind in "iu":
            k += 1
        else:
            break
    if k and all(isinstance(i, slice) and i == slice(None) for i in indices[k:]):
        return k
    return 0

def add_at(target, indices, values):
    """
    In place target[indices] += values where repeated indices are summed (xp.add.at semantics).
    This is the backward of indexing, so we pick the cheapest correct path:
    """
    if isinstance(target, np.ndarray):
        return np.add.at(target, indices, values)

    index_tuple = indices if isinstance(indices, tuple) else (indices,)
    if all(_is_basic_index(i) for i in index_tuple):
        target[indices] += values
        return
    if len(index_tuple) == 1 and isinstance(index_tuple[0], cp.ndarray) and index_tuple[0].dtype == cp.bool_ \
            and target.flags.c_contiguous:
        ### Selected rows are unique, target[mask] += v would do a masked get AND a masked set ###
        mask = index_tuple[0]
        rows = target.reshape((-1,) + target.shape[mask.ndim:])
        flat_index = cp.flatnonzero(mask)
        rows[flat_index] += values
        return

    k = _num_leading_int_arrays(index_tuple)
    values_dtype = str(getattr(values, "dtype", ""))
    ### Kernel needs rows it can address: C contiguous (any k), or a 1D/2D strided view like a transposed grad ###
    addressable = target.flags.c_contiguous or (k == 1 and target.ndim <= 2 and all(st >= 0 for st in target.strides))
    if k and addressable and str(target.dtype) in _SCATTER_DTYPES \
            and values_dtype in _SCATTER_DTYPES and _get_sorted_segment_add():
        row_shape = target.shape[k:]
        rows = target if k == 1 else target.reshape((-1,) + row_shape)
        if k == 1:
            flat_index = index_tuple[0].ravel()
        else:
            flat_index = cp.ravel_multi_index(cp.broadcast_arrays(*index_tuple[:k]), target.shape[:k], mode="wrap").ravel()
        ### cupy wraps out of bound / negative indices in the forward, so we do too ###
        flat_index = flat_index % rows.shape[0]
        values = cp.broadcast_to(values, tuple(np.broadcast_shapes(*[i.shape for i in index_tuple[:k]])) + row_shape)
        _sorted_segment_add(rows, flat_index, values.reshape((flat_index.shape[0],) + row_shape))
        return

    if _is_bf16(target):
        tmp = target.astype(cp.float32)
        cp.add.at(tmp, indices, values.astype(cp.float32) if hasattr(values, "astype") else values)
        target[...] = tmp.astype("bfloat16")
        return
    return cp.add.at(target, indices, values)

def _coerce_index(index):
    ### Module level on purpose: a recursive closure inside __getitem__ would ###
    ### reference itself and create cyclic garbage on every indexing call ###
    if isinstance(index, tuple):
        return tuple(_coerce_index(i) for i in index)
    if isinstance(index, Array):
        return index._array
    # Handle mytorch.Tensor (assumes it has .data as Array)
    if hasattr(index, 'data') and isinstance(index.data, Array):
        return index.data._array
    return index

class Array:
    _binary_ufuncs = {
        "__add__": "add", "__radd__": "add",
        "__sub__": "subtract", "__rsub__": "subtract",
        "__mul__": "multiply", "__rmul__": "multiply",
        "__truediv__": "true_divide", "__rtruediv__": "true_divide",
        "__floordiv__": "floor_divide", "__rfloordiv__": "floor_divide",
        "__matmul__": "matmul", "__rmatmul__": "matmul",
        "__pow__": "power", "__rpow__": "power",
        "__mod__": "remainder", "__rmod__": "remainder",
        "__and__": "bitwise_and", "__rand__": "bitwise_and",
        "__or__": "bitwise_or", "__ror__": "bitwise_or",
        "__xor__": "bitwise_xor", "__rxor__": "bitwise_xor",
        "__lt__": "less", "__le__": "less_equal",
        "__gt__": "greater", "__ge__": "greater_equal",
        "__eq__": "equal", "__ne__": "not_equal",
    }

    _inplace_ops = {
        "__iadd__": "add",
        "__isub__": "subtract",
        "__imul__": "multiply",
        "__itruediv__": "true_divide",
        "__ifloordiv__": "floor_divide",
        "__imatmul__": "matmul",
        "__ipow__": "power",
        "__imod__": "remainder",
        "__iand__": "bitwise_and",
        "__ior__": "bitwise_or",
        "__ixor__": "bitwise_xor",
    }

    _unary_ufuncs = {
        "__neg__": "negative",
        "__pos__": "positive",
        "__abs__": "absolute",
        "__invert__": "invert",
    }

    def __init__(self, data, device=None, dtype=None):
        
        ### What device do you want to use? ###
        if device is not None:

            ### If a device is provided, no matter where our data is it is moved there ###
            ### Parse the device string `cpu`, `cuda`, `cuda:{idx}`
            if device == "cpu":
                tgt_device = "cpu"
                tgt_device_idx = None

            elif "cuda" in device:

                if not CUDA_AVAILABLE:
                    raise RuntimeError("CUDA Not supported, check cupy installation")
                
                tgt_device, tgt_device_idx = self.__parse_cuda_str(device_str=device)

                ### Make sure our tgt device is available ###
                if tgt_device_idx + 1 > NUM_AVAIL_GPUS:
                    raise RuntimeError(f"cuda:{tgt_device_idx} does not exist")

        ### If the device is not provided default to CPU (no matter what data is)
        ### we can guess based on the device of the input 
        else:
  
            if hasattr(data, "device"):
        
                if isinstance(data.device, str):
                    if "cuda" in data.device:
                        tgt_device, tgt_device_idx = self.__parse_cuda_str(device_str=str(data.device))

                    else:
                        tgt_device, tgt_device_idx = "cpu", None

                elif isinstance(data.device, cp.cuda.device.Device):
                    tgt_device = "cuda"
                    tgt_device_idx = data.device.id
            else:
                tgt_device = "cpu"
                tgt_device_idx = None

        ### Figure out dtype, always default to float32 (dtypes stored in .dtypes file) ###
        if dtype is None:
            # If data already has a dtype, use it
            if hasattr(data, "dtype"):
                current_dtype = str(data.dtype)
                if current_dtype == "float64":
                    dtype = "float32"
                elif current_dtype == "int64":
                    dtype = "int32"
                else:
                    dtype = current_dtype
            else:
                # Default fallback
                dtype = "float32"
        else:
            
            if not isinstance(dtype, str):
                dtype = str(dtype)
                
        ### If input is already Array Type, grab its data
        if isinstance(data, Array):
            self._array = data._array

        ### If its just a np.ndarray or cp.ndarray then just grab that data 
        elif isinstance(data, (np.ndarray, cp.ndarray)):
            self._array = data    
        
        ### If its just an int/float/list/tuple etc... ###
        else:
            self._array = np.array(data)

        ### Map to Correct Device ###
        src_dev = "cpu" if isinstance(self._array, np.ndarray) else f"cuda:{self._array.device.id}"
        self._array = self.__move_array(self._array, 
                                        src_dev=src_dev, 
                                        tgt_dev=tgt_device,
                                        tgt_dev_idx=tgt_device_idx)
        
        ### Map to Correct dtype (if on GPU always have to do under context manager) ###
        ### All cupy ops default on "cuda:0", but if we have tensors in "cuda:1" and are ###
        ### doing anything to them, we have to make sure we do it under the context ###
        current_dtype = str(self._array.dtype)
        if current_dtype != dtype:
            if "cuda" in tgt_device and CUDA_AVAILABLE:
                with cp.cuda.Device(tgt_device_idx):
                    self._array = self._array.astype(dtype)
            else:
                self._array = self._array.astype(dtype)

        ### Cache frequently accessed attributes ###
        self._xp = np if isinstance(self._array, np.ndarray) else cp
        self._dev_id = None if self._xp is np else self._array.device.id
        self._device = "cpu" if self._xp is np else f"cuda:{self._dev_id}"
        # self._dtype = str(self._array.dtype)

    @property
    def xp(self):
        return self._xp

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def size(self):
        return self._array.size

    @property
    def T(self):
        return Array(self._array.T, device=self._device)
    
    def astype(self, dtype, copy=False):
        if self.dtype == dtype:
            return self
        if self._xp is np:
            self._array = self._array.astype(dtype, copy=copy)
        else:
            with cp.cuda.Device(self._dev_id):
                self._array = self._array.astype(dtype, copy=copy)

        return self
    
    def to(self, device):
        
        ### "cuda" defaults to first gpu ###
        if device == "cuda":
            device = "cuda:0"
        
        ### if our tgt device is the same as current device, theres nothing to do ###
        if device == self._device:
            return self
        else:
            if device == "cpu":
                tgt_dev = "cpu"
                tgt_dev_idx = None
            else:
                tgt_dev, tgt_dev_idx = self.__parse_cuda_str(device)
 
            return Array(data=self.__move_array(arr=self._array, 
                                                src_dev=self._device,
                                                tgt_dev=tgt_dev, 
                                                tgt_dev_idx=tgt_dev_idx),
                         device=device, 
                         dtype=self.dtype)

    def __parse_cuda_str(self, device_str):
        tgt_device = "cuda"
        tgt_device_idx = int(device_str.split(":")[-1]) if ":" in device_str else 0
        return tgt_device, tgt_device_idx
    
    def __move_array(self, arr, src_dev, tgt_dev, tgt_dev_idx=None):
        src_tgt = src_dev if src_dev == "cpu" else "cuda"
        src_idx = None if src_dev == "cpu" else int(src_dev.split(":")[-1])
        tgt_idx = tgt_dev_idx if tgt_dev == "cuda" else None
        if src_tgt == tgt_dev and src_idx == tgt_idx:
            return arr
        if tgt_dev == "cuda":
            if not CUDA_AVAILABLE:
                raise RuntimeError("CUDA Not supported, check cupy installation")
            
            ### default device is "cuda:0"
            if tgt_dev_idx is None:
                tgt_dev_idx = 0

            with cp.cuda.Device(tgt_dev_idx):
                return cp.asarray(arr)
        else:
            return cp.asnumpy(arr) # cp.asnumpy will also work on numpy arrays, basically a no_op

    def asnumpy(self):
        if self._device == "cpu":
            return self._array
        return cp.asnumpy(self._array)

    def _coerce_other(self, other):
        if isinstance(other, Array):
            return other._array, other._device
        if isinstance(other, np.ndarray):
            return other, "cpu"
        if CUDA_AVAILABLE and isinstance(other, cp.ndarray):
            return other, f"cuda:{other.device.id}"
        ### cupy promotes bfloat16 op python-float to float32, so make scalars bfloat16 ###
        if self._xp is not np and isinstance(other, (int, float)) and not isinstance(other, bool) and _is_bf16(self._array):
            with cp.cuda.Device(self._dev_id):
                return cp.full((), other, dtype="bfloat16"), None
        return other, None

    @classmethod
    def _make_binary_op(cls, ufunc_name, reflect=False):
        def op(self, other):
            other_arr, other_dev = self._coerce_other(other)
            
            if other_dev is not None and other_dev != self._device:
                raise RuntimeError(f"Expected all tensors to be on the "
                 f"same device, but found at least two devices, "
                 f"{self._device} and {other_dev}!")
            
            rhs = other_arr
            xp = self._xp
            func = getattr(xp, ufunc_name)

            if reflect:
                _in = (rhs, self._array)
            else:
                _in = (self._array, rhs)

            if xp is cp:
                with cp.cuda.Device(self._dev_id):
                    if ufunc_name == "matmul" and _matmul_needs_bf16_path(*_in):
                        res = _bf16_batched_matmul(*_in)
                    else:
                        res = func(*_in)
            else:
                res = func(*_in)

            return Array(res, device=self._device)
        
        return op

    @classmethod
    def _make_unary_op(cls, ufunc_name):
        def op(self):
            func = getattr(self._xp, ufunc_name)

            if self._xp is np:
                res = func(self._array)
            else:
                with cp.cuda.Device(self._dev_id):
                    res = func(self._array)
            return Array(res, device=self._device)
        return op

    @classmethod
    def _make_inplace_op(cls, ufunc_name):
        def op(self, other):
            other_arr, other_dev = self._coerce_other(other)
            
            # print(other_dev, self._device)
            if other_dev is not None and other_dev != self._device:
                raise RuntimeError(f"Expected all tensors to be on the "
                 f"same device, but found at least two devices, "
                 f"{self._device} and {other_dev}!")
            
            func = getattr(self._xp, ufunc_name)

            if self._xp is np:
                func(self._array, other_arr, out=self._array)
            else:
                with cp.cuda.Device(self._dev_id):
                    func(self._array, other_arr, out=self._array)
            return self
        return op

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        """
        Pretty printing (PyTorch-like) for Array
        """

        # Access underlying array (numpy or cupy) for string conversion
        data = self._array

        # Convert array to a formatted string
        data_str = self.xp.array2string(
            data,
            separator=" ",
            precision=5,
            floatmode="fixed",
            max_line_width=80
        )

        # Indent continuation lines (like torch does)
        lines = data_str.split("\n")
        if len(lines) > 1:
            indent = " " * len("Array(")
            data_str = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

        # Device info (only show if GPU)
        device_info = f", device='{self.device}'" if "cuda" in self.device else ""

        # Final string with dtype always showsn
        return f"Array({data_str}, dtype={self.dtype}{device_info})"

    def __array_function__(self, func, types, args, kwargs):

        """
        For some methods like np.concatenate(), it will trigger the 
        __array_function__ method instead. Basically, whenever
        np.concatenate is called with one or more arguments that are 
        not standard numpy arrays, it goes here to see how to handle them

        You can find more info here: https://numpy.org/doc/stable/reference/arrays.classes.html

        """
        
        ### I am only building this assuming everything is of Array type ###
        if not all(issubclass(t, Array) for t in types):
            return NotImplemented

        ### Get all devices (should all be the same) ###
        devices = set()
        bf16_inputs = []
        def handler(x):
            if isinstance(x, Array):
                devices.add(x._device)
                if _is_bf16(x._array):
                    bf16_inputs.append(True)
                return x._array
            elif isinstance(x, (list, tuple)):
                return type(x)(handler(y) for y in x)
            elif isinstance(x, dict):
                return {k: handler(v) for k, v in x.items()}
            else:
                return x

        handled_args = handler(args)
        handled_kwargs = handler(kwargs)

        ### If more than one device, we cant do the op ###
        if len(devices) > 1:
            raise RuntimeError(f"Expected all tensors to be on the "
                f"same device, but found at least two devices!")

        ### If nothing we passed in is an array then we will use the ###
        ### device that we set in self ###
        if not devices:
            device = self._device
        else:
            device = list(devices)[0]

        ### Get Backend for Operation ###
        xp = cp if "cuda" in device else np

        ### Get the Method ###
        xp_func = getattr(xp, func.__name__, None)

        if xp_func is None:
            return NotImplemented

        ### Run the Operation ###
        if "cuda" in device:
            _, dev_id = self.__parse_cuda_str(device)
            with cp.cuda.Device(dev_id):
                result = xp_func(*handled_args, **handled_kwargs)
        else:
            result = xp_func(*handled_args, **handled_kwargs)

        ### Wrap back into Array if result is an array ###
        if isinstance(result, (np.ndarray, cp.ndarray if CUDA_AVAILABLE else type(None))):
            return Array(_keep_bf16(result, bool(bf16_inputs)), device=device)
        
        return result
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle numpy/cupy ufuncs, enforcing device consistency.
        """
        
        arrays = []
        devices = set()

        # Extract underlying arrays and track devices
        for x in inputs:
            if isinstance(x, Array):
                arrays.append(x._array)
                devices.add(x._device)
            else:
                arrays.append(x)
                if isinstance(x, np.ndarray):
                    devices.add("cpu")
                elif CUDA_AVAILABLE and isinstance(x, cp.ndarray):
                    devices.add(f"cuda:{x.device.id}")
     
        # Enforce single-device rule
        if len(devices) > 1:
            raise RuntimeError(f"All inputs must be on the same device, found: {devices}")

        # Pick device
        device = list(devices)[0] if devices else "cpu"

        # Run the ufunc under the correct context
        if "cuda" in device:
            _, dev_id = self.__parse_cuda_str(device)
            with cp.cuda.Device(dev_id):
                ### cupy promotes bfloat16 op python-float to float32, so make scalars bfloat16 ###
                ### (cp.full is a device-side fill, no host->device sync) ###
                if any(_is_bf16(a) for a in arrays):
                    arrays = [cp.full((), a, dtype="bfloat16") if isinstance(a, (int, float)) and not isinstance(a, bool) else a
                              for a in arrays]
                if ufunc.__name__ == "matmul" and method == "__call__" and not kwargs and _matmul_needs_bf16_path(*arrays):
                    result = _bf16_batched_matmul(*arrays)
                elif ufunc.__name__ == "add" and method == "at" and _is_bf16(arrays[0]):
                    result = add_at(*arrays, **kwargs)
                else:
                    result = getattr(ufunc, method)(*arrays, **kwargs)
        else:
            result = getattr(ufunc, method)(*arrays, **kwargs)

        # Wrap result back into Array if applicable
        if isinstance(result, (np.ndarray, cp.ndarray)):
            return Array(result, device=device)
        return result

    def __getitem__(self, idx):
    
        idx = _coerce_index(idx)
    
        if self.xp == np:
            result = self._array[idx]
        else:
            with cp.cuda.Device(self._array.device.id):
                result = self._array[idx]

        return Array(result, device=self.device)

    def __setitem__(self, idx, value):
        # Allow assigning Array, ndarray, or scalar
        if isinstance(value, Array):
            value = value._array
        self._array[idx] = value

    @property
    def __cuda_array_interface__(self):
        """
        cupy rebuilds any foreign array (e.g. `raw_cupy * Array`, cp.concatenate([Array, ...])) from this
        interface. A bfloat16 typestr is '<V2', which reads back as a structured void dtype, so we advertise
        the registered 'bfloat16' name instead (same memory, no copy). Numpy backed arrays raise AttributeError.
        """
        iface = self._array.__cuda_array_interface__
        if _is_bf16(self._array):
            iface = dict(iface)
            iface["typestr"] = "bfloat16"
            iface.pop("descr", None)
        return iface

    def __getattr__(self, name):
        """
        To get all the attributes of np or cp that we did not 
        explicitly define!
        """
        if hasattr(self._array, name):
            attr = getattr(self._array, name)

            ### Methods like .copy(), .sum(), .reshape() would otherwise leak ###
            ### raw np/cp arrays, so wrap array results back into Array ###
            ### (.get() is the explicit GPU -> numpy escape hatch, leave it raw) ###
            if callable(attr) and name != "get":
                array_types = (np.ndarray, cp.ndarray) if CUDA_AVAILABLE else (np.ndarray,)
                def _wrapped(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if isinstance(result, array_types):
                        return Array(result, device=self._device)
                    return result
                return _wrapped
            return attr
        raise AttributeError(f"'Array' object has no attribute '{name}'")
    
    @classmethod
    def _wrap_factory(cls, xp_func, *args, device="cpu", dtype="float32", **kwargs):
        """
        Wrap numpy/cupy factory functions (zeros, ones, arange, etc.)
        """
        # Pick xp from device
        xp = np if "cpu" in device else cp

        # Parse CUDA device
        _, tgt_device_idx = ("cpu", None)
        if "cuda" in device:
            _, tgt_device_idx = cls(None).__parse_cuda_str(device)

        # Run factory function under correct CUDA context
        if xp == cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = getattr(xp, xp_func)(*args, **kwargs)
        else:
            arr = getattr(xp, xp_func)(*args, **kwargs)

        if dtype is not None:
            current_dtype = str(arr.dtype)
            if current_dtype != dtype:
                if xp == np:
                    arr = arr.astype(dtype)
                else:
                    with cp.cuda.Device(tgt_device_idx):
                        arr = arr.astype(dtype)

        return cls(arr, device=device, dtype=str(arr.dtype))
    
    @classmethod
    def zeros(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("zeros", shape, device=device, dtype=dtype)

    @classmethod
    def ones(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("ones", shape, device=device, dtype=dtype)

    @classmethod
    def empty(cls, shape, device="cpu", dtype="float32"):
        return cls._wrap_factory("empty", shape, device=device, dtype=dtype)

    @classmethod
    def full(cls, shape, fill_value, device="cpu", dtype="float32"):
        return cls._wrap_factory("full", shape, fill_value, device=device, dtype=dtype)

    @classmethod
    def arange(cls, start, end=None, step=1, device="cpu", dtype="float32"):
        if end is None:
            end = start
            start = 0

        return cls._wrap_factory("arange", start, end, step, device=device, dtype=dtype)

    @classmethod
    def linspace(cls, start, end, num=50, device="cpu", dtype="float32"):
        xp = np if "cpu" in device else cp
        arr = xp.linspace(start, end, num=num, dtype=dtype)
        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def eye(cls, N, M=None, k=0, device="cpu", dtype="float32"):
        return cls._wrap_factory("eye", N, M, k, device=device, dtype=dtype)

    @classmethod
    def randn(cls, shape, device="cpu", dtype="float32"):
        xp = np if "cpu" in device else cp
        tgt_device_idx = None
        
        if "cuda" in device:
           _, tgt_device_idx = cls(None).__parse_cuda_str(device)
        
        # Generate array on the correct device
        if xp is cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = xp.random.randn(*shape).astype(dtype)
        else:
            arr = xp.random.randn(*shape).astype(dtype)
        
        return cls(arr, device=device, dtype=str(arr.dtype))
    
    @classmethod
    def rand(cls, shape, device="cpu", dtype="float32"):
        xp = np if "cpu" in device else cp
        tgt_device_idx = None

        if "cuda" in device:
            _, tgt_device_idx = cls(None).__parse_cuda_str(device)

        # Generate array on the correct device
        if xp is cp:
            with cp.cuda.Device(tgt_device_idx):
                arr = xp.random.rand(*shape).astype(dtype)
        else:
            arr = xp.random.rand(*shape).astype(dtype)

        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def randint(cls, low, high, shape, device="cpu", dtype="int32"):
        xp = np if "cpu" in device else cp
        arr = xp.random.randint(low, high, size=shape, dtype=dtype)
        return cls(arr, device=device, dtype=str(arr.dtype))

    @classmethod
    def tril(cls, x, k=0, device="cpu", dtype="float32"):
        return cls._wrap_factory("tril", x, k=k, device=device, dtype=dtype)
 
    @classmethod
    def zeros_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("zeros_like", other, device=device, dtype=dtype)

    @classmethod
    def ones_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("ones_like", other, device=device, dtype=dtype)

    @classmethod
    def empty_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("empty_like", other, device=device, dtype=dtype)
    
    @classmethod
    def full_like(cls, other, fill_value, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls._wrap_factory("full_like", other, fill_value, device=device, dtype=dtype)
    
    @classmethod
    def randn_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls.randn(other.shape, device=device, dtype=dtype)
    
    @classmethod
    def rand_like(cls, other, device=None, dtype=None):
        device = device or other.device
        dtype = dtype or str(other.dtype)
        return cls.rand(other.shape, device=device, dtype=dtype)
    
    
# Attach binary, unary, and inplace operations
for dunder, ufunc in Array._binary_ufuncs.items():
    reflect = dunder.startswith("__r")
    setattr(Array, dunder, Array._make_binary_op(ufunc, reflect=reflect))

for dunder, ufunc in Array._unary_ufuncs.items():
    setattr(Array, dunder, Array._make_unary_op(ufunc))

for dunder, ufunc in Array._inplace_ops.items():
    setattr(Array, dunder, Array._make_inplace_op(ufunc))


if __name__ == "__main__":
    def test_array_operations():
        # --- CPU arrays ---
        a_cpu = Array([1,2,3], device="cpu")
        b_cpu = Array([4,5,6], device="cpu")

        # Binary operations (add, sub, mul)
        assert np.allclose((a_cpu + b_cpu).asnumpy(), np.array([5,7,9]))
        assert np.allclose((b_cpu - a_cpu).asnumpy(), np.array([3,3,3]))
        assert np.allclose((a_cpu * b_cpu).asnumpy(), np.array([4,10,18]))

        # Unary operations
        assert np.allclose((-a_cpu).asnumpy(), np.array([-1,-2,-3]))
        assert np.allclose((+b_cpu).asnumpy(), np.array([4,5,6]))
        assert np.allclose(abs(Array([-1,-2,-3])).asnumpy(), np.array([1,2,3]))

        # Inplace operations
        a_copy = Array([1,2,3])
        a_copy += Array([10,20,30])
        assert np.allclose(a_copy.asnumpy(), np.array([11,22,33]))

        # --- GPU arrays ---
        if CUDA_AVAILABLE:
            a_gpu = Array([1,2,3], device="cuda:0")
            b_gpu = Array([4,5,6], device="cuda:0")

            # Binary ops
            assert np.allclose((a_gpu + b_gpu).asnumpy(), np.array([5,7,9]))
            assert np.allclose((b_gpu - a_gpu).asnumpy(), np.array([3,3,3]))
            assert np.allclose((a_gpu * b_gpu).asnumpy(), np.array([4,10,18]))

            # Unary ops
            assert np.allclose((-a_gpu).asnumpy(), np.array([-1,-2,-3]))
            assert np.allclose((+b_gpu).asnumpy(), np.array([4,5,6]))

            # Inplace ops
            a_copy_gpu = Array([1,2,3], device="cuda:0")
            a_copy_gpu *= Array([10,20,30], device="cuda:0")
            assert np.allclose(a_copy_gpu.asnumpy(), np.array([10,40,90]))

            # Device mismatch errors
            try:
                _ = a_cpu + a_gpu
            except RuntimeError as e:
                assert "device" in str(e)

            try:
                _ = np.add(a_cpu, a_gpu)
            except RuntimeError as e:
                assert "device" in str(e)

            # Using .xp backend
            assert np.allclose(np.add(a_gpu, a_gpu).asnumpy(), np.array([2,4,6]))

        # Factory functions
        z = Array.zeros((2,2), device="cpu")
        assert np.allclose(z.asnumpy(), np.zeros((2,2)))

        o = Array.ones((2,2), device="cpu")
        assert np.allclose(o.asnumpy(), np.ones((2,2)))

        r = Array.arange(3, device="cpu")
        assert np.allclose(r.asnumpy(), np.array([0,1,2]))

        print("All CPU/GPU tests passed!")

   
    test_array_operations()