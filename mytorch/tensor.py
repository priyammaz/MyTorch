import numpy as np
import weakref
from functools import wraps
from contextlib import contextmanager
import warnings
from . import _array as ap
from .dtypes import *

class no_grad:
    def __enter__(self):
        self.old_state = Tensor._build_graph
        Tensor._build_graph = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Tensor._build_graph = self.old_state
        # Returning False ensures exceptions inside the block are not suppressed
        return False

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
class Tensor:
    
    _build_graph = True

    def __init__(self, 
                 data, 
                 requires_grad=False,
                 grad_fn=None, 
                 grad_fn_name=None,
                 device=None,
                 dtype=None):
        
        ### If our data is not an Array type, convert it ###
        ### Array handles everything regarding our basic ops ###
        ### device, dtype, and anything else numpy would do ###
        self._data = ap.Array(data=data, 
                              device=device, 
                              dtype=dtype)
        
        ### Set Autograd Variables ### 
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad_fn_name = grad_fn_name
        self.grad = None
        self._is_leaf = self.requires_grad and (self.grad_fn is None)
        self._parents = ()
        self._version = 0
        self._retain_grad = False
        self._warn_retain_grad = False

    @property
    def xp(self):
        return self._data.xp

    @property
    def data(self):
        """
        simple (view) access to data
        """
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = ap.Array(value)
        return self

    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def device(self):
        return self._data.device
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def ndim(self):
        return len(self._data.shape)
    
    @property
    def is_leaf(self):
        return self._is_leaf
    
    def __repr__(self):
        """
        Pretty printing
        """

        ### Access underlying _array for printing only ! ###
        data = self.data._array

        # Convert array to string
        data_str = self.xp.array2string(
            data,
            separator=" ",
            precision=5,
            floatmode="fixed",
            max_line_width=80
        )

        # Indent all lines after the first like PyTorch
        lines = data_str.split("\n")
        if len(lines) > 1:
            indent = " " * len("tensor(")
            data_str = lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

        # Grad / requires_grad info
        grad_info = ""
        if getattr(self, "requires_grad", False):
            if getattr(self, "grad_fn", None) is not None:
                grad_info = f", grad_fn={getattr(self, 'grad_fn_name', None)}"
            else:
                grad_info = ", requires_grad=True"

        # Device info
        device_info = ""
        if "cuda" in self.device:
            device_info = f", device={self.device}"

        # Final string
        return f"tensor({data_str}{grad_info}{device_info})"
    
    def to(self, device):
        ### use the setter of our data attribute to replace our ###
        ### existing self.data with the new one on the new device ###
        self.data = self.data.to(device)
        return self

    @classmethod
    def build_graph_enabled(cls):
        return cls._build_graph
    
    @staticmethod
    def _check_broadcast(a, b):

        ## Verify that two numpy arrays are broadcastable ###
        ## This means a and b have the same number of dimensions ###
        ## I.E (1x3) + (1x1) summation is broadcasting

        ### We only really care about this when both a and b requires gradients ###
        ### as if they dont, then either a or b are just some constant ###

        ## Numpy technically supports broadcasting even when the dimensionality ###
        ## is not the same (1 x 3) + (1, ) but we wont for simplicity! ###
        if (len(a.shape) != len(b.shape)) and (a.requires_grad and b.requires_grad):
            raise ValueError(f"Incompatible Operation between {a.shape} and {b.shape}")

    def _broadcasted_grad_accumulate(self, x_shape, x_grad):

        grad_shape = x_grad.shape

        assert len(x_shape) == len(grad_shape), "Gradient and tensor shapes must be the same length! Only different by broadcasting"

        sum_axes = [idx for idx, (x_dim, grad_dim) in enumerate(zip(x_shape, grad_shape)) if x_dim == 1 and grad_dim != 1]
        if sum_axes:
            x_grad = np.sum(x_grad, axis=tuple(sum_axes), keepdims=True)

        return x_grad
    
    def retain_grad(self):
        
        if not self._warn_retain_grad:
            warnings.warn(
                "You are retaining graph, intermediate gradients may not be cleared!!"
            )
            self._warn_retain_grad = True

        ### Leaf Tensors always retain grad ###
        if self.is_leaf:
            return 

        self._retain_grad = True

    def backward(self, grad=None, retain_graph=False):
        
        if retain_graph:
            if not self._warn_retain_grad:
                warnings.warn(
                    "You are retaining graph, intermediate gradients may not be cleared!!"
                )
                self._warn_retain_grad = True

        # Initialize output gradient
        if grad is None:
            grad = ap.Array.ones_like(self.data, dtype=self.dtype, device=self.device)

        self.grad = grad
 
        # Build topo-order
        visited = set()
        topo_order = []

        def build_topo(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            parents = getattr(t, "_parents", ())
            if parents is None:
                parents = []
            for parent_ref in parents:
                parent = parent_ref()
                if parent is not None:
                    build_topo(parent)
            topo_order.append(t)

        build_topo(self)

        # Iterate in reverse topological order
        for t in reversed(topo_order):
            if t.grad_fn is not None:
                t.grad_fn(t.grad)  # accumulate into parents
                
                ### Drop references immediately ###
                retain_this = getattr(t, "_retain_grad", False) or retain_graph
                
                if not retain_this:
                    # Clear backward references for GC
                    t.grad_fn = None
                    t._parents = None

                    # Non-leaf tensors don't need their grad anymore
                    if not t.is_leaf:
                        t.grad = None


    ###################################
    ######## BINARY OPERATIONS ########
    ###################################

    """
    Binary operations are those that happen between two different Tensors!
    Simple examples are Sum, Subtract, Mult, Div, MatMul, etc... These are all
    operations that occur between two different inputs!
    """
    def __add__(self, val):

        """
        Sum of two tensors (with accumulation for brodcasting)
        O = A + B
        dO/dA = 1
        dO/dB = 1
        """

        ### If Val is a Tensor, Then Check Devices ###
        if isinstance(val, Tensor): 

            ### Check Broadcast Shape ###
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape

        ### If we are just summing with an scalar ###
        ### Just cast to our dtype without any issues ###
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Use standard __add__ to actually add tensors together ###
        output = np.add(self.data, val_data)
        
        ### Define Backward Function ###
        def _add_backward(input_grad):

            if self.requires_grad:
                self_grad = self._broadcasted_grad_accumulate(self.shape, input_grad)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                
                self_grad = None

            if val_requires_grad:
                val_grad = self._broadcasted_grad_accumulate(val_shape, input_grad)
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad

                val_grad = None

        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_add_backward if requires_grad else None,
                        grad_fn_name="<AddBackward>" if requires_grad else None,
                        device=self.device)

        ### Set Parents ###
        if requires_grad:
            output._add_parents(self, val if isinstance(val, Tensor) else None)

        return output

    def __radd__(self, val):

        """
        add is not an ordered operation, A + B is the same as B + A

        In A + B, our self is A and val is B
        When we do A + B, what is really happening is A.__add__(B). 

        But if A is an integer and B is a Tensor, python integers dont know how to work with our
        own tensor operations. This will throw an error and then try __radd__.  
    
        __radd__ will reverse the operands and do B.__add__(A), using our own Tensor __add__ written above instead.  
        Our __add__ we wrote for the tensor does know how to interface python numbers and tensors so we can then do the operation!

        """
        return self + val
    
    def __iadd__(self, val):
        """
        Inplace operation to enable self += val
            - prevent inplace operation on leaf tensors that require grad
            - tracks version for non-leaf tensors to see if there is a mismatch
        
        The problem is, leaf tensors for us are typically our parameters. So we typically only
        want to change them through our gradient updates. On the other hand, if I manually change
        them in place, i dont really know what to do with that. PyTorch for this reason just simplifies
        it and throws an error any time we are tracking gradients on a leaf tensor and apply some inplace op. 

        On the other hand, on non-leaf tensors (created through performing ops on our leaf tensors) it is 
        still fine for the most part but with a condition. During our forward propagation, the metadata from
        that graph is stored for backprop. Now if I manually change something in the graph inplace, 
        that graph is no longer storing the correct information and backprop wont be accurate anymore. 

        Thus pytorch employs a versioning index, and every inplace op increments the version. If there 
        is a mismatch during backprop between the version that was on the graph and the version thats 
        currently there (indicating some inplace op happened after graph creation), then we raise and error. 
        
        Technically, we should just never do inplace ops, because what are you really saving?? 
        
        a+=b vs a = a + b... 

        wow...

        Either way lets implement it to still have something close to torch!
        """

        if self.requires_grad and self.is_leaf:
            raise RuntimeError("A leaf Tensor that requires grad is being used in an in-place operation")
        
        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Capture current version of the tensor ###
        saved_version = getattr(self, "_version", 0)

        ### Capture the old grad function to use ###
        old_self_grad_fn = getattr(self, "grad_fn", None)
        old_val_grad_fn = getattr(val, "grad_fn", None)

        ### inplace op ###
        self.data += val_data

        ### increment version (default 0 if it doesn't exist but it should always) ###
        self._version = getattr(self, "_version", 0) + 1
        
        ### Handle Backward with versioning ###
        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        if requires_grad:
            def _iadd_backward(input_grad):
                # Version check on leaf tensors where we really care ###
                if self.is_leaf and self._version != saved_version + 1:
                    raise RuntimeError(
                        "one of the variables needed for gradient computation "
                        "has been modified by an in-place operation"
                    )

                ### If we arent a leaf tensor we just use our original grad fn ###
                if self.requires_grad:
                    grad_self = self._broadcasted_grad_accumulate(self.shape, input_grad)
                    
                    ### If a leaf tensor just accumulate grads like normal ###
                    if self.is_leaf or getattr(self, "_retrain_grad", False):
                        if self.grad is None:
                            self.grad = grad_self
                        else:
                            self.grad += grad_self

                    ### If not a leaf tensor, just use the old grad function ###
                    elif not self.is_leaf and old_self_grad_fn is not None:
                        old_self_grad_fn(grad_self)
                    
                if val_requires_grad:
    
                    grad_val = val._broadcasted_grad_accumulate(val_shape, input_grad)
                    if val.is_leaf or getattr(val, "_retrain_grad", False):
                        if val.grad is None:
                            val.grad = grad_val
                        else:
                            val.grad += grad_val
                        
                    elif not self.is_leaf and old_val_grad_fn is not None:
                        old_val_grad_fn(grad_val)

            self.grad_fn = _iadd_backward
            self.grad_fn_name = "<IAddBackward>"

        return self

    def __sub__(self, val):

        """
        Same as __add__ but now subtraction (with accumulation for broadcasting)
        O = A - B
        dO/dA = 1
        dO/dB = -1
        """

        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape

        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None

        output = np.subtract(self.data, val_data)
        
        ### Define Backward Function ###
        def _sub_backward(input_grad):
            if self.requires_grad:
                # self_grad = input_grad
                self_grad = self._broadcasted_grad_accumulate(self.shape, input_grad)
                
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None

            if val_requires_grad:
                # val_grad = -input_grad
                val_grad = self._broadcasted_grad_accumulate(val_shape, -input_grad)
                
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad

                val_grad = None

        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_sub_backward if requires_grad else None,
                        grad_fn_name="<SubBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self, val if isinstance(val, Tensor) else None)
        
        return output
    
    def __rsub__(self, val):

        """
        Subtraction is an ordered operation. Lets say we want A - B where A is self and B is val
        if A is not a tensor (i.e. an int or float), __sub__ will throw an error as it doesnt know
        how to do an operation with our own tensor.

        This will enter __rsub__ where we flip the operands where B is now self and A is val. If we want
        A - B, we need to do -1 * B + A, using our __add__. 

        There are a bunch of ways to handle these exceptions, this is just one of them!
        """

        return -1 * self + val

    def __isub__(self, val):
        """
        Inplace op to enable self -= val
        """

        if self.requires_grad and self.is_leaf:
            raise RuntimeError("A leaf Tensor that requires grad is being used in an in-place operation")
        
        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Capture current version of the tensor ###
        saved_version = getattr(self, "_version", 0)

        ### Capture the old grad function to use ###
        old_grad_fn = getattr(self, "grad_fn", None)

        ### inplace op ###
        self.data -= val_data

        ### increment version (default 0 if it doesn't exist) ###
        self._version = getattr(self, "_version", 0) + 1

        ### Handle Backward with versioning ###
        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        if requires_grad:
            def _isub_backward(input_grad):
                # Version check
                if self.is_leaf and self._version != saved_version + 1:
                    raise RuntimeError(
                        "one of the variables needed for gradient computation "
                        "has been modified by an in-place operation"
                    )

                if self.requires_grad:
                    grad_self = self._broadcasted_grad_accumulate(self.shape, input_grad)
                    
                    ### If a leaf tensor just accumulate grads like normal ###
                    if self.is_leaf:
                        if self.grad is None:
                            self.grad = grad_self
                        else:
                            self.grad += grad_self

                    ### If not a leaf tensor, just use the old grad function ###
                    else:
                        old_grad_fn(grad_self)
                    
                if val_requires_grad:
                    grad_val = val._broadcasted_grad_accumulate(val_shape, -input_grad)
                    if val.is_leaf:
                        if val.grad is None:
                            val.grad = grad_val
                        else:
                            val.grad += grad_val
                    else:
                        val.grad_fn(grad_val)

            self.grad_fn = _isub_backward
            self.grad_fn_name = "<ISubBackward>"

        return self

    def __mul__(self, val):

        """
        Element-wise multiplication of two tensors (with accumulation for broadcasting)

        O = A * B
        dO/dA = B
        do/dB = A
        """

        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape

        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
            
        output = np.multiply(self.data, val_data)

        def _mul_backward(input_grad):

            if self.requires_grad:
                self_grad = np.multiply(input_grad, val_data)
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                
                self_grad = None

            if val_requires_grad:
                val_grad = np.multiply(input_grad, self.data)
                val_grad = self._broadcasted_grad_accumulate(val_shape, val_grad)
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad

                val_grad = None

        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad, 
                        grad_fn=_mul_backward if requires_grad else None,
                        grad_fn_name="<MulBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self, val if isinstance(val, Tensor) else None)

        return output
    
    def __rmul__(self, val):
        return self * val

    def __imul__(self, val):
        """
        Inplace op to enable self *= val
        """

        if self.requires_grad and self.is_leaf:
            raise RuntimeError("A leaf Tensor that requires grad is being used in an in-place operation")
        
        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Capture current version of the tensor ###
        saved_version = getattr(self, "_version", 0)

        ### Capture the old grad function to use ###
        old_grad_fn = getattr(self, "grad_fn", None)

        ### inplace op ###
        self.data *= val_data

        ### increment version (default 0 if it doesn't exist) ###
        self._version = getattr(self, "_version", 0) + 1

        ### Handle Backward with versioning ###
        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        if requires_grad:
            def _imul_backward(input_grad):
                # Only check version for leaf tensors
                if self.is_leaf and self._version != saved_version + 1:
                    raise RuntimeError(
                        "one of the variables needed for gradient computation "
                        "has been modified by an in-place operation"
                    )

                # Gradient w.r.t. self
                if self.requires_grad:
                    grad_self = input_grad * val_data
                    grad_self = self._broadcasted_grad_accumulate(self.shape, grad_self)

                    if self.is_leaf:
                        if self.grad is None:
                            self.grad = grad_self
                        else:
                            self.grad += grad_self
                    else:
                        if old_grad_fn is not None:
                            old_grad_fn(grad_self)

                # Gradient w.r.t. val
                if val_requires_grad:
                    grad_val = input_grad * self.data
                    grad_val = val._broadcasted_grad_accumulate(val_shape, grad_val)

                    if val.is_leaf:
                        if val.grad is None:
                            val.grad = grad_val
                        else:
                            val.grad += grad_val
                    else:
                        if val.grad_fn is not None:
                            val.grad_fn(grad_val)

            self.grad_fn = _imul_backward
            self.grad_fn_name = "<IMulBackward>"

        return self

    def __neg__(self):
        return self * -1

    def __matmul__(self, val):

        ### Compute MatMul ###
        output_data = np.matmul(self.data, val.data)

        def _matmul_backward(input_grad):

            if self.requires_grad:
                grad_self = np.matmul(input_grad, val.data.swapaxes(-1, -2))
                
                if self.grad is None:
                    self.grad = grad_self
                else:
                    self.grad += grad_self

                grad_self = None

            if val.requires_grad:
                grad_val = np.matmul(self.data.swapaxes(-1, -2), input_grad)
                
                if val.grad is None:
                    val.grad = grad_val
                else:
                    val.grad += grad_val
                
                grad_val = None

        requires_grad = (self.requires_grad or val.requires_grad) and Tensor.build_graph_enabled()
        out = Tensor(
            output_data,
            requires_grad=requires_grad,
            grad_fn=_matmul_backward if requires_grad else None,
            grad_fn_name="<MatmulBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self, val)

        return out

    def __truediv__(self, val):

        """
        Element-wise Division of two tensors (accumulated grad for broadcasting)

        O = A/B
        dO/dA = 1/B
        dO/dB = -A/B^2

        """

        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape

        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None

        output = self.data / val_data

        def _div_backward(input_grad):
            if self.requires_grad:
                self_grad = input_grad / val_data
                self_grad = self._broadcasted_grad_accumulate(self.shape, self_grad)
                
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                
                self_grad = None

            if val_requires_grad:
                val_grad = input_grad * -1 * self.data / (val_data**2)
                val_grad = self._broadcasted_grad_accumulate(val_shape, val_grad)
                
                if val.grad is None:
                    val.grad = val_grad
                else:
                    val.grad += val_grad
                
                val_grad = None
        
        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_div_backward if requires_grad else None,
                        grad_fn_name="<DivBackward>" if requires_grad else None,
                        device=self.device
                    )

        if requires_grad:
            output._add_parents(self, val if isinstance(val, Tensor) else None)

        return output

    def __rtruediv__(self, val):
        
        """
        Div is an ordered operation. Lets say we want A/B, in the case of __div__ A is self and B is val. 
        if A is not a Tensor (i.e. an int or float), A / B will throw an error beacuse we only can divide a tensor by a tensor
        In this case, __rtruediv__ will be called where A is now val and B is self (the operands have been flipped)
        We can then convert A (our non-tensor) which is in val to a tensor and then perform val / self to call __div__ again where
        A and B are both now tensors
        """
        ### if val is not a tensor alredy, we will add as a constant without gradients ###
        if not isinstance(val, Tensor): 
            val = Tensor(val, dtype=self.dtype)
        return val / self

    def __itruediv__(self, val):
        """
        Inplace op to enable self /= val
        """

        if self.requires_grad and self.is_leaf:
            raise RuntimeError("A leaf Tensor that requires grad is being used in an in-place operation")
        
        if isinstance(val, Tensor): 
            self._check_broadcast(self, val)

            val_data = val.data
            val_requires_grad = val.requires_grad
            val_shape = val.shape
        else:
            val_data = ap.Array(val, dtype=self.dtype, device=self.device)
            val_requires_grad = False
            val_shape = None
        
        ### Capture current version of the tensor ###
        saved_version = getattr(self, "_version", 0)

        ### Capture the old grad function to use ###
        old_grad_fn = getattr(self, "grad_fn", None)

        ### inplace op ###
        self.data /= val_data

        ### increment version (default 0 if it doesn't exist) ###
        self._version = getattr(self, "_version", 0) + 1

        ### Handle Backward with versioning ###
        requires_grad = (self.requires_grad or val_requires_grad) and Tensor.build_graph_enabled()
        if requires_grad:
            # Gradient w.r.t. self
            def _idiv_backward(input_grad):
                # Only check version for leaf tensors
                if self.is_leaf and self._version != saved_version + 1:
                    raise RuntimeError(
                        "one of the variables needed for gradient computation "
                        "has been modified by an in-place operation"
                    )

                # Gradient w.r.t. self
                if self.requires_grad:
                    grad_self = input_grad / val_data
                    grad_self = self._broadcasted_grad_accumulate(self.shape, grad_self)

                    if self.is_leaf:
                        if self.grad is None:
                            self.grad = grad_self
                        else:
                            self.grad += grad_self
                    else:
                        if old_grad_fn is not None:
                            old_grad_fn(grad_self)

                # Gradient w.r.t. val
                if val_requires_grad:
                    grad_val = input_grad * (-self.data / (val_data**2))
                    grad_val = val._broadcasted_grad_accumulate(val_shape, grad_val)

                    if val.is_leaf:
                        if val.grad is None:
                            val.grad = grad_val
                        else:
                            val.grad += grad_val
                    else:
                        if val.grad_fn is not None:
                            val.grad_fn(grad_val)

            self.grad_fn = _idiv_backward
            self.grad_fn_name = "<IDivBackward>"

        return self

    ########################
    ### UNARY OPERATIONS ###
    ########################
    
    """
    Unary operations are those that only have a single input. These are operations
    that are performed on each element of the tensors independently. 
    """
    def __pow__(self, exponent):

        """
        Element-wise exponentiation of matrix (assuming exponent is non-learnable for simplicity)
        O = A^K
        dO/dA = K * A^(k-1)
        """

        output = self.data ** exponent
    
        def _pow_backward(input_grad):
            self_grad = input_grad * (exponent * self.data ** (exponent-1))
            
            if self.grad is None:
                self.grad = self_grad
            else:
                self.grad += self_grad

            self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_pow_backward if requires_grad else None,
                        grad_fn_name="<PowBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self)

        return output
    
    def exp(self):
        """
        Element-wise exponentiation of the base e.
        O = e^A
        dO/dA = e^A
        """
        out_data = np.exp(self.data)

        def _exp_backward(input_grad):
            if self.requires_grad:
                self_grad = input_grad * out_data  # use forward output to save recomputation
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_exp_backward if requires_grad else None,
            grad_fn_name="<ExpBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def log(self):

        """
        Element-wise log with base e
        O = log(A)
        dO/dA = 1/a
        """

        output = np.log(self.data)
   
        def _log_backward(input_grad): 

            if self.requires_grad:
                self_grad = input_grad * (1/self.data)

                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output, 
                        requires_grad=requires_grad,
                        grad_fn=_log_backward if requires_grad else None, 
                        grad_fn_name="<LogBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self)

        return output

    def abs(self):
        """
        Element-wise absolute value
        O = |A|
        dO/dA = sign(A) where sign(0) = 0
        """
        output = np.abs(self.data)
        
        def _abs_backward(input_grad):
            if self.requires_grad:
                print("WOW")
                # Gradient is sign of input: 1 for positive, -1 for negative, 0 for zero
                self_grad = input_grad * np.sign(self.data)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                self_grad = None
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_abs_backward if requires_grad else None,
                        grad_fn_name="<AbsBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self)
        return output

    def clamp(self, min_val=None, max_val=None):
        """
        Element-wise clamp (clip) values between min and max
        O = clamp(A, min, max)
        dO/dA = 1 if min <= A <= max, else 0
        """
        output = np.clip(self.data, min_val, max_val)
        
        def _clamp_backward(input_grad):
            if self.requires_grad:
                # Gradient is 1 where value is within bounds, 0 where clamped
                mask = (self.data >= (min_val if min_val is not None else -np.inf)) & \
                    (self.data <= (max_val if max_val is not None else np.inf))
                self_grad = input_grad * mask
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                self_grad = None
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_clamp_backward if requires_grad else None,
                        grad_fn_name="<ClampBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self)
        return output

    def sqrt(self):
        """
        Element-wise square root
        O = sqrt(A)
        dO/dA = 1 / (2 * sqrt(A))
        """

        if (self<0).any():
            warnings.warn("sqrt operation received negative values, which will produce NaN", 
                        RuntimeWarning)
        
        output = np.sqrt(self.data)
        
        def _sqrt_backward(input_grad):
            if self.requires_grad:
                # Gradient is 1 / (2 * sqrt(A))
                self_grad = input_grad * (1 / (2 * np.sqrt(self.data)))
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                self_grad = None
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_sqrt_backward if requires_grad else None,
                        grad_fn_name="<SqrtBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self)
        return output

    def sin(self):
        """
        Element-wise sine
        O = sin(A)
        dO/dA = cos(A)
        """
        output = np.sin(self.data)
        
        def _sin_backward(input_grad):
            if self.requires_grad:
                # Gradient is cos(A)
                self_grad = input_grad * np.cos(self.data)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                self_grad = None
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_sin_backward if requires_grad else None,
                        grad_fn_name="<SinBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self)
        return output

    def cos(self):
        """
        Element-wise cosine
        O = cos(A)
        dO/dA = -sin(A)
        """
        output = np.cos(self.data)
        
        def _cos_backward(input_grad):
            if self.requires_grad:
                # Gradient is -sin(A)
                self_grad = input_grad * (-np.sin(self.data))
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                self_grad = None
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_cos_backward if requires_grad else None,
                        grad_fn_name="<CosBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self)
        return output

    def tan(self):
        """
        Element-wise tangent
        O = tan(A)
        dO/dA = sec^2(A) = 1/cos^2(A)
        """
        output = np.tan(self.data)
        
        def _tan_backward(input_grad):
            if self.requires_grad:
                # Gradient is sec^2(A) = 1/cos^2(A)
                self_grad = input_grad * (1 / np.cos(self.data)**2)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                self_grad = None
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        output = Tensor(output,
                        requires_grad=requires_grad,
                        grad_fn=_tan_backward if requires_grad else None,
                        grad_fn_name="<TanBackward>" if requires_grad else None,
                        device=self.device)
        
        if requires_grad:
            output._add_parents(self)
        return output


    ######################
    ### COMPARISON OPS ###
    ######################
    """
    These are non-differentiable ops that just let us do some
    comparisons between tensors!
    """
    def _compare(self, other, op):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = other
        return Tensor(op(self.data, other_data), requires_grad=False)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._compare(other, lambda a, b: a != b)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)
    
    def any(self):
        return self.xp.any(self.data._array)
    
    #####################################
    ### INDEXING/RESHAPING OPERATIONS ###
    #####################################
    """
    These operations involve either indexing a tensor or
    reshaping them in some way!
    """
    def __getitem__(self, idx):
        """
        Supports slices, ints, arrays, and tuple-of-arrays indexing.
        """
        
        # Convert Tensor indices to cp arrays
        if isinstance(idx, Tensor):
            idx = idx.data

        if isinstance(idx, (list, tuple)):
            idx = tuple(
                (i.data.astype(self.xp.int64) if isinstance(i, Tensor) else self.xp.array(i, dtype=self.xp.int64))
                if isinstance(i, (list, Tensor)) else i
                for i in idx
            )
        
        out_data = self.data[idx]

        def _index_backward(input_grad):

            if self.requires_grad:
                if self.grad is None:
                    self.grad = ap.Array.zeros_like(self.data, dtype=self.data.dtype)
  
                # Convert index to raw array if needed
                actual_idx = idx
                if isinstance(idx, Tensor):
                    actual_idx = idx.data
                if isinstance(actual_idx, ap.Array):
                    actual_idx = actual_idx._array

                # Elementwise assignment for fancy indexing
                np.add.at(self.grad, actual_idx, input_grad)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(out_data,
                    requires_grad=requires_grad,
                    grad_fn=_index_backward if requires_grad else None,
                    grad_fn_name="<IndexBackward>" if requires_grad else None,
                    device=self.device)
        
        if requires_grad:
            out._add_parents(self)

        return out
    
    def __setitem__(self, idx, value):
        """
        Supports slices, ints, arrays, Tensors, and tuple-of-arrays indexing.
        Performs in-place assignment on .data.
        Gradient is not tracked for item assignment (non-differentiable op).
        """
        if isinstance(idx, Tensor):
            idx = idx.data

        if isinstance(idx, (list, tuple)):
            idx = tuple(
                (i.data.astype(self.xp.int64) if isinstance(i, Tensor) else self.xp.array(i, dtype=self.xp.int64))
                if isinstance(i, (list, Tensor)) else i
                for i in idx
            )

        if isinstance(value, Tensor):
            value = value.data

        self.data[idx] = value

    def transpose(self, dim1, dim2):
        """
        Swap two dimensions of the tensor.
        """
        out_data = self.data.swapaxes(dim1, dim2)
 
        def _transpose_backward(input_grad):
            # Just swap back the same two dims
            if self.requires_grad:
                self_grad = input_grad.swapaxes(dim1, dim2)

                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(out_data,
                    requires_grad=requires_grad,
                    grad_fn=_transpose_backward if requires_grad else None,
                    grad_fn_name="<TransposeBackward>" if requires_grad else None,
                    device=self.device)
        
        if requires_grad:
            out._add_parents(self)

        return out
    
    def permute(self, *dims):
        """
        Permute tensor dimensions according to dims.
        Example: (0, 2, 1) will reorder axes in that order.
        """
        out_data = np.transpose(self.data, axes=dims)

        def _permute_backward(input_grad):
            if self.requires_grad:
                # Inverse permutation
                inv_dims = np.argsort(dims)
                self_grad = np.transpose(input_grad, axes=inv_dims)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(out_data,
                    requires_grad=requires_grad,
                    grad_fn=_permute_backward if requires_grad else None,
                    grad_fn_name="<PermuteBackward>" if requires_grad else None,
                    device=self.device)
        
        if requires_grad:
            out._add_parents(self)

        return out
    
    def reshape(self, *shape):
        """
        Reshape the tensor. Gradients are reshaped back to the original shape during backprop.
        """
        out_data = self.data.reshape(*shape)

        def _reshape_backward(input_grad):
            if self.requires_grad:
                self_grad = input_grad.reshape(self.data.shape)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_reshape_backward if requires_grad else None,
            grad_fn_name="<ReshapeBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)
  
        return out
    
    def flatten(self, start_dim=0, end_dim=-1):
        """
        Flatten the tensor by reshaping it into a contiguous range of dimensions.

        Args:
            start_dim: first dimension to flatten (default: 0)
            end_dim: last dimension to flatten (default: -1, meaning last dimension)

        Gradients are reshaped back to the original shape during backprop.
        """
        # Handle negative indices
        if end_dim < 0:
            end_dim = self.data.ndim + end_dim
        if start_dim < 0:
            start_dim = self.data.ndim + start_dim

        # Calculate new shape
        if start_dim == 0 and end_dim == self.data.ndim - 1:
            # Flatten everything
            new_shape = (-1,)
        else:
            # Flatten only the specified range
            new_shape = (
                self.data.shape[:start_dim] +
                (int(np.prod(self.data.shape[start_dim:end_dim+1])),) +
                self.data.shape[end_dim+1:]
            )

        out_data = self.data.reshape(new_shape)

        def _flatten_backward(input_grad):
            if self.requires_grad:
                self_grad = input_grad.reshape(self.data.shape)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_flatten_backward if requires_grad else None,
            grad_fn_name="<FlattenBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)

        return out

    def unsqueeze(self, dim=0):
  
        out_data = np.expand_dims(self.data, axis=dim)

        def _unsqueeze_backward(input_grad):
            if self.requires_grad:
                
                ### Accumulate all grads along the dimension we added ###
                grad = input_grad.sum(axis=dim)

                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_unsqueeze_backward if requires_grad else None,
            grad_fn_name="<UnsqueezeBackward>" if requires_grad else None,
            device=self.device,
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def squeeze(self, dim=None):

        shape = list(self.shape)

        if dim is None:
            # Drop all size-1 dimensions
            out_shape = [s for s in shape if s != 1]
        else:
            if shape[dim] != 1:
                raise ValueError(f"Cannot squeeze dimension {dim} of size {shape[dim]}")
            out_shape = shape[:dim] + shape[dim+1:]

        out_data = self.data.reshape(out_shape)

        def _squeeze_backward(input_grad):
            if self.requires_grad:
                if dim is None:
                    # Put all size-1 dims back
                    grad = input_grad.reshape(shape)
                else:
                    # Only reinsert the squeezed dimension
                    grad = np.expand_dims(input_grad, axis=dim)

                grad = grad.astype(self.data.dtype, copy=False)

                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_squeeze_backward if requires_grad else None,
            grad_fn_name="<SqueezeBackward>" if requires_grad else None,
            device=self.device,
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def chunk(input, chunks, dim=0):

        """
        Split a tensor into `chunks` along dimension `dim`.
        Returns a list of Tensors.

        to make this easy we use slices

        a = ["a", "b", "c", "d", "e", "f", "g"]
        a[1:3] = ["b", "c", "d"]
        a[slice(1,3)] = ["b", "c", "d"]
        """

        size = input.shape[dim]
        if size % chunks != 0:
            raise ValueError(f"Cannot split dimension {dim} of size {size} into {chunks} equal chunks")
        
        chunk_size = size // chunks
        out_tensors = []

        for i in range(chunks):
            start, end = i * chunk_size, (i + 1) * chunk_size

            # Slice the underlying array directly
            idx = [slice(None)] * input.ndim
            idx[dim] = slice(start, end)
            slice_data = input.data[tuple(idx)]

            def _chunk_backward(input_grad, start=start, end=end):
                if input.requires_grad:
                    grad = input.xp.zeros_like(input.data, dtype=input.data.dtype)
                    
                    # Ensure input_grad has the correct shape
                    grad_slice_shape = list(grad.shape)
                    grad_slice_shape[dim] = end - start
                    grad_slice = input_grad.reshape(grad_slice_shape)

                    # Insert gradient slice into the right position
                    grad_idx = [slice(None)] * grad.ndim
                    grad_idx[dim] = slice(start, end)
                    grad[tuple(grad_idx)] = grad_slice

                    # Accumulate gradients
                    if input.grad is None:
                        input.grad = grad
                    else:
                        input.grad += grad

            requires_grad = input.requires_grad and Tensor.build_graph_enabled()
            out = Tensor(
                slice_data,
                requires_grad=requires_grad,
                grad_fn=_chunk_backward if requires_grad else None,
                grad_fn_name="<ChunkBackward>" if requires_grad else None,
                device=input.device
            )

            if requires_grad:
                out._add_parents(input)

            out_tensors.append(out)

        return out_tensors

    ############################
    ### REDUCTION OPERATIONS ###
    ############################
    """
    These operations compute some value across
    one or more dimensions of a single tensor!
    """
    def sum(self, dim=None, keepdims=False):
        """
        Sum across a dimension.
        Forward: output = self.data.sum(axis=dim, keepdims=keepdims)
        Backward: distribute incoming gradient to all elements along summed axes.
        """
        out_data = self.data.sum(axis=dim, keepdims=keepdims)

        def _sum_backward(input_grad):
            if self.requires_grad:
                # Broadcast input gradient to input shape
                self_grad = np.broadcast_to(input_grad, self.shape)
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_sum_backward if requires_grad else None,
            grad_fn_name="<SumBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def cumsum(self, dim=None):
        """
        Cumulative sum along a dimension.
        Forward: out[i] = sum_{j<=i} x[j]
        Backward: grad_x[i] = sum_{j>=i} grad_y[j]

        if our inputs are [a,b,c,d]
        then our cumulative sum is:
        [a, a+b, a+b+c, a+b+c+d]

        Then in our backward pass we have grads 
        [g1, g2, g3, g4]

        and so then we see that a contributed to g1, g2, g3, and g4
        b contributed to g2, g3, g4
        c contributed to g3, g4
        d contributd to g4
        """
        out_data = self.data.cumsum(axis=dim)

        def _cumsum_backward(input_grad):
            if self.requires_grad:
                # Reverse, cumsum, then reverse again
                grad_reversed = np.flip(input_grad, axis=dim)
                grad_input = np.cumsum(grad_reversed, axis=dim)
                grad_input = np.flip(grad_input, axis=dim)

                if self.grad is None:
                    self.grad = grad_input
                else:
                    self.grad += grad_input

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_cumsum_backward if requires_grad else None,
            grad_fn_name="<CumsumBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)

        return out

    def mean(self, dim=None, keepdims=False):
        """
        Mean across a dimension.
        Forward: output = self.data.mean(axis=dim, keepdims=keepdims)
        Backward: broadcast incoming gradient and divide by number of elements summed.
        """

        ### if no dim is provided we reduce on all dims ###
        if dim is None:
            dim = tuple(range(len(self.shape)))

        out_data = self.data.mean(axis=dim, keepdims=keepdims)

        def _mean_backward(input_grad):

            if self.requires_grad:
                # Compute number of elements reduced over
                dims = dim if isinstance(dim, tuple) else (dim,)
                num_vals_averaged = np.prod([self.shape[d] for d in dims])

                # Broadcast upstream gradient and scale
                self_grad = np.broadcast_to(input_grad, self.shape) / num_vals_averaged
                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad

                self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_mean_backward if requires_grad else None,
            grad_fn_name="<MeanBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def var(self, dim=None, keepdims=False):
        """
        Variance along a given dimension.
        Var = mean((x - mean(x))^2)
        
        Backward: dVar/dx = 2 * (x - mean(x)) / N * input_grad
        """

        ### if no dim is provided we reduce on all dims ###
        if dim is None:
            dim = tuple(range(len(self.shape)))

        # Forward pass
        mean_vals = self.data.mean(axis=dim, keepdims=True)
        var_vals = ((self.data - mean_vals) ** 2).mean(axis=dim, keepdims=keepdims)

        def _var_backward(input_grad):
            if self.requires_grad:
                # Broadcast input gradient to input shape
                input_grad_broadcast = np.broadcast_to(input_grad, self.shape)
                
                # Number of elements reduced over
                dims = dim if isinstance(dim, tuple) else (dim,)
                num_vals_reduced = np.prod([self.shape[d] for d in dims])
                
                # Gradient formula: 2/N * (x - mean(x)) * upstream gradient
                centered = self.data - mean_vals
                self_grad = 2.0 * centered * input_grad_broadcast / num_vals_reduced

                if self.grad is None:
                    self.grad = self_grad
                else:
                    self.grad += self_grad
                
                self_grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            var_vals,
            requires_grad=requires_grad,
            grad_fn=_var_backward if requires_grad else None,
            grad_fn_name="<VarBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)

        return out

    def max(self, dim=None, keepdims=False):
        """
        Compute max along axis with autograd support.
        Only propagate gradient to the positions where the maximum occurred.
        """

        ### if no dim is provided we reduce on all dims ###
        if dim is None:
            dim = tuple(range(len(self.shape)))
            
        out_data = self.data.max(axis=dim, keepdims=keepdims)

        def _max_backward(input_grad):
            
            if self.requires_grad:

                grad = self.xp.zeros_like(self.data, dtype=self.data.dtype)

                # Broadcast input_grad if needed
                if dim is not None and not keepdims:
                    input_grad = np.expand_dims(input_grad, dim)

                # Broadcast to match self shape
                input_grad = input_grad * np.ones_like(self.data, dtype=self.data.dtype)
                
                # Only propagate gradient to positions where max occurred
                mask = (self.data == (out_data if keepdims else np.expand_dims(out_data, dim)))
                grad += input_grad * mask
    
                # Call backward on self
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

                grad = None

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_max_backward if requires_grad else None,
            grad_fn_name="<MaxBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def argmax(self, dim=-1):
        """
        Compute the indices of the maximum value along a dimension.
        Note: argmax is non-differentiable.
        """
        out_data = self.data.argmax(axis=dim)

        def _argmax_backward(input_grad):
            # No gradient flows through argmax
            return ap.Array.zeros_like(self.data, dtype=self.data.dtype)

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_argmax_backward if requires_grad else None,
            grad_fn_name="<ArgmaxBackward>" if requires_grad else None,
            device=self.device
        )

        if requires_grad:
            out._add_parents(self)

        return out

    #################
    ### OTHER OPS ###
    #################
    """
    These are just a collection of helpful operations that come in handy!
    """
    def masked_fill(self, mask, value):
    
        # forward
        out_data = np.where(mask.data, value, self.data)

        def _masked_fill_backward(input_grad):
            if self.requires_grad:
                # Only pass gradient where mask == False
                grad = np.where(mask.data, 0, input_grad)

                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_masked_fill_backward if requires_grad else None,
            grad_fn_name="<MaskedFillBackward>" if requires_grad else None,
            device=self.device,
        )

        if requires_grad:
            out._add_parents(self)

        return out
    
    def sort(self, dim=-1, descending=False):
        
        # Forward: sort the data and get indices
        sorted_indices = np.argsort(self.data, axis=dim)
        if descending:
            sorted_indices = np.flip(sorted_indices, axis=dim)
        
        out_data = np.take_along_axis(self.data, sorted_indices, axis=dim)
        
        def _sort_backward(input_grad):
            inv_indices = np.argsort(sorted_indices, axis=dim)
            input_grad = np.take_along_axis(input_grad, inv_indices, axis=dim)
            if self.grad is None:
                self.grad = input_grad
            else:
                self.grad += input_grad

        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_sort_backward if requires_grad else None,
            grad_fn_name="<SortBackward>" if requires_grad else None,
            device=self.device,
        )
        
        if requires_grad:
            out._add_parents(self)
        
        # Return values and indices (wrapped in Tensor for indices)
        indices_tensor = Tensor(sorted_indices, requires_grad=False, device=device, dtype=int32)

        return out, indices_tensor

    def argsort(self, dim=-1, descending=False):
        
        sorted_indices = np.argsort(self.data, axis=dim)
        if descending:
            sorted_indices = np.flip(sorted_indices)
        
        def _argsort_backward(input_grad):
            # No gradient flows through argsort
            return ap.Array.zeros_like(self.data, dtype=self.data.dtype)
        
        requires_grad = self.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            sorted_indices,
            requires_grad=requires_grad,
            grad_fn=_argsort_backward if requires_grad else None,
            grad_fn_name="<ArgsortBackward>" if requires_grad else None,
            device=self.device,
            dtype=int32
        )

        if requires_grad:
            out._add_parents(self)

        return out

    def _add_parents(self, *parents):
        """
        Store references to parent tensors as weakrefs.
        """

        if not isinstance(parents, (list, tuple)):
            parents = (parents)
        self._parents = tuple(weakref.ref(p) for p in parents if p is not None)

    def item(self):
        if self.data.size != 1:
            raise ValueError("only one element tensors can be converted to a Python scalar")
        if "cuda" in self.device:
            return self.data.flatten()[0].get().item()
        else:
            return self.data.flatten()[0].item()

    def astype(self, dtype):

        ### Update the Tensors Dtype using setter ###
        self.data = self._data.astype(dtype)

        return self
        
    def contiguous(self):
        ### To map to contiguous we have to use the proper backend here ###
        self.data = self.xp.ascontiguousarray(self.data._array, dtype=self.data.dtype)
        return self
    
    def detach(self):

        detached = Tensor(
            self.data,  
            requires_grad=False,
            grad_fn=None,
            grad_fn_name=None,
            device=self.device
        )

        return detached
    
    def numpy(self):
        return self.data.asnumpy()
    
    def __len__(self):
        return self.shape[0]

##################################################################
### TENSOR FACTORY ###############################################
##################################################################                    
### This is just a warpper on our Class methods in _array.py! ####
##################################################################

def _tensor_from_array(func, device="cpu", dtype=None, requires_grad=False):
    arr = func() 
    return Tensor(arr, device=device, dtype=dtype or str(arr.dtype), requires_grad=requires_grad)

# Shape-based factories
def zeros(*shape, device="cpu", dtype=float32, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    return _tensor_from_array(lambda: ap.Array.zeros(shape, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def ones(*shape, device="cpu", dtype=float32, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    return _tensor_from_array(lambda: ap.Array.ones(shape, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def empty(*shape, device="cpu", dtype=float32, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    return _tensor_from_array(lambda: ap.Array.empty(shape, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def full(*shape, fill_value, device="cpu", dtype=float32, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    return _tensor_from_array(lambda: ap.Array.full(shape, fill_value, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)
# Sequences 
def arange(start=None, end=None, step=1, *, device="cpu", dtype="int32", requires_grad=False):
    if end is None:
        if start is None:
            raise TypeError("arange() missing required argument 'end'")
        start, end = 0, start

    return _tensor_from_array(
        lambda: ap.Array.arange(start, end, step, device=device, dtype=dtype),
        device=device,
        dtype=dtype,
        requires_grad=requires_grad
    )

def linspace(start, end, num=50, device="cpu", dtype=float32, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.linspace(start, end, num, device, dtype), 
                              requires_grad=requires_grad)
# Eye and triangular
def eye(N, M=None, k=0, device="cpu", dtype=float32, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.eye(N, M=M, k=k, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def tril(x, k=0, device="cpu", dtype=float32, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.tril(x.data, k=k, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)
# Random arrays
def randn(*shape, device="cpu", dtype=float32, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    return _tensor_from_array(lambda: ap.Array.randn(shape, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def rand(*shape, device="cpu", dtype=float32, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = shape[0]
    return _tensor_from_array(lambda: ap.Array.rand(shape, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def randint(low, high, shape, device="cpu", dtype=int32, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.randint(low=low, high=high, shape=shape, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def randn_like(tensor, device=None, dtype=None, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.randn_like(tensor.data, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def rand_like(tensor, device=None, dtype=None, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.rand_like(tensor.data, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)
# Like zeros/ones_like
def zeros_like(tensor, device=None, dtype=None, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.zeros_like(tensor.data, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def ones_like(tensor, device=None, dtype=None, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.ones_like(tensor.data, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def empty_like(tensor, device=None, dtype=None, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.empty_like(tensor.data, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)

def full_like(tensor, fill_value, device=None, dtype=None, requires_grad=False):
    return _tensor_from_array(lambda: ap.Array.full_like(tensor.data, fill_value, device=device, dtype=dtype),
                              device=device, dtype=dtype, requires_grad=requires_grad)