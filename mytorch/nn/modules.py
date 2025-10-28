"""
All modules will be set on the CPU (numpy) 
but can easily be moved to cuda internally
using .to()!
"""
import math
import numpy as np
import cupy as cp
from ..tensor import Tensor, zeros, ones
from . import functional as F
from . import initializations as init

try:
    import triton
    FUSED_AVAIL = True
except:
    FUSED_AVAIL = False

######################
### Generic Module ###
######################
class Module:
    def __init__(self):
        
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self.training = True

    def __setattr__(self, name, value):

        if "_modules" not in self.__dict__:
            if isinstance(value, (Module, Tensor)):
                raise RuntimeError(
                    f"Cannot assign {type(value).__name__} to '{name}' "
                    "before calling super().__init__() in your Module subclass."
                )
            return object.__setattr__(self, name, value)

        # Register parameters
        if isinstance(value, Tensor):
            self._parameters[name] = value

        # Register submodules (including ModuleList)
        elif isinstance(value, Module):
            self._modules[name] = value

        # Always assign normally
        return object.__setattr__(self, name, value)

    def parameters(self, memo=None):

        """
        Returns a generator of parameters and only returns unique tensors. 
        Referenced tensors (like in weight tying) are not included here!
        """

        if memo is None:
            memo = set()

        for param in self._parameters.values():

            if "cuda" in param.device:
                ptr = param.data.data.ptr
            else:
                ptr = id(param.data)

            if param is not None and ptr not in memo:
                memo.add(ptr)
                yield param

        for module in self._modules.values():
            yield from module.parameters(memo)

    def _parameters_no_dedup(self, prefix=""):
        """Yield all parameters including duplicates"""
        for param in self._parameters.values():
            yield param

        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix="", memo=None):

        if memo is None:
            memo = set()

        for name, param in self._parameters.items():

            if "cuda" in param.device:
                ptr = param.data.data.ptr
            else:
                ptr = id(param.data)

            if param is not None and ptr not in memo:
                memo.add(ptr)
                full = f"{prefix}{name}" if prefix else name
                yield full, param

        for name, m in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from m.named_parameters(sub_prefix, memo)

    def _named_parameters_no_dedup(self, prefix=""):
        """Yield all parameters with names, including duplicates"""
        for name, param in self._parameters.items():
            full_name = f"{prefix}{name}" if prefix else name
            yield full_name, param
        for name, module in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from module._named_parameters_no_dedup(sub_prefix)
            
    def register_buffer(self, name, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError("Buffers must be Tensors")
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def named_buffers(self, prefix="", memo=None):

        if memo is None:
            memo = set()
        
        for name, buf in self._buffers.items():
            if buf is not None:
                # Use same deduplication logic as parameters
                if "cuda" in buf.device:
                    ptr = buf.data.data.ptr
                else:
                    ptr = id(buf.data)
                
                if ptr not in memo:
                    memo.add(ptr)
                    full = f"{prefix}{name}" if prefix else name
                    yield full, buf
        
        # Recursively get buffers from submodules
        for name, m in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from m.named_buffers(sub_prefix, memo)
    
    def _named_buffers_no_dedup(self, prefix=""):
        """Yield all buffers with names, including duplicates"""
        for name, param in self._buffers.items():
            full_name = f"{prefix}{name}" if prefix else name
            yield full_name, param
        for name, module in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from module._named_parameters_no_dedup(sub_prefix)

    def to(self, device):
        """
        Moves all parameters and buffers of this module to the given device.
        """
        # Move parameters
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = param.to(device)
                object.__setattr__(self, name, self._parameters[name])

        # Move buffers
        for name, buf in self._buffers.items():
            if buf is not None:
                self._buffers[name] = buf.to(device)
                object.__setattr__(self, name, self._buffers[name])

        # Recursively move submodules
        for m in self._modules.values():
            m.to(device)

        return self

    def apply(self, fn):
        
        """
        Function to apply to all modules (mainly for weight init)
        """
        fn(self)
        for m in self._modules.values():
            m.apply(fn)
        return self

    def _extra_repr(self):
        return ""

    def _repr(self, indent=0):
        model_name = self.__class__.__name__
        ind = "   " * indent
        extra = self._extra_repr()
        if not self._modules:  # leaf
            return f"{ind}{model_name}({extra})\n"
        s = f"{ind}{model_name}(\n"
        for key, val in self._modules.items():
            s += f"{ind}  ({key}): {val._repr(indent + 1).lstrip()}"
        s += f"{ind})\n"
        return s

    def __repr__(self):
        return self._repr(indent=0).rstrip()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def state_dict(self):
        """
        Returns a dictionary of all parameters and buffers as NumPy arrays
        """
        state = {}
        
        # Save all parameters recursively
        for name, param in self._named_parameters_no_dedup():
            if "cuda" in param.device:
                state[name] = param.numpy()
            else:
                state[name] = param.numpy()
        
        # Save all buffers recursively
        for name, buf in self._named_buffers_no_dedup():
            if buf is not None:
                if "cuda" in buf.device:
                    state[name] = buf.numpy()
                else:
                    state[name] = buf.numpy()
        
        return state
    
    def load_state_dict(self, state_dict, strict=True, device="cpu"):
        """
        Loads parameters and buffers from a state_dict (NumPy arrays).
        
        Args:
            state_dict: dict of parameter/buffer names to NumPy arrays.
            strict: whether to enforce exact key match.
            device: "cpu" or "cuda" — where to load the arrays.
        """
        missing_keys = []
        unexpected_keys = list(state_dict.keys())
        
        # Utility to move arrays to correct backend
        # Default to float32 here
        def to_device(array):
            if device == "cuda":
                return cp.asarray(array, dtype=cp.float32)
            else:
                return array.astype(np.float32)
        
        # Load parameters recursively
        for name, param in self._named_parameters_no_dedup():
            if name in state_dict:
                param.data[:] = to_device(state_dict[name])
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)
        
        # Load buffers recursively
        for name, buf in self._named_buffers_no_dedup():
            if name in state_dict:
                buf.data[:] = to_device(state_dict[name])
                if name in unexpected_keys:
                    unexpected_keys.remove(name)
            else:
                missing_keys.append(name)
        
        if strict:
            error_msgs = []
            if missing_keys:
                error_msgs.append(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                error_msgs.append(f"Unexpected keys: {unexpected_keys}")
            if error_msgs:
                raise RuntimeError("Error(s) in loading state_dict:\n" + "\n".join(error_msgs))
            else:
                return "<All Keys Matched Successfully>"
        else:
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                return "<All Keys Matched Successfully>"
            else:
                return missing_keys, unexpected_keys

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()

###############################
### STACK MODULES INTO LIST ###
###############################
class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._modules_list = []
        if modules is not None:
            for m in modules:
                self.append(m)

    def append(self, module):
        if not isinstance(module, Module):
            raise TypeError("ModuleList can only contain Module instances")
        self._modules_list.append(module)
        # Assign an integer name so it shows up in _modules for the parent
        setattr(self, str(len(self._modules_list)-1), module)

    def __getitem__(self, idx):
        return self._modules_list[idx]

    def __len__(self):
        return len(self._modules_list)

    def __iter__(self):
        return iter(self._modules_list)

    def __repr__(self):
        out = "ModuleList([\n"
        for i, layer in enumerate(self._modules_list):
            out += f"  ({i}): {layer}\n"
        out += "])"
        return out

class Sequential(Module):
    def __init__(self, *modules):
        """
        Sequential container: applies modules in the order they are passed.
        Usage:
            net = Sequential(
                Linear(10, 20),
                ReLU(),
                Linear(20, 5)
            )
        """
        super().__init__()
        self.layers = ModuleList(modules)  # store in a ModuleList

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def _extra_repr(self):
        return ", ".join([layer.__class__.__name__ for layer in self.layers])

#######################
### STANDARD LAYERS ###
#######################
class Linear(Module):
    """
    Optimized Linear layer with manual backward.
    """
    def __init__(self, in_features, out_features, bias=True, auto=False, fused=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused = fused
        self.bias = bias
        self.auto = auto

        if self.auto and fused:
            raise Exception("Modules with full autograd cannot be fused, turn off fused")

        if fused and not FUSED_AVAIL:
            raise Exception("Triton Installation Necessary for Fused Ops")
        
        self.weight = zeros((out_features, in_features), requires_grad=True)
        k = math.sqrt(1 / in_features)
        init.uniform_(self.weight, -k, k)

        if self.bias:
            self.use_bias = True
            self.bias = zeros((out_features,), requires_grad=True)
            init.uniform_(self.bias)
        else:
            self.use_bias = False
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"
    
    def _extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"
    
    def forward(self, x: Tensor):
        output = F.linear(x, weight=self.weight, bias=self.bias, auto=self.auto, fused=self.fused)
        return output
    
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        limit = 1.0 / np.sqrt(embedding_dim)
        self.weight = zeros((num_embeddings, embedding_dim), requires_grad=True)
        init.uniform_(self.weight, -limit, limit)

    def __call__(self, indices):
        return self.forward(indices)
    
    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
    
    def _extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
    
    def forward(self, indices):
        embeds = self.weight[indices]
        return embeds
    
class Dropout(Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.p = dropout_p
        self.training = True

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Dropout(p={self.p})"
    
    def _extra_repr(self):
        return f"p={self.p}"
    
    def forward(self, x):
        enable_dropout = self.training and (self.p > 0)
        return F.dropout(x, self.p, enable_dropout)

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Kaiming initialization (like PyTorch default for conv1d)
        self.weight = zeros(shape=(out_channels, in_channels, kernel_size), requires_grad=True)
        init.kaiming_uniform_(self.weight)

        if bias:
            self.use_bias = True
            self.bias = zeros(shape=(out_channels,), requires_grad=True)
        else:
            self.use_bias = False
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        args = [f"{self.in_channels}", f"{self.out_channels}", f"kernel_size={self.kernel_size}"]
        
        if self.stride != 1:
            args.append(f"stride={self.stride}")
        if self.padding != 0:
            args.append(f"padding={self.padding}")
        if not self.use_bias:
            args.append("bias=False")
    
        return "Conv1d(" + ", ".join(args) + ")"
    
    def _extra_repr(self):
        return (
            f"{self.in_channels}, "
            f"{self.out_channels}, "
            f"kernel_size=({self.kernel_size}), "
            f"stride={self.stride}, " 
            f"padding={self.padding}, bias={self.use_bias}"
        )

    def forward(self, x: Tensor):
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, fused=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.dilation = dilation
        self.fused = fused

        # Kaiming initialization (like PyTorch default for conv2d)
        self.weight = zeros((out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        init.kaiming_uniform_(self.weight)

        if bias:
            self.use_bias = True
            self.bias = zeros((out_channels,), requires_grad=True)
        else:
            self.use_bias = False
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        args = [f"{self.in_channels}", f"{self.out_channels}", f"kernel_size={self.kernel_size}"]
        
        if self.stride != 1:
            args.append(f"stride=({self.stride}, {self.stride})")
        if self.padding != 0:
            args.append(f"padding=({self.padding}, {self.padding})")
        if not self.use_bias:
            args.append("bias=False")
    
        return "Conv2d(" + ", ".join(args) + ")"
    
    def _extra_repr(self):
        return (
            f"{self.in_channels}, "
            f"{self.out_channels}, "
            f"kernel_size=({self.kernel_size}, {self.kernel_size}), "
            f"stride=({self.stride},{self.stride}), " 
            f"padding=({self.padding}, {self.padding}), bias={self.use_bias}"
        )

    def forward(self, x: Tensor):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.fused)

class ConvTranspose1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias_flag = bias

        # Initialize weights (Kaiming)
        self.weight = zeros(shape=(in_channels, out_channels, kernel_size), requires_grad=True)
        init.kaiming_uniform_(self.weight)

        if bias:
            self.use_bias = True
            self.bias = zeros(shape=(out_channels,), requires_grad=True)
        else:
            self.use_bias = False
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        args = [f"{self.in_channels}", f"{self.out_channels}", f"kernel_size={self.kernel_size}"]
        if self.stride != 1:
            args.append(f"stride={self.stride}")
        if self.padding != 0:
            args.append(f"padding={self.padding}")
        if self.output_padding != 0:
            args.append(f"output_padding={self.output_padding}")
        if not self.use_bias:
            args.append("bias=False")
        return "ConvTranspose1d(" + ", ".join(args) + ")"

    def _extra_repr(self):
        return (
            f"{self.in_channels}, "
            f"{self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"output_padding={self.output_padding}, "
            f"bias={self.use_bias}"
        )

    def forward(self, x: Tensor):
        return F.conv_transpose1d(
            x,
            self.weight,
            self.bias if self.use_bias else None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )
    
class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias

        # Kaiming initialization (like PyTorch default for conv_transpose2d)
        self.weight = zeros(shape=(in_channels, out_channels, kernel_size, kernel_size), requires_grad=True)
        init.kaiming_uniform_(self.weight)

        if bias:
            self.use_bias = True
            self.bias = zeros(shape=(out_channels,), requires_grad=True)
        else:
            self.use_bias = False
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        args = [f"{self.in_channels}", f"{self.out_channels}", f"kernel_size={self.kernel_size}"]
        
        if self.stride != 1:
            args.append(f"stride=({self.stride}, {self.stride})")
        if self.padding != 0:
            args.append(f"padding=({self.padding}, {self.padding})")
        if self.output_padding != 0:
            args.append(f"output_padding=({self.output_padding}, {self.output_padding})")
        if not self.use_bias:
            args.append("bias=False")
    
        return "ConvTranspose2d(" + ", ".join(args) + ")"
    
    def _extra_repr(self):
        return (
            f"{self.in_channels}, "
            f"{self.out_channels}, "
            f"kernel_size=({self.kernel_size}, {self.kernel_size}), "
            f"stride=({self.stride},{self.stride}), " 
            f"padding=({self.padding}, {self.padding}), "
            f"output_padding=({self.output_padding}, {self.output_padding}), bias={self.use_bias}"
        )

    def forward(self, x: Tensor):
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding)
    
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        args = [f"kernel_size=({self.kernel_size}, {self.kernel_size})"]
        
        if self.stride != self.kernel_size:
            args.append(f"stride=({self.stride}, {self.stride})")
        if self.padding != 0:
            args.append(f"padding=({self.padding}, {self.padding})")
    
        return "MaxPool2d(" + ", ".join(args) + ")"
    
    def _extra_repr(self):
        return (
            f"kernel_size=({self.kernel_size}, {self.kernel_size}), "
            f"stride=({self.stride}, {self.stride}), "
            f"padding=({self.padding}, {self.padding})"
        )

    def forward(self, x):
        return F.maxpool2d(x, self.kernel_size, self.stride, self.padding)
    
class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        args = [f"kernel_size=({self.kernel_size}, {self.kernel_size})"]
        
        if self.stride != self.kernel_size:
            args.append(f"stride=({self.stride}, {self.stride})")
        if self.padding != 0:
            args.append(f"padding=({self.padding}, {self.padding})")
    
        return "AvgPool2d(" + ", ".join(args) + ")"
    
    def _extra_repr(self):
        return (
            f"kernel_size=({self.kernel_size}, {self.kernel_size}), "
            f"stride=({self.stride}, {self.stride}), "
            f"padding=({self.padding}, {self.padding})"
        )

    def forward(self, x):
        return F.averagepool2d(x, self.kernel_size, self.stride, self.padding)
    
class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"AdaptiveAvgPool2d(output_size={self.output_size})"

    def _extra_repr(self):
        return f"output_size={self.output_size}"

    def forward(self, x):
        """
        Compute kernel_size and stride based on input size and desired output size,
        then call the existing averagepool2d function.
        """
        B, C, H, W = x.data.shape

        # Compute kernel_size and stride
        # Use floor division to ensure integer kernel sizes
        K = H - (self.output_size[0] - 1)
        S = 1
        padding = 0  # No padding needed, as kernel size is adapted

        # Call the existing averagepool2d function
        return F.averagepool2d(x, kernel_size=K, stride=S, padding=padding)
    
############################
### NORMALIZATION LAYERS ###
############################
class LayerNorm(Module):
    def __init__(self, normalized_shape, bias=True, eps=1e-5, auto=False, fused=False):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.embed_dim = int(np.prod(normalized_shape))
        
        self.auto = auto
        self.fused = fused
        self.eps = eps

        if self.auto and fused:
            raise Exception("Modules with full autograd cannot be fused, turn off fused")

        if fused and not FUSED_AVAIL:
            raise Exception("Triton Installation Necessary for Fused Ops")

        # Learnable parameters: always 1D over the *product* of normalized dims
        self.weight = ones((self.embed_dim,), requires_grad=True)
        self.bias = zeros((self.embed_dim,), requires_grad=True) if bias else None

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        if not x.shape[-len(self.normalized_shape):] == self.normalized_shape:
            raise ValueError(
                f"Expected trailing dimensions to be {self.normalized_shape}, "
                f"but got {x.shape[-len(self.normalized_shape):]}"
            )

        original_shape = x.shape
        norm_ndim = len(self.normalized_shape)

        # Reshape: collapse normalized dims into the last axis
        if norm_ndim > 1:
            new_shape = (*original_shape[:-norm_ndim], self.embed_dim)
            x_reshaped = x.reshape(*new_shape)
        else:
            x_reshaped = x  # already correct shape

        # Call your existing layernorm (expects last dim to be normalized)
        out = F.layernorm(
            x_reshaped,
            self.weight,
            self.bias,
            eps=self.eps,
            training=self.training,
            auto=self.auto,
            fused=self.fused,
        )

        # Reshape back to original
        if norm_ndim > 1:
            out = out.reshape(original_shape)

        return out

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, bias={self.bias is not None})"

    def _extra_repr(self):
        return f"{self.normalized_shape}"


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.weight = ones((num_features, ), requires_grad=True)
        self.bias = zeros((num_features, ), requires_grad=True)

        # Non-trainable buffers
        self.register_buffer("running_mean", ones((num_features, ), requires_grad=False))
        self.register_buffer("running_var", zeros((num_features, ), requires_grad=False))

    def forward(self, x):
        return F.batchnorm(
            input=x,
            weight=self.weight,
            bias=self.bias,
            running_mean=self.running_mean,
            running_var=self.running_var,
            momentum=self.momentum,
            eps=self.eps,
            training=self.training,
        )

    def __repr__(self):
        return f"BatchNorm2d({self.num_features}, eps={self.eps}, momentum={self.momentum})"
    def _extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"
    
############################
### ACTIVATION FUNCTIONS ###
############################
class Sigmoid(Module):
    def __init__(self, auto=False):
        super().__init__()
        self.auto = auto

    def __repr__(self):
        return "Sigmoid()"

    def forward(self, x):
        return F.sigmoid(x, auto=self.auto)
    
class ReLU(Module):
    def __init__(self, auto=False):
        super().__init__()
        self.auto = auto
    
    def forward(self, x):
        return F.relu(x, auto=self.auto)
    
class GELU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.gelu(x)

class Softmax(Module):
    def __init__(self, auto=False, fused=False):
        super().__init__()
        self.auto = auto

        if self.auto and fused:
            raise Exception("Modules with full autograd cannot be fused, turn off fused")

        if fused and not FUSED_AVAIL:
            raise Exception("Triton Installation Necessary for Fused Ops")

        self.fused = fused

    def forward(self, x, dim):
        return F.softmax(x, dim=dim, auto=self.auto, fused=self.fused)
    
######################
### LOSS FUNCTIONS ###
######################
class CrossEntropyLoss(Module):
    def __init__(self, auto=False, fused=False):
        super().__init__()
        self.auto = auto
        
        if self.auto and fused:
            raise Exception("Modules with full autograd cannot be fused, turn off fused")
        
        if fused and not FUSED_AVAIL:
            raise Exception("Triton Installation Necessary for Fused Ops")
        
        self.fused = fused

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, auto=self.auto, fused=self.fused)
    
class MSELoss(Module):
    def __init__(self, auto=False):
        super().__init__()
        self.auto = auto
    def forward(self, pred, labels):
        return F.mse_loss(pred, labels, auto=self.auto)