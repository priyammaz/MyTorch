
import math
import cupy as cp
from ..recursive_tensor import Tensor
from . import recursive_functional as F

######################
### Generic Module ###
######################
class Module:
    def __init__(self):
        # use object.__setattr__ to avoid recursion
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self.training = True

    def __setattr__(self, name, value):
        # Catch the common mistake: forgot super().__init__()
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

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix=""):
        for name, p in self._parameters.items():
            full = f"{prefix}{name}" if prefix else name
            yield full, p
        for name, m in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from m.named_parameters(sub_prefix)

    def register_buffer(self, name, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError("Buffers must be Tensors")
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

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
        Returns a dictionary of all parameters as NumPy arrays (CPU-friendly).
        """
        return {name: cp.asnumpy(tensor.data) for name, tensor in self.named_parameters()}
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Loads parameters from a state_dict (NumPy arrays), converting to CuPy.
        """
        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        for name, param in self.named_parameters():
            if name in state_dict:
                # Convert NumPy array to CuPy and copy into Tensor
                param.data[:] = cp.asarray(state_dict[name], dtype=cp.float32)
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
                print("<All Keys Matched Successfully>")

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
    def __init__(self, in_features, out_features, bias=True, auto=False):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.auto = auto

        # Weight initialization
        k = math.sqrt(1 / in_features)
        self.W = Tensor(
            cp.random.uniform(-k, k, size=(in_features, out_features), dtype=cp.float32),
            requires_grad=True
        )

        if self.bias:
            self.b = Tensor(
                cp.random.uniform(-k, k, size=(out_features,), dtype=cp.float32),
                requires_grad=True
            )
        else:
            self.b = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"
    
    def _extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias}"
    
    def forward(self, x: Tensor):
        output = F.linear(x, weight=self.W, bias=self.b, auto=self.auto)
        return output
    
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Tensor((cp.random.randn(num_embeddings, embedding_dim) / cp.sqrt(num_embeddings)).astype(cp.float32), requires_grad=True)

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
        self.keep_p = 1.0 - dropout_p
        self.training = True
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Dropout(p={self.p})"
    
    def _extra_repr(self):
        return f"p={self.p}"
    
    def forward(self, x):
        
        if not self.training or self.p == 0.0:
            return x

        ### This is slow as it creates a tensor the same shape as your activations ###
        mask = cp.random.random_sample(x.shape, dtype=cp.float32)
        mask = mask >= self.p

        ### Reweight Non-Masked Positions ###
        mask = mask / (1.0 - self.p)
        out = x * mask
    
        return out

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Kaiming initialization (like PyTorch default for conv2d)
        k = math.sqrt(3 / (in_channels * kernel_size * kernel_size))
        self.W = Tensor(
            cp.random.uniform(-k, k, size=(out_channels, in_channels, kernel_size, kernel_size), dtype=cp.float32),
            requires_grad=True
        )

        if bias:
            self.b = Tensor(
                cp.random.uniform(-k, k, size=(out_channels,), dtype=cp.float32),
                requires_grad=True
            )
        else:
            self.b = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        args = [f"{self.in_channels}", f"{self.out_channels}", f"kernel_size={self.kernel_size}"]
        
        if self.stride != 1:
            args.append(f"stride=({self.stride}, {self.stride})")
        if self.padding != 0:
            args.append(f"padding=({self.padding}, {self.padding})")
        if not self.bias:
            args.append("bias=False")
    
        return "Conv2d(" + ", ".join(args) + ")"
    
    def _extra_repr(self):
        return (
            f"{self.in_channels}, "
            f"{self.out_channels}, "
            f"kernel_size=({self.kernel_size}, {self.kernel_size}), "
            f"stride=({self.stride},{self.stride}), " 
            f"padding=({self.padding}, {self.padding}), bias={self.bias}"
        )

    def forward(self, x: Tensor):
        return F.conv2d(x, self.W, self.b, self.stride, self.padding)
    
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
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
    
############################
### NORMALIZATION LAYERS ###
############################
class LayerNorm(Module):
    def __init__(self, embed_dim, auto=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.auto = auto
        self.gamma = Tensor(cp.ones(shape=(embed_dim), dtype=cp.float32), requires_grad=True)
        self.beta = Tensor(cp.zeros(shape=(embed_dim), dtype=cp.float32), requires_grad=True)

    def __call__(self, x):
        return self.forward(x)
    
    def __repr__(self):
        return f"LayerNorm({self.embed_dim})"
    
    def _extra_repr(self):
        return f"{self.embed_dim}"
    
    def forward(self, x):
        return F.layernorm(x, self.gamma, self.beta, auto=self.auto)
    
class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = Tensor(cp.ones(num_features, dtype=cp.float32), requires_grad=True)
        self.beta = Tensor(cp.zeros(num_features, dtype=cp.float32), requires_grad=True)

        # Non-trainable buffers
        self.register_buffer("running_mean", Tensor(cp.zeros(num_features, dtype=cp.float32)))
        self.register_buffer("running_var", Tensor(cp.ones(num_features, dtype=cp.float32)))

    def forward(self, x):
        return F.batchnorm(
            input=x,
            gamma=self.gamma,
            beta=self.beta,
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
    def __init__(self, auto=False):
        super().__init__()
        self.auto = auto

    def forward(self, x, dim):
        return F.softmax(x, dim=dim, auto=self.auto)
    
######################
### LOSS FUNCTIONS ###
######################
class CrossEntropyLoss(Module):
    def __init__(self, auto=False):
        super().__init__()
        self.auto = auto
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, auto=self.auto)
    
class MSELoss(Module):
    def __init__(self, auto=False):
        super().__init__()
        self.auto = auto
    def forward(self, pred, labels):
        return F.mse_loss(pred, labels, auto=self.auto)