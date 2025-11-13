import numpy as np
import cupy as cp
from mytorch import Tensor

class Module:
    def __init__(self):
        
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers = set()
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
                ptr = param.data.data.ptr # <- use cupy array pointer
            else:
                ptr = id(param.data)  # <- use standard id 

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

        """
        Same as parameters, but we also return the name
        of the params with it
        """
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
            
    def register_buffer(self, name, tensor, persistent=True):
        """
        Add buffers (non learnable params) to the module
        """
        if not isinstance(tensor, Tensor):
            raise TypeError("Buffers must be Tensors")
        self._buffers[name] = tensor

        if not persistent:
            self._non_persistent_buffers.add(name)

        object.__setattr__(self, name, tensor)

    def named_buffers(self, prefix="", memo=None, persistent_only=False):
        """
        Get the buffer with its name
        """
        if memo is None:
            memo = set()
        
        for name, buf in self._buffers.items():

            ### Skip non persistent buffers if we only want persistent ###
            if persistent_only and name in self._non_persistent_buffers:
                continue

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
            yield from m.named_buffers(sub_prefix, memo, persistent_only)
    
    def _named_buffers_no_dedup(self, prefix="", persistent_only=False):
        """Yield all buffers with names, including duplicates"""
        for name, param in self._buffers.items():
            if persistent_only and name in self._non_persistent_buffers:
                continue
            full_name = f"{prefix}{name}" if prefix else name
            yield full_name, param

        for name, module in self._modules.items():
            sub_prefix = f"{prefix}{name}." if prefix else f"{name}."
            yield from module._named_buffers_no_dedup(sub_prefix, persistent_only)

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
        """recurisvely update this to get names of params"""
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
        
        # Save all buffers recursively (only want to save persistent buffers)
        for name, buf in self._named_buffers_no_dedup(persistent_only=True):
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
                try:
                    param.data[:] = to_device(state_dict[name])
                except:
                    print(f"Failed to load {name}. Expected {param.shape}, got {state_dict[name].shape}")
                    continue
                unexpected_keys.remove(name)
            else:
                missing_keys.append(name)
        
        # Load buffers recursively
        for name, buf in self._named_buffers_no_dedup(persistent_only=True):
            if name in state_dict:
                buf.data[:] = to_device(state_dict[name])
                try:
                    buf.data[:][:] = to_device(state_dict[name])
                except:
                    print(f"Failed to load {name}. Expected {param.shape}, got {state_dict[name].shape}")
                    continue
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
        """recursively set all modules to training mode"""
        self.training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        """recursively set all modules to eval mode"""
        self.training = False
        for m in self._modules.values():
            m.eval()
