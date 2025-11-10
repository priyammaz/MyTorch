from .base_module import Module

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