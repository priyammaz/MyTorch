from ..base_module import Module
import mytorch.nn.functional as F

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