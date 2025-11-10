import mytorch
from ..base_module import Module
from ... import initializations as init
import mytorch.nn.functional as F

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
        self.weight = mytorch.zeros(shape=(in_channels, out_channels, kernel_size, kernel_size), requires_grad=True)
        init.kaiming_uniform_(self.weight)

        if bias:
            self.use_bias = True
            self.bias = mytorch.zeros(shape=(out_channels,), requires_grad=True)
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

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding)
    