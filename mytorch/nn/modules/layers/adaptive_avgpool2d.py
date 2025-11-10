from ..base_module import Module
import mytorch.nn.functional as F

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