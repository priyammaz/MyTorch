from ..base_module import Module
import mytorch.nn.functional as F

class GELU(Module):
    def __init__(self, approximate="none", fused=False):
        """
        approximate: 'none' (exact, default) | 'tanh', same as torch.nn.GELU
        """
        super().__init__()
        self.approximate = approximate
        self.fused = fused

    def forward(self, x):
        return F.gelu(x, approximate=self.approximate, fused=self.fused)
