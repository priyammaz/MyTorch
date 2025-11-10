from ..base_module import Module
import mytorch.nn.functional as F

class GELU(Module):
    def __init__(self, fused=False):
        super().__init__()
        self.fused = fused

    def forward(self, x):
        return F.gelu(x, fused=self.fused)