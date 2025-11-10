from ..base_module import Module
import mytorch.nn.functional as F

class SiLU(Module):
    def __init__(self, auto=False, fused=False):
        super().__init__()
        self.auto = auto
        self.fused = fused
    
    def forward(self, x):
        return F.silu(x, auto=self.auto, fused=self.fused)