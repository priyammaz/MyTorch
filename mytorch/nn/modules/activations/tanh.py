from ..base_module import Module
import mytorch.nn.functional as F

class Tanh(Module):
    def __init__(self, fused=False):
        super().__init__()
        self.fused = fused
    
    def forward(self, x):
        return F.tanh(x, fused=self.fused)