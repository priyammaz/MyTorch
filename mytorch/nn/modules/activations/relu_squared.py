from ..base_module import Module
import mytorch.nn.functional as F

class ReLUSquared(Module):
    def __init__(self, auto=False, fused=False):
        super().__init__()
        self.auto = auto
        self.fused = fused
    
    def forward(self, x):
        return F.relu_squared(x, auto=self.auto, fused=self.fused)
    

