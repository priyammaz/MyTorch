from ..base_module import Module
import mytorch.nn.functional as F

class LeakyReLU(Module):
    def __init__(self, auto=False, fused=False):
        super().__init__()
        self.auto = auto
        self.fused = fused
    
    def forward(self, x):
        return F.leaky_relu(x, auto=self.auto, fused=self.fused)
    

