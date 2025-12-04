from ..base_module import Module
import mytorch.nn.functional as F

class SwiGLU(Module):
    def __init__(self, auto=False, fused=False):
        super().__init__()
        self.auto = auto
        self.fused = fused
    
    def forward(self, input_a, input_b):
        return F.swiglu(input_a, input_b, auto=self.auto, fused=self.fused)