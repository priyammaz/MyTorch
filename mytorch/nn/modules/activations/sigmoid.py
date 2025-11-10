from ..base_module import Module
import mytorch.nn.functional as F

class Sigmoid(Module):
    def __init__(self, auto=False, fused=False):
        super().__init__()
        self.auto = auto
        self.fused = fused

    def __repr__(self):
        return "Sigmoid()"

    def forward(self, x):
        return F.sigmoid(x, auto=self.auto, fused=self.fused)
    

