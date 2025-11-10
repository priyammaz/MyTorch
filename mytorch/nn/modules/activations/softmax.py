from ..base_module import Module
import mytorch.nn.functional as F

class Softmax(Module):
    def __init__(self, auto=False, fused=False):
        super().__init__()
        self.auto = auto
        self.fused = fused

    def forward(self, x, dim):
        return F.softmax(x, dim=dim, auto=self.auto, fused=self.fused)