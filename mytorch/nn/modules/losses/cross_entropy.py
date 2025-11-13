from ..base_module import Module
import mytorch.nn.functional as F

class CrossEntropyLoss(Module):
    def __init__(self, auto=False, fused=False, softcap=None):
        super().__init__()
        self.auto = auto
        self.fused = fused
        self.softcap = softcap

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, auto=self.auto, fused=self.fused, softcap=self.softcap)