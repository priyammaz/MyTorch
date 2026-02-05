from ..base_module import Module
import mytorch.nn.functional as F

class CrossEntropyLoss(Module):
    def __init__(self, fused=False, softcap=None):
        super().__init__()
        self.fused = fused
        self.softcap = softcap

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, fused=self.fused, softcap=self.softcap)