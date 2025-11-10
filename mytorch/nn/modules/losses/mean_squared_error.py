from ..base_module import Module
import mytorch.nn.functional as F

class MSELoss(Module):
    def __init__(self, auto=False):
        super().__init__()
        self.auto = auto
        
    def forward(self, pred, labels):
        return F.mse_loss(pred, labels, auto=self.auto)