from ..base_module import Module
import mytorch.nn.functional as F

class Dropout(Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.p = dropout_p
        self.training = True

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Dropout(p={self.p})"
    
    def _extra_repr(self):
        return f"p={self.p}"
    
    def forward(self, x):
        enable_dropout = self.training and (self.p > 0)
        return F.dropout(x, self.p, training=enable_dropout)