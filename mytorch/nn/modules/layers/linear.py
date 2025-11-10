import math
import mytorch
from ..base_module import Module
from ... import initializations as init
import mytorch.nn.functional as F

class Linear(Module):

    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 auto=False, 
                 fused=False):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.auto = auto
        self.fused = fused
        
        self.weight = mytorch.zeros((out_features, in_features), requires_grad=True)
        k = math.sqrt(1 / in_features)
        init.uniform_(self.weight, -k, k)

        if self.bias:
            self.use_bias = True
            self.bias = mytorch.zeros((out_features,), requires_grad=True)
            init.uniform_(self.bias)
        else:
            self.use_bias = False
            self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"
    
    def _extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"
    
    def forward(self, x):
        output = F.linear(x, weight=self.weight, bias=self.bias, auto=self.auto, fused=self.fused)
        return output
    