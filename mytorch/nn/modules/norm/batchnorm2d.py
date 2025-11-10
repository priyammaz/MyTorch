import mytorch
from ..base_module import Module
import mytorch.nn.functional as F

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.weight = mytorch.ones((num_features, ), requires_grad=True)
        self.bias = mytorch.zeros((num_features, ), requires_grad=True)

        # Non-trainable buffers
        self.register_buffer("running_mean", mytorch.ones((num_features, ), requires_grad=False))
        self.register_buffer("running_var", mytorch.zeros((num_features, ), requires_grad=False))

    def forward(self, x):
        return F.batchnorm(
            input=x,
            weight=self.weight,
            bias=self.bias,
            running_mean=self.running_mean,
            running_var=self.running_var,
            momentum=self.momentum,
            eps=self.eps,
            training=self.training,
        )

    def __repr__(self):
        return f"BatchNorm2d({self.num_features}, eps={self.eps}, momentum={self.momentum})"
    def _extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}"