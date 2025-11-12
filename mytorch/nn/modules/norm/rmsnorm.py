# mytorch/nn/modules/rmsnorm.py
import mytorch
from ..base_module import Module
import mytorch.nn.functional as F
import numpy as np

class RMSNorm(Module):
    """
    Root-Mean-Square Layer Normalization.
    """
    def __init__(self, normalized_shape, weight=True, eps=1e-5, auto=False, fused=False):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.embed_dim = int(np.prod(normalized_shape))
        self.auto = auto
        self.fused = fused
        self.eps = eps

        self.weight = mytorch.ones((self.embed_dim,), requires_grad=True) if weight else None
        self.bias = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if not x.shape[-len(self.normalized_shape):] == self.normalized_shape:
            raise ValueError(
                f"Expected trailing dimensions to be {self.normalized_shape}, "
                f"but got {x.shape[-len(self.normalized_shape):]}"
            )

        original_shape = x.shape
        norm_ndim = len(self.normalized_shape)

        if norm_ndim > 1:
            new_shape = (*original_shape[:-norm_ndim], self.embed_grad)
            x_reshaped = x.reshape(*new_shape)
        else:
            x_reshaped = x  

        out = F.rmsnorm(
            x_reshaped,
            self.weight,
            eps=self.eps,
            training=self.training,
            fused=self.fused
        )

        if norm_ndim > 1:
            out = out.reshape(original_shape)

        return out

    def __repr__(self):
        return f"RMSNorm({self.normalized_shape}, weight={self.weight is not None})"

    def _extra_repr(self):
        return f"{self.normalized_shape}"