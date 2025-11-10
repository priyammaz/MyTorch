import mytorch
from ..base_module import Module
import mytorch.nn.functional as F
import numpy as np

class LayerNorm(Module):
    def __init__(self, normalized_shape, bias=True, eps=1e-5, auto=False, fused=False):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.embed_dim = int(np.prod(normalized_shape))
        self.auto = auto
        self.fused = fused
        self.eps = eps

        # Learnable parameters: always 1D over the *product* of normalized dims
        self.weight = mytorch.ones((self.embed_dim,), requires_grad=True)
        self.bias = mytorch.zeros((self.embed_dim,), requires_grad=True) if bias else None

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

        # Reshape: collapse normalized dims into the last axis
        if norm_ndim > 1:
            new_shape = (*original_shape[:-norm_ndim], self.embed_dim)
            x_reshaped = x.reshape(*new_shape)
        else:
            x_reshaped = x  # already correct shape

        # Call your existing layernorm (expects last dim to be normalized)
        out = F.layernorm(
            x_reshaped,
            self.weight,
            self.bias,
            eps=self.eps,
            training=self.training,
            auto=self.auto,
            fused=self.fused,
        )

        # Reshape back to original
        if norm_ndim > 1:
            out = out.reshape(original_shape)

        return out

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, bias={self.bias is not None})"

    def _extra_repr(self):
        return f"{self.normalized_shape}"