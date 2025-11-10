import math
import mytorch
from ..base_module import Module
from ... import initializations as init
import mytorch.nn.functional as F

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, fused=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fused = fused

        limit = 1.0 / math.sqrt(embedding_dim)
        self.weight = mytorch.zeros((num_embeddings, embedding_dim), requires_grad=True)
        init.uniform_(self.weight, -limit, limit)

    def __call__(self, indices):
        return self.forward(indices)
    
    def __repr__(self):
        return f"Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})"
    
    def _extra_repr(self):
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
    
    def forward(self, indices):
        return F.embedding(indices=indices, weight=self.weight, fused=self.fused)
    
    