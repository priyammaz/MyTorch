"""
This is an simple as possible GPT model, and will be our first stage
of building MyTorch!

At this point we have defined the following things:

- Linear Layers
- LayerNorm
- Embedding Layers
- Cross Entropy
- Adam Optimizer
"""

import mytorch
import mytorch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class BabyGPTConfig:
    
    vocab_size: int =  65
    max_seq_len: int = 256
    embed_dim: int = 384
    mlp_ratio: int = 4
    num_blocks: int = 6
    num_heads: int = 6
    mlp_dropout_p: float = 0.0
    attn_dropout_p: float = 0.0 
    use_full_auto: bool = False

class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        ### Embeddings for Tokens ###
        self.char_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)

        ### Positional Embeddings ###
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embed_dim)

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        ### Convert Tokens to Embeddings ###
        x = self.char_embeddings(input_ids)
      
        ### Add Positional Information ###
        avail_idx = mytorch.arange(start=0, end=seq_length).to(input_ids.device)
        pos_embed = self.position_embeddings(avail_idx).reshape(1, seq_length, self.config.embed_dim)
        x = x + pos_embed

        return x
    
class Attention(nn.Module):
    """
    Standard causal attention
    """

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        ### Sanity Checks ###
        assert config.embed_dim % config.num_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

        ### Attention Projections ###
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, 
                                  auto=config.use_full_auto)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, 
                                  auto=config.use_full_auto)
        
        ### Dropouts ###
        self.proj_drop = nn.Dropout(dropout_p=config.attn_dropout_p)
        self.attn_drop = nn.Dropout(dropout_p=config.attn_dropout_p)

        ### Causal Mask ###
        causal_positions = (mytorch.tril(mytorch.ones((1,1,config.max_seq_len,config.max_seq_len))) == 0)
        causal_mask = mytorch.masked_fill(mytorch.zeros((1,1,config.max_seq_len, config.max_seq_len)), causal_positions, value=float("-inf"))  
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x):

        batch, seq_len, embed_dim = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*embed_dim]

        # Reshape to multi-head
        qkv = qkv.reshape(batch, seq_len, self.num_heads, 3 * self.head_dim)

        # Transpose to [batch, num_heads, seq_len, 3*head_dim]
        qkv = qkv.transpose(1, 2)

        # Chunk last dim into q, k, v
        q, k, v = mytorch.chunk(qkv, 3, dim=-1)  # each [batch, num_heads, seq_len, head_dim]

        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add -inf to zero out non-causal positions
        scores = scores + self.causal_mask[:, :, :seq_len, :seq_len].astype(scores.data.dtype)

        # Compute softmax
        softmax_attention = mytorch.nn.functional.softmax(scores, dim=-1)
        dropped_attention = self.attn_drop(softmax_attention)

        # Attention output
        output = dropped_attention @ v

        # Return back to original shape 
        output = output.transpose(1, 2).reshape(batch, seq_len, embed_dim)
        
        # Output projection 
        output = self.out_proj(output)
        output = self.proj_drop(output)
        
        return output
    
class FeedForward(nn.Module):
    """
    Regular MLP module after our attention computation. 
    """
    def __init__(self, 
                 config):
        
        super().__init__()

        self.config = config

        hidden_size = config.embed_dim * config.mlp_ratio
        
        self.intermediate_dense = nn.Linear(config.embed_dim, hidden_size, 
                                            auto=config.use_full_auto)
        self.activation = nn.GELU()

        self.intermediate_dropout = nn.Dropout(config.mlp_dropout_p)

        self.out_proj = nn.Linear(hidden_size, config.embed_dim, 
                                  auto=config.use_full_auto)
        
        self.output_dropout = nn.Dropout(config.mlp_dropout_p)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)
        x = self.out_proj(x)
        x = self.output_dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()      

        self.config = config
        self.embed_dim = config.embed_dim
        self.attention = Attention(config)
        
        self.layernorm1 = nn.LayerNorm(config.embed_dim, auto=config.use_full_auto)
        
        self.feedforward = FeedForward(config)
        self.layernorm2 = nn.LayerNorm(config.embed_dim, auto=config.use_full_auto)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.feedforward(self.layernorm2(x))
        return x

class BabyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embeddings = Embeddings(config)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config)

            for _ in range(config.num_blocks)
        ])

        self.final_layer_norm = nn.LayerNorm(config.embed_dim, auto=config.use_full_auto)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, auto=config.use_full_auto)

        ### Initialize Weights ###
        self.apply(_init_weights)
        for name, param in self.named_parameters():
            if "out_proj" in name:
                mytorch.nn.init.normal_(param, mean=0.0, std=(0.02/np.sqrt(2 * config.num_blocks)))

    def forward(self, x):

        ### Get our embeddings ###
        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_layer_norm(x)
        x = self.lm_head(x)

        return x

def _init_weights(module):
    if isinstance(module, nn.Linear):
        mytorch.nn.init.normal_(module.weight, mean=0, std=0.02)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        mytorch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            mytorch.nn.init.ones_(module.weight)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)