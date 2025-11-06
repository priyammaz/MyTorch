import numpy as np
import mytorch
import mytorch.nn as nn
import mytorch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPT2Config:
    
    vocab_size: int =  50257
    max_seq_len: int = 1024
    embed_dim: int = 768
    mlp_ratio: int = 4
    num_blocks: int = 12
    num_heads: int = 12
    mlp_dropout_p: float = 0.0
    attn_dropout_p: float = 0.0 # Currently not supported for our flash attn
    use_bias: bool = False
    use_full_auto: bool = False
    use_fused_ops: bool = False

class Embeddings(nn.Module):

    def __init__(self, vocab_size, embed_dim, context_length, fused):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.fused = fused

        ### Embeddings for Tokens ###
        self.char_embeddings = nn.Embedding(vocab_size, embed_dim, fused=self.fused)

        ### Positional Embeddings ###
        self.position_embeddings = nn.Embedding(context_length, embed_dim, fused=self.fused)

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        ### Convert Tokens to Embeddings ###
        x = self.char_embeddings(input_ids)

        ### Add Positional Information ###
        avail_idx = mytorch.arange(start=0, end=seq_length).to(input_ids.device)
        pos_embed = self.position_embeddings(avail_idx).reshape(1, seq_length, self.embed_dim)
        x = x + pos_embed

        return x

class Attention(nn.Module):

    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 attn_dropout_p=0.1, 
                 context_length=1024,
                 use_bias=True,
                 auto=False, 
                 fused=False):
        
        super().__init__()

        ### Sanity Checks ###
        assert embed_dim % num_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.fused = fused

        ### Attention Projections ###
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias, auto=auto)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias, auto=auto)
        self.proj_drop = nn.Dropout(dropout_p=attn_dropout_p)

        if not self.fused:
            self.attn_drop = nn.Dropout(dropout_p=attn_dropout_p) # Currently only support attn dropout in non-fused
            self.softmax = nn.Softmax(auto=auto, fused=fused)

            ### If we are not using fused attention we need to manually pass in our 
            ### attention mask! So lets just save it as a buffer right here!
            causal_positions = (mytorch.tril(mytorch.ones((1,1,context_length,context_length))) == 0)
            causal_mask = mytorch.masked_fill(mytorch.zeros((1,1,context_length, context_length)), causal_positions, value=float("-inf"))  
            self.register_buffer("causal_mask", causal_mask)

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

        if not self.fused:

            # Compute attention scores
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
     
            masked_scores = scores + self.causal_mask[:, :, :seq_len, :seq_len].astype(scores.data.dtype)

            softmax_attention = self.softmax(masked_scores, dim=-1)
            dropped_attention = self.attn_drop(softmax_attention)

            # Attention output
            output = dropped_attention @ v

        else:
   
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
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
                 embed_dim, 
                 mlp_ratio=4, 
                 mlp_dropout_p=0.1,
                 use_bias=True,
                 auto=False):
        super().__init__()
        hidden_size = embed_dim * mlp_ratio

        self.intermediate_dense = nn.Linear(embed_dim, hidden_size, bias=use_bias, auto=auto)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(mlp_dropout_p)

        self.out_proj = nn.Linear(hidden_size, embed_dim, bias=use_bias, auto=auto)
        self.output_dropout = nn.Dropout(mlp_dropout_p)

    def forward(self, x):

        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)

        x = self.out_proj(x)
        x = self.output_dropout(x)

        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 dropout_p, 
                 mlp_ratio=4,
                 context_length=1024,
                 use_bias=True,
                 auto=False,
                 fused=False):
        
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, 
                                   num_heads=num_heads, 
                                   attn_dropout_p=dropout_p, 
                                   context_length=context_length,
                                   use_bias=use_bias,
                                   auto=auto, 
                                   fused=fused)
        
        self.layernorm1 = nn.LayerNorm(embed_dim, bias=use_bias, auto=auto, fused=fused)
        self.feedforward = FeedForward(embed_dim, mlp_ratio, dropout_p, use_bias, auto=auto)
        self.layernorm2 = nn.LayerNorm(embed_dim, bias=use_bias, fused=fused)

    def forward(self, x):
        
        attn_out = self.attention(self.layernorm1(x))
        x = x + attn_out
        mlp_out = self.feedforward(self.layernorm2(x))
        x = x + mlp_out
     
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        
        super().__init__()

        self.config = config
 
        self.embeddings = Embeddings(vocab_size=config.vocab_size, 
                                     embed_dim=config.embed_dim, 
                                     context_length=config.max_seq_len,
                                     fused=config.use_fused_ops)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=config.embed_dim, 
                             num_heads=config.num_heads, 
                             dropout_p=config.mlp_dropout_p, 
                             mlp_ratio=config.mlp_ratio,
                             context_length=config.max_seq_len,
                             use_bias=config.use_bias, 
                             fused=config.use_fused_ops, 
                             auto=config.use_full_auto)

            for _ in range(config.num_blocks)
        ])

        self.final_layer_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, 
                                 config.vocab_size, 
                                 bias=config.use_bias,
                                 auto=config.use_full_auto)

        ### Initialize Weights ###
        self.apply(_init_weights)
        for name, param in self.named_parameters():
            if "out_proj" in name:
                mytorch.nn.init.normal_(param, mean=0.0, std=(0.02/np.sqrt(2 * config.num_blocks)))

        ### Weight tying ###
        self.lm_head.weight = self.embeddings.char_embeddings.weight

    def forward(self, x):

        x = self.embeddings(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_layer_norm(x)    
        x = self.lm_head(x)
        
        return x
    
### Standard Weight Init for Transformers ###
def _init_weights(module):
    if isinstance(module, nn.Linear):
        mytorch.nn.init.normal_(module.weight, mean=0, std=0.02)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        mytorch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, nn.LayerNorm):
        mytorch.nn.init.ones_(module.weight)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)