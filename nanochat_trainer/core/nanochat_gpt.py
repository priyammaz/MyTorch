"""
This model will be close the to the nanoChat model you can find here!
https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py

From the Docstring:

- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference

Also, this will ONLY Support Fused ops. The model is too big to train/inference without it
This means you will need the Triton install!
"""

import mytorch
import mytorch.nn as nn
import mytorch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 2**16
    sequence_length: int = 2048
    embed_dim: int = 1280
    mlp_ratio: int = 4
    num_blocks: int = 20
    num_q_heads: int = 10
    num_kv_heads: int = 10 # can be reduced for group query attention
    dropout_p: float = 0.0
    rope_base: float = 10000
    softcap: int = 15

def norm(x, training=True):
    """
    parameterless norm
    """
    return F.rmsnorm(x, weight=None, training=training, fused=True)

class Attention(nn.Module):

    def __init__(self, 
                 config,
                 layer_idx):
        
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        ### Attention Head Dim ###
        self.embed_dim = config.embed_dim
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.embed_dim // config.num_q_heads

        assert (self.num_kv_heads <= self.num_q_heads) and (self.num_q_heads % self.num_kv_heads == 0), "Q Heads must be divisible by KV Heads"
        assert self.head_dim == (self.embed_dim//self.num_kv_heads), "Head dim must be the same for Q and KV"
        
        self.gqa_enable = True if self.num_kv_heads != self.num_q_heads else False 
        self.q_proj = nn.Linear(self.embed_dim, self.num_q_heads * self.head_dim, bias=False, fused=True)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False, fused=True)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=False, fused=True)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, fused=True)
        self.proj_drop = nn.Dropout(dropout_p=config.dropout_p)

    def forward(self, x, cos_sin, cache=None):
  
        batch, seq_len, embed_dim = x.shape

        ### Compute QKV ###
        q = self.q_proj(x).reshape(batch, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        ### Apply Rotary Embeddings ### 
        cos, sin = cos_sin
        q, k = F.apply_rotary_pos_embed(q, k, cos, sin, unsqueeze_dim=2, auto=False, fused=True)
        
        ### Norm Q,K ###
        q, k = norm(q), norm(k)

        ### Make batch x num_heads x seq_len x embed_dim ###
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)
        

        ### We only need causal attention at the beginning, but after we are just passing in a single query ###
        ### at a time, so causal no longer matters! So we check, if we pass in a cache AND the cache is not ###
        ### empty then we know we are decoding with a single query token. If the cache is empty (but we do  ###
        ### pass it in) then that is our first step that will fill the cache, use causal ###
        is_causal = True
        if (cache is not None) and (cache.pos != 0):
            is_causal = False
            
        output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=self.gqa_enable)
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

        ## If using fused ops we can fuse the relu_squared activation right into forward pass ###
        self.intermediate_dense = nn.Linear(config.embed_dim, hidden_size, bias=False, fused=True, act_func="relu_squared")
        self.intermediate_dropout = nn.Dropout(config.dropout_p)
        self.out_proj = nn.Linear(hidden_size, config.embed_dim, bias=False, fused=True)
        self.output_dropout = nn.Dropout(config.dropout_p)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = self.intermediate_dropout(x)
        x = self.out_proj(x)
        x = self.output_dropout(x)
        return x
    
class TransformerBlock(nn.Module): 
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attention = Attention(config, layer_idx)
        self.mlp = FeedForward(config)
    
    def forward(self, x, cos_sin, cache):
        x = x + self.attention(norm(x), cos_sin, cache)
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module): 
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx) for layer_idx in range(config.num_blocks)
            ]
        )
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

        cos, sin = F.precompute_rotary_cos_sin(
            head_dim=config.embed_dim//config.num_q_heads, 
            max_position_embeddings=config.sequence_length, 
            base=config.rope_base
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, input_ids, target_ids=None, cache=None):
        
        batch_size, seq_len = input_ids.shape

        ### Get starting point (incase we have previous cache) ###
        start_idx = 0 if cache is None else cache.pos
        assert start_idx + seq_len <= self.cos.shape[1], "Sequence length grew past your rotary embedding cache"

        ### Get corresponding rotary embeds ###
        cos_sin = self.cos[:, start_idx:start_idx+seq_len], self.sin[:, start_idx:start_idx+seq_len]

        ### Compute Embeddings ###
        x = self.embeddings(input_ids)

        ### prenorm ###
        x = norm(x)

        ### Pass through blocks ###
        for block in self.blocks:
            x = block(x, cos_sin, cache)
        
        ### Post Norm ###
        x = norm(x)

        ### Forward through LM Head ###
        logits = self.lm_head(x)

        ### If no targets are provided we will just apply our softcap to the logits and return ###
        ### Also I assume if no targets we are in inference mode, return the cache back as well! ###
        if target_ids is None:
            logits = self.config.softcap * mytorch.nn.functional.tanh(logits / self.config.softcap, fused=True)
            to_return = (logits, )
            if cache is not None:
                to_return += (cache, )
            return to_return
        
        ### If targets are provided we can compute a loss! ###    
        ### Fused CE has softcap built right in so we can use that ###
        ### again I assume you are using fused ops here because why would you want to train a ###
        ### massive model without it? ###

        ### I assume we are training here, and we dont need the cache in training (it is unused) so we dont pass it in ###
        else:
            loss = mytorch.nn.functional.cross_entropy(logits, target_ids, softcap=self.config.softcap, fused=True)
            return logits, loss
        
if __name__ == "__main__":

    model = GPT(config=GPTConfig()).to("cuda")
    rand = mytorch.randint(0,5000, shape=(2,2048)).to("cuda")
    out = model(rand)
    print(out.shape)

    total = 0
    import numpy as np
    for name, param in model.named_parameters():
        total += np.prod(param.shape)
    print(total)
