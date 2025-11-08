"""
A nothing fancy GPT2 with a KV Cache integrated in for Inference Time!

The gpt2-base model has only a sequence length of 1024 so you cant really appreciate
the KV cache, but its there anyway! 
"""
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

class Cache:

    """
    Barebones KV Cache that stores Keys/Values in the [B x Heads x Seq Len x Head Dim] shape!
    """
    def __init__(self, config):

        ### Key/Value Cache (List of Tensor, where list is over model layers) ###
        ### We use these empty tensors as a placeholder ###
        self.key_cache = [mytorch.Tensor([]) for _ in range(config.num_blocks)]
        self.value_cache = [mytorch.Tensor([]) for _ in range(config.num_blocks)]
        
        ### What is our max context size? ###
        self.max_seq_len = config.max_seq_len 

    def __repr__(self):        

        ### If Cache is Empty ###
        if self.key_cache[0].shape == (0,):
            cached_tokens = 0
        else:
            cached_tokens = self.key_cache[0].shape[-2]

        return f"DyanmicCache(Num_Layers: {len(self.key_cache)} | Cached Tokens: {cached_tokens})"
        
    def update(self, key_states, value_states, layer_idx):
        """
        Basically a deque. We start filling in tokens, but once we fill our context length
        we will start dropping old tokens to make room for the new ones!
        """
        ### Append New key/Value states to key/value cache ###
        ### If we are starting then there is nothing so just fill it in ###
        if self.get_seq_len(layer_idx) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        
        ### Otherwise we can concatenate the new key/value onto the old ones ###
        else:
            self.key_cache[layer_idx] = mytorch.concatenate([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = mytorch.concatenate([self.value_cache[layer_idx], value_states], dim=-2)

        current_len = self.key_cache[layer_idx].shape[-2]
        if current_len > self.max_seq_len:
            num_to_remove = current_len - self.max_seq_len
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, num_to_remove:, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, num_to_remove:, :]
   
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_len(self, layer_idx=0):
        
        ### If Cache is Empty ###
        if self.key_cache[layer_idx].shape == (0,):
            cached_tokens = 0
        else:
            cached_tokens = self.key_cache[layer_idx].shape[-2]

        return cached_tokens

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

    def forward(self, 
                input_ids, 
                past_length=0):

        batch_size, seq_length = input_ids.shape

        ### Convert Tokens to Embeddings ###
        x = self.char_embeddings(input_ids)
      
        ### Add Positional Information ###
        avail_idx = mytorch.arange(start=0, end=seq_length).to(input_ids.device) + past_length
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
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias, auto=auto, fused=self.fused)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias, auto=auto, fused=self.fused)
        self.proj_drop = nn.Dropout(dropout_p=attn_dropout_p)

        if not self.fused:
            self.attn_drop = nn.Dropout(dropout_p=attn_dropout_p) # Currently only support attn dropout in non-fused
            self.softmax = nn.Softmax(auto=auto, fused=self.fused)

            ### If we are not using fused attention we need to manually pass in our 
            ### attention mask! So lets just save it as a buffer right here!
            causal_positions = (mytorch.tril(mytorch.ones((1,1,context_length,context_length))) == 0)
            causal_mask = mytorch.masked_fill(mytorch.zeros((1,1,context_length, context_length)), causal_positions, value=float("-inf"))  
            self.register_buffer("causal_mask", causal_mask)

    def forward(self, x, cache=None, layer_idx=None):
  
        batch, seq_len, embed_dim = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*embed_dim]

        # Reshape to multi-head
        qkv = qkv.reshape(batch, seq_len, self.num_heads, 3 * self.head_dim)

        # Transpose to [batch, num_heads, seq_len, 3*head_dim]
        qkv = qkv.transpose(1, 2)

        # Chunk last dim into q, k, v
        q, k, v = mytorch.chunk(qkv, 3, dim=-1)  # each [batch, num_heads, seq_len, head_dim]
        
        if cache is not None and layer_idx is not None:
            k, v = cache.update(k, v, layer_idx)
   
        ### This branch ends up being about half as fast as fused (flash) attention. The main bottle neck is the 
        ### softmax operation over long sequences. Naive softmax is pretty expensive! You can test this by
        ### changing the softmax above to use fused softmax BUT you may as well just use flash attention if 
        ### fused is available!
        if not self.fused:
    
            # Compute attention scores
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

            # If our queries are a single input and we are doing attention with our k/v cache 
            # which is by definition before the queries, then we dont really need a causal mask!
            # thus we only need our masking when the cache is empty or if there is no cache at all
            if ((cache is not None) and (cache.get_seq_len == 0)) or (cache is None):
                kv_seq_len = k.shape[-2]
                scores = scores + self.causal_mask[:, :, :seq_len, :kv_seq_len].astype(scores.data.dtype)

            softmax_attention = self.softmax(scores, dim=-1)
            dropped_attention = self.attn_drop(softmax_attention)

            # Attention output
            output = dropped_attention @ v

        else:

            ### We only need causal attention at the beginning, but after we are just passing in a single query ###
            ### at a time, so causal no longer matters! So we check, if we pass in a cache AND the cache is not ###
            ### empty then we know we are decoding with a single query token. If the cache is empty (but we do  ###
            ### pass it in) then that is our first step that will fill the cache, use causal ###
            is_causal = True
            if (cache is not None) and (cache.get_seq_len() != 0):        
                is_causal = False
            
            ### CAVEAT. Our flash attention isnt as flexible as the actual torch.nn.functional.sdpa. The thing is that 
            ### if our queries are a different length from the keys and values (which it will be if using cache) then
            ### then two thigns happen:

            ### 1) This will actually trigger our cross attention branch (even though we arent doing cross attention its in
            ###    same spirit of things). But remember that  our cross attention doesnt support causal 
            ###    masking. This is OK though as when we are doing KV Cache, our query need to attend to ALL the keys/values
            ###    as we are at the end of the sequence attending backwards, so no causality mask is needed

            ### 2) In the very first forward pass with our input, we may have multiple tokens. This needs to be processed like 
            ###    normal with a causal mask. Thus we had the check earlier, if our cache is empty use causal mask. The Q,K,V
            ###    in the first forward pass will all have the same sequence length so this will trigger normal self attention
            output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            
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
                 auto=False,
                 fused=False):
        super().__init__()
        hidden_size = embed_dim * mlp_ratio

        self.intermediate_dense = nn.Linear(embed_dim, hidden_size, bias=use_bias, auto=auto, fused=fused)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(mlp_dropout_p)

        self.out_proj = nn.Linear(hidden_size, embed_dim, bias=use_bias, auto=auto, fused=fused)
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
        self.feedforward = FeedForward(embed_dim, mlp_ratio, dropout_p, use_bias, auto=auto, fused=fused)
        self.layernorm2 = nn.LayerNorm(embed_dim, bias=use_bias, fused=fused)

    def forward(self, x, cache=None, layer_idx=None):
        attn_out = self.attention(self.layernorm1(x), cache=cache, layer_idx=layer_idx)
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

        self.final_layer_norm = nn.LayerNorm(config.embed_dim, 
                                             auto=config.use_full_auto, 
                                             fused=config.use_fused_ops)
        self.lm_head = nn.Linear(config.embed_dim, 
                                 config.vocab_size, 
                                 bias=config.use_bias,
                                 auto=config.use_full_auto, 
                                 fused=config.use_fused_ops)

        ### Initialize Weights ###
        self.apply(_init_weights)
        for name, param in self.named_parameters():
            if "out_proj" in name:
                mytorch.nn.init.normal_(param, mean=0.0, std=(0.02/np.sqrt(2 * config.num_blocks)))

        ### Weight tying ###
        self.lm_head.weight = self.embeddings.char_embeddings.weight

    def forward(self, x, cache=None):

        ### How many tokens have we processed already? 
        past_length = cache.get_seq_len() if cache is not None else 0
      
        ### Get our embeddings ###
        x = self.embeddings(x, past_length)

        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cache=cache, layer_idx=layer_idx)
            

        x = self.final_layer_norm(x)    
        x = self.lm_head(x)
        
        ### Return the KV Cache if we want to use it again (dont really care durning training) ###
        if cache is not None:
            return x, cache
        
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