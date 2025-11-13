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
    use_layernorm_weight: bool = True
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

    def __init__(self, config):
        super().__init__()

        self.config = config

        ### Embeddings for Tokens ###
        self.char_embeddings = nn.Embedding(config.vocab_size, 
                                            config.embed_dim, 
                                            fused=config.use_fused_ops)

        ### Positional Embeddings ###
        self.position_embeddings = nn.Embedding(config.max_seq_len, 
                                                config.embed_dim, 
                                                fused=config.use_fused_ops)

    def forward(self, 
                input_ids, 
                past_length=0):

        batch_size, seq_length = input_ids.shape

        ### Convert Tokens to Embeddings ###
        x = self.char_embeddings(input_ids)
      
        ### Add Positional Information ###
        avail_idx = mytorch.arange(start=0, end=seq_length).to(input_ids.device) + past_length
        pos_embed = self.position_embeddings(avail_idx).reshape(1, seq_length, self.config.embed_dim)
        x = x + pos_embed

        return x

class Attention(nn.Module):

    def __init__(self, 
                 config):
        
        super().__init__()

        self.config = config

        ### Sanity Checks ###
        assert config.embed_dim % config.num_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.fused = config.use_fused_ops

        ### Attention Projections ###
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, 
                                  bias=config.use_bias, 
                                  auto=config.use_full_auto, 
                                  fused=self.fused)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, 
                                  bias=config.use_bias, 
                                  auto=config.use_full_auto,
                                  fused=self.fused)
        
        self.proj_drop = nn.Dropout(dropout_p=config.attn_dropout_p)

        if not self.fused:
            self.attn_drop = nn.Dropout(dropout_p=config.attn_dropout_p) # Currently only support attn dropout in non-fused
            self.softmax = nn.Softmax(auto=config.use_full_auto) # We technically have fused softmax, but flash attn is better

            ### If we are not using fused attention we need to manually pass in our 
            ### attention mask! So lets just save it as a buffer right here!
            causal_positions = (mytorch.tril(mytorch.ones((1,1,config.max_seq_len,config.max_seq_len))) == 0)
            causal_mask = mytorch.masked_fill(mytorch.zeros((1,1,config.max_seq_len, config.max_seq_len)), causal_positions, value=float("-inf"))  
            self.register_buffer("causal_mask", causal_mask, persistent=False)

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
                 config):
        
        super().__init__()

        self.config = config

        hidden_size = config.embed_dim * config.mlp_ratio
        self.fused = config.use_fused_ops 

        if self.fused:
            ## If using fused ops we can fuse the GELU activation right into forward pass ###
            self.intermediate_dense = nn.Linear(config.embed_dim, hidden_size, 
                                                bias=config.use_bias,
                                                auto=config.use_full_auto,
                                                fused=self.fused, 
                                                act_func="gelu")
        
        else:
            ### Otherwise we do them sequentially like normal! (This will be slower/use more memory!)
            ### technically we can use the fused linear/fused gelu here too, but thats ok, we prefer
            ### to fuse the activation straight in!
            self.intermediate_dense = nn.Linear(config.embed_dim, hidden_size, 
                                                bias=config.use_bias,
                                                auto=config.use_full_auto)
            self.activation = nn.GELU()

        self.intermediate_dropout = nn.Dropout(config.mlp_dropout_p)

        self.out_proj = nn.Linear(hidden_size, config.embed_dim, 
                                  bias=config.use_bias, 
                                  auto=config.use_full_auto,
                                  fused=self.fused)
        self.output_dropout = nn.Dropout(config.mlp_dropout_p)

    def forward(self, x):
        x = self.intermediate_dense(x)
        if not self.fused:
            x = self.activation(x)
        x = self.intermediate_dropout(x)
        x = self.out_proj(x)
        x = self.output_dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, 
                 config):
        
        super().__init__()      

        self.config = config
        self.embed_dim = config.embed_dim
        self.attention = Attention(config)
        
        self.layernorm1 = nn.LayerNorm(config.embed_dim, 
                                       weight=config.use_layernorm_weight,
                                       bias=config.use_bias,
                                       auto=config.use_full_auto,
                                       fused=config.use_fused_ops)
        
        self.feedforward = FeedForward(config)
        self.layernorm2 = nn.LayerNorm(config.embed_dim, 
                                       weight=config.use_layernorm_weight,
                                       bias=config.use_bias,
                                       auto=config.use_full_auto,
                                       fused=config.use_fused_ops)

    def forward(self, x, cache=None, layer_idx=None):
        x = x + self.attention(self.layernorm1(x), cache=cache, layer_idx=layer_idx)
        x = x + self.feedforward(self.layernorm2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        
        super().__init__()

        self.config = config
        self.embeddings = Embeddings(config)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config)

            for _ in range(config.num_blocks)
        ])

        self.final_layer_norm = nn.LayerNorm(config.embed_dim,
                                             weight=config.use_layernorm_weight,
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
        if module.weight is not None:
            mytorch.nn.init.ones_(module.weight)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)