"""
This will be basically a whatever transformer decoder that takes bits
and bobs from here and there. Its not optimal but its ours! Main 
ideas will come from nanoChat and Llama!

Defaults make this a 560,988,160 Parameter Model! Not too bad given we arent using PyTorch!

Thought process:
- Rotary embeds (why not its defacto these days)
- Parameterless norm seems to work in nanoChat so we do the same here!
- Relu squared activation (who would have though that would work lol https://arxiv.org/abs/2402.03804) 
- Softcapping is nice addition (and i already wrote the kernel might as well use it)
- Bias disabled everywhere, who wants biases anyway??

Also, this will ONLY Support Fused ops. The model is too big to train/inference without it
This means you will need the Triton install!
"""
import math
import mytorch
import mytorch.nn as nn
import mytorch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int =  2**16
    max_seq_len: int = 2048
    embed_dim: int = 1280
    mlp_ratio: int = 4
    num_blocks: int = 20
    num_q_heads: int = 10
    num_kv_heads: int = 5
    dropout_p: float = 0.0
    use_fused_ops: bool = True
    use_bias: bool = False
    rope_base: float = 10000
    softcap: float = 15.0

def norm(x, training=True):
    """
    parameterless rmsnorm
    """
    return F.rmsnorm(x, weight=None, training=training, fused=True)

class Embeddings(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        ### Embeddings for Tokens ###
        self.embeddings = nn.Embedding(config.vocab_size, 
                                       config.embed_dim, 
                                       fused=config.use_fused_ops)

    def forward(self, input_ids):

        ### Convert Tokens to Embeddings ###
        x = self.embeddings(input_ids)

        return x

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

        ### Attention Head Dim ###
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_q_heads
        self.head_dim = config.embed_dim // config.num_q_heads
        self.fused = config.use_fused_ops
        self.num_groups = self.num_q_heads // self.num_kv_heads

        ### Attention Projections ###
        self.q_proj = nn.Linear(self.embed_dim, self.num_q_heads * self.head_dim, bias=config.use_bias, fused=self.fused)
        self.k_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.use_bias, fused=self.fused)
        self.v_proj = nn.Linear(self.embed_dim, self.num_kv_heads * self.head_dim, bias=config.use_bias, fused=self.fused)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias, fused=self.fused)
        
        self.proj_drop = nn.Dropout(dropout_p=config.dropout_p)

        if not self.fused:
            self.attn_drop = nn.Dropout(dropout_p=config.dropout_p) # Currently only support attn dropout in non-fused
            self.softmax = nn.Softmax() # We technically have fused softmax, but flash attn is better

            ### If we are not using fused attention we need to manually pass in our 
            ### attention mask! So lets just save it as a buffer right here!
            causal_positions = (mytorch.tril(mytorch.ones((1,1,config.max_seq_len,config.max_seq_len))) == 0)
            causal_mask = mytorch.masked_fill(mytorch.zeros((1,1,config.max_seq_len, config.max_seq_len)), causal_positions, value=float("-inf"))  
            self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x, cos_sin, cache):
  
        batch, seq_len, embed_dim = x.shape

        ### QKV projection
        q = self.q_proj(x).reshape(batch, seq_len, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        
        ### Rotary embeds
        cos, sin = cos_sin
        q, k = F.apply_rotary_pos_embed(q, k, cos, sin, unsqueeze_dim=2, fused=self.fused)

        ### Prenorm q,k
        q, k = norm(q), norm(k)

        ### Make batch x num_heads x seq_len x embed_dim ###
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        if cache is not None:
            k, v = cache.update(k, v, self.layer_idx)
   
        ### This branch ends up being about half as fast as fused (flash) attention. The main bottle neck is the 
        ### softmax operation over long sequences. Naive softmax is pretty expensive! You can test this by
        ### changing the softmax above to use fused softmax BUT you may as well just use flash attention if 
        ### fused is available!
       
        if not self.fused:
            
            # Repeat for GQA
            if self.num_groups > 1:
                k = k.repeat_interleave(self.num_groups, dim=1)
                v = v.repeat_interleave(self.num_groups, dim=1)

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
            if (cache is not None) and (cache.pos != 0):    
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
            output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, enable_gqa=(self.num_groups > 1))
            
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
                                                fused=self.fused, 
                                                act_func="relu_squared")
        
        else:
            ### Otherwise we do them sequentially like normal! (This will be slower/use more memory!)
            ### technically we can use the fused linear/fused gelu here too, but thats ok, we prefer
            ### to fuse the activation straight in!
            self.intermediate_dense = nn.Linear(config.embed_dim, hidden_size, 
                                                bias=config.use_bias)
            self.activation = nn.ReLUSquared()

        self.intermediate_dropout = nn.Dropout(config.dropout_p)

        self.out_proj = nn.Linear(hidden_size, config.embed_dim, 
                                  bias=config.use_bias, 
                                  fused=self.fused)
        
        self.output_dropout = nn.Dropout(config.dropout_p)

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
                 config,
                 layer_idx):
        
        super().__init__()      

        self.config = config
        self.embed_dim = config.embed_dim
        self.attention = Attention(config, layer_idx)        
        self.feedforward = FeedForward(config)

    def forward(self, x, cos_sin, cache=None):
        x = x + self.attention(norm(x), cos_sin, cache)
        x = x + self.feedforward(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        
        super().__init__()

        self.config = config
        self._gradient_checkpointing_enabled = False

        self.embeddings = Embeddings(config)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx) for layer_idx in range(config.num_blocks)
        ])

        self.lm_head = nn.Linear(config.embed_dim, 
                                 config.vocab_size, 
                                 bias=config.use_bias,
                                 fused=config.use_fused_ops)

        ### Initialize Weights ###
        self.apply(_init_weights)
        for name, param in self.named_parameters():
            # This one is new to me, normally we reduce the mangitude of weights 
            # of the last layer of blocks so we reduce having an exploding 
            # magnitude right at the beginning of training, but nanoChat
            # just zeros it out to completely reduce this effect! So ill
            # give it a try to!!
            # in practice i see that the grad norm starts out very small 
            # and slowly rises and then falls again, rather than just 
            # starting large and falling, so nice trick here!
            if ("out_proj" in name) or ("lm_head" in name):
                mytorch.nn.init.zeros_(param)

        cos, sin = F.precompute_rotary_cos_sin(
            head_dim=config.embed_dim//config.num_q_heads, 
            max_position_embeddings=config.max_seq_len, 
            base=config.rope_base
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing_enabled = True

    def forward(self, input_ids, target_ids, cache=None):

        batch_size, seq_len = input_ids.shape

        ### Get starting point (incase we have previous cache) ###
        start_idx = 0 if cache is None else cache.pos
        assert start_idx + seq_len <= self.cos.shape[1], "Sequence length grew past your rotary embedding cache"

        ### Get our Cos/Sin for our Positions ###
        cos_sin = self.cos[:, start_idx:start_idx+seq_len], self.sin[:, start_idx:start_idx+seq_len]

        ### Get our embeddings ###
        x = self.embeddings(input_ids)

        ### Transformer magic ###
        for block in self.blocks:
            if not self._gradient_checkpointing_enabled:
                x = block(x, cos_sin, cache)
            else:
                ### Checkpointing is for training only no cache ###
                x = mytorch.utils.checkpoint(block, x, cos_sin)
        
        ### Post Norm ###
        x = norm(x)

        ### Projection Head ###
        logits = self.lm_head(x)
        
        ### If no targets are provided we will just apply our softcap to the logits and return ###
        ### Also I assume if no targets we are in inference mode, return the cache back as well! ###
        if target_ids is None:
            logits = self.config.softcap * F.tanh(logits / self.config.softcap, fused=self.config.use_fused_ops)
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
            loss = F.cross_entropy(logits, 
                                   target_ids, 
                                   softcap=self.config.softcap,
                                   fused=self.config.use_fused_ops)
            return logits, loss
    
### Standard Weight Init for Transformers ###
def _init_weights(module):
    if isinstance(module, nn.Linear):
        # https://arxiv.org/pdf/2310.17813
        fan_out, fan_in = module.weight.shape
        std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
        mytorch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)
            
    elif isinstance(module, nn.Embedding):
        mytorch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        if module.weight is not None:
            mytorch.nn.init.ones_(module.weight)
        if module.bias is not None:
            mytorch.nn.init.zeros_(module.bias)