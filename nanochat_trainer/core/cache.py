import mytorch

class KVCache:
    
    def __init__(self, 
                 batch_size, 
                 num_heads, 
                 seq_len, 
                 head_dim, 
                 num_layers):
        
        self.cache_shape = (num_layers, batch_size, num_heads, seq_len, head_dim)
        self.k_cache = None
        self.v_cache = None
        self._pos = 0

    @property
    def pos(self):
        return self._pos
    
    def reset(self):
        self._pos = 0
    
    def update(self, 
               key_states, 
               value_states, 
               layer_idx):
        
        ### Init KV Cache if it doesnt exist ###
        if self.k_cache and self.v_cache is None:
            self.k_cache = mytorch.empty(*self.cache_shape, dtype=key_states.dtype, device=key_states.device)
            self.v_cache = mytorch.empty(*self.cache_shape, dtype=value_states.dtype, device=value_states.device)

        ### Get the shape of the data we are passing in ###
        batch_size, num_heads, seq_len, embed_dim = key_states.shape
        
        ### Get start/end position in cache to store these keys/values ###
        start, end = self.pos, self.pos + seq_len

        ### Grow Cache if We fill it up ###
        assert end <= self.cache_shape[-2], "You have filled up your Cache!"

        ### Store in Cache ###
        self.k_cache[layer_idx, :, :, start:end, :] = key_states
        self.v_cache[layer_idx, :, :, start:end, :] = value_states

        ### After the last layer of the model is cached we increment _pos ###
        if layer_idx == (self.cache_shape[0] - 1):
            self.pos = end
        
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    