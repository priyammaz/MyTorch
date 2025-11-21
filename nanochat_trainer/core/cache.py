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
    
    def repeat(self, num_repeats):
        """
        Given that we already have some stuff in our cache, we may want to repeat it
        so we can do batch generation. For example, for a single input, we may want to 
        generate 5 different outputs. But all of these generations will share the same
        initial KV Cache, so lets just expand it out

        CAVEAT: Only will support for batch_size=1 for now. We will have batch generation
        but not batch inputs (as its more annoying to deal with different seq_len inputs)
        """

        if self.k_cache is None or self.v_cache is None:
            raise Exception("Cannot repeat if cache is empty!!")
        assert self.k_cache.shape[1] == 1, "Repeat only supported for single sample inference"

        self.k_cache = mytorch.concatenate([self.k_cache for _ in range(num_repeats)], dim=1)
        self.v_cache = mytorch.concatenate([self.v_cache for _ in range(num_repeats)], dim=1)
        self.cache_shape = self.k_cache.shape

    def update(self, 
               key_states, 
               value_states, 
               layer_idx):

        ### Init KV Cache if it doesnt exist ###
        if self.k_cache is None and self.v_cache is None:
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
            self._pos = end
        
        return self.k_cache[layer_idx, :, :, :end, :], self.v_cache[layer_idx, :, :, :end, :]
    