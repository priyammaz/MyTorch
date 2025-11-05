"""
This is nearly identical to fused_ops/flash_attention.py except
we only support cross attention here! Cross attention implies the length of our queries may not match
the length of our Keys and Values.

We could have put this all together in the single flash_attention.py, but
it makes that code more challenging and this is tough enough. So to maintain
readability (at the cost of extra lines) we separate them out!

WHAT CHANGED: 

### Change 1 ###
When doing Self-Attention, the SEQ_LEN is the for Q and K/V. So we had 
to only deal with that single SEQ_LEN parameter. Now we two separate lengths:

SEQ_LEN_Q, SEQ_LEN_KV

### Change 2 ###
Removed all Causal related flags. Cross Attention is not causal, so we remove
that flag!

### Change 3 ###
In the original flash_attention.py we had a kernel called `_attn_bwd` inside which we 
called `_attn_bwd_dq` and `_attn_bwd_dk_dv`. This meant each thread was responsible for 
a block of dQ as well as a block of dK,dV! This only worked out so easily though because
our attention matrix was just a square. 

Now that we have different sequence lengths, we cannot divide our attention matrix into the
same number of blocks in each direction. This means we will either have to have extra threads
that dont do anything, or we wont have enough to cover one of the dimensions. Its much simpler
here to just not do that and have to separate kernel calls, one that computes dQ and another 
that computes dK,dV! So you will find in our `fused_cross_sdpa_backward` that we do exactly that!
"""

import os
import torch
import cupy as cp
import triton
import triton.language as tl
# from .flags import DLPACK_DISABLE, AUTOTUNE_MODE
DLPACK_DISABLE = False
AUTOTUNE_MODE="none"

def get_fwd_autotune_configs():
    # Read the autotune mode from environment variable, default to "none"
    mode = AUTOTUNE_MODE

    if mode == "none":
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_KV": 32},
                num_stages=2,
                num_warps=4,
            )
        ]

    # Minimal configs for "slow" mode (fewer configurations for faster tuning)
    if mode == "medium":
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_Q in [32, 64]  # Reduced set
            for BLOCK_SIZE_KV in [16, 32]  # Reduced set
            for num_stages in [2, 3]      # Reduced set
            for num_warps in [4, 8]      # Reduced set
            if BLOCK_SIZE_KV < BLOCK_SIZE_Q
        ]
    # Comprehensive configs for "max" mode (full search space)
    else:  # mode == "max" or any other value defaults to max
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_Q in [16, 32, 64, 128]
            for BLOCK_SIZE_KV in [16, 32, 64]
            for num_stages in [2, 3, 4]
            for num_warps in [4, 8, 16]
            if BLOCK_SIZE_KV < BLOCK_SIZE_Q
        ]

def get_preprocess_autotune_configs():
    # Read the autotune mode from environment variable, default to "none"
    mode = AUTOTUNE_MODE
    
    # Single config for "none" mode (no autotuning, fixed configuration)
    if mode == "none":
        return [
            triton.Config(
                {"BLOCK_SIZE": 64},
                num_stages=2,
                num_warps=4,
            )
        ]
    
    # Reduced configs for "medium" mode (fewer configurations for faster tuning)
    if mode == "medium":
        return [
            triton.Config(
                {"BLOCK_SIZE": BLOCK_SIZE},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE in [32, 64]  # Reduced set
            for num_stages in [2, 3]    # Reduced set
            for num_warps in [4, 8]     # Reduced set
        ]
    
    # Comprehensive configs for "max" mode (full search space)
    else:  # mode == "max" or any other value defaults to max
        return [
            triton.Config(
                {"BLOCK_SIZE": BLOCK_SIZE},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE in [16, 32, 64, 128]
            for num_stages in [2, 3, 4]
            for num_warps in [4, 8, 16]
        ]

def get_bwd_dq_autotune_configs():
    # Read the autotune mode from environment variable, default to "none"
    mode = AUTOTUNE_MODE
    
    # Single config for "none" mode (no autotuning, fixed configuration)
    if mode == "none":
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": 32, "BLOCK_SIZE_KV": 16},
                num_stages=2,
                num_warps=4,
            )
        ]
    
    # Reduced configs for "medium" mode (fewer configurations for faster tuning)
    if mode == "medium":
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_Q in [32, 64]  # Reduced set
            for BLOCK_SIZE_KV in [16, 32]  # Reduced set
            for num_stages in [2, 3]         # Reduced set
            for num_warps in [4, 8]          # Reduced set
            if BLOCK_SIZE_KV < BLOCK_SIZE_Q
        ]
    
    # Comprehensive configs for "max" mode (full search space)
    else:  # mode == "max" or any other value defaults to max
        return [
            triton.Config(
                {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_Q in [16, 32, 64, 128]
            for BLOCK_SIZE_KV in [16, 32, 64]
            for num_stages in [2, 3, 4]
            for num_warps in [4, 8, 16]
            if BLOCK_SIZE_KV < BLOCK_SIZE_Q
        ]

def get_bwd_dkdv_autotune_configs():
    # Read the autotune mode from environment variable, default to "none"
    mode = AUTOTUNE_MODE
    
    # Single config for "none" mode (no autotuning, fixed configuration)
    if mode == "none":
        return [
            triton.Config(
                {"BLOCK_SIZE_KV": 32, "BLOCK_SIZE_Q": 16},
                num_stages=2,
                num_warps=4,
            )
        ]
    
    # Reduced configs for "medium" mode (fewer configurations for faster tuning)
    if mode == "medium":
        return [
            triton.Config(
                {"BLOCK_SIZE_KV": BLOCK_SIZE_KV, "BLOCK_SIZE_Q": BLOCK_SIZE_Q},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_KV in [32, 64]  # Reduced set
            for BLOCK_SIZE_Q in [16, 32]  # Reduced set
            for num_stages in [2, 3]         # Reduced set
            for num_warps in [4, 8]          # Reduced set
            if BLOCK_SIZE_Q < BLOCK_SIZE_KV
        ]
    
    # Comprehensive configs for "max" mode (full search space)
    else:  # mode == "max" or any other value defaults to max
        return [
            triton.Config(
                {"BLOCK_SIZE_KV": BLOCK_SIZE_KV, "BLOCK_SIZE_Q": BLOCK_SIZE_Q},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_KV in [16, 32, 64, 128]
            for BLOCK_SIZE_Q in [16, 32, 64]
            for num_stages in [2, 3, 4]
            for num_warps in [4, 8, 16]
            if BLOCK_SIZE_Q < BLOCK_SIZE_KV
        ]

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    attn_mask_ptr,
    BLOCK_SIZE_KV,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN_Q,
    SEQ_LEN_KV,
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16
    USE_CUSTOM_MASK: tl.constexpr, 
    stride_mask_batch,  
    stride_mask_head, 
    stride_mask_q,
    stride_mask_kv,
    index_batch, 
    index_head,
):      
    """
    The inner loop of the forward flash attention method grabs a chunk of queries
    and loops through all the Keys/Values also in chunks, using online-softmax as we go
    """

    lo, hi = 0, SEQ_LEN_KV

    ### KV pointers are currently pointing at the very start of the Key/Value for this ###
    ### Specific batch and head. In the case of STAGE=1 or ELSE, we just start at 0. We will ###
    ### piece by piece load BLOCK_SIZE_KV sizes of our Keys nad Values and do our ops there ###
    ### but in STAGE=2, we only want to do the ops on the diagonal values, so we need to advance ###
    ### our index to there ###

    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # Keys are transposed so SEQ dim is second
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    ### Loop over our Ks and Vs ###
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):

        ### Let the compiler know that start_n is a multiple of BLOCK_N ###
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        ### Compute a mask so we dont access token indexes longer than our sequence 
        kv_indices = start_kv + offs_kv
        kv_padding_mask = kv_indices < SEQ_LEN_KV

        ### Load our K and V Blocks ###
        K_block = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        ### We have a (Q_BLOCK_SIZE x E) and (E x KV_BLOCK_SIZE) matricies ###
        ### we can just use dot to do our dot product computation ### 
        QK_block = tl.dot(Q_block, K_block)

        ### If we are not pass_type==1, then we are either processing pre-diagonal blocks ###
        ### or we are just processing all blocks. In either case, we want to make sure that ###
        ### we mask our any invalid positions in our QK Block and dont have to worry about inside block transitions ###
        QK_block += tl.where(kv_padding_mask[None, :], 0, float("-inf"))

        ### If we are using attention mask ###
        if USE_CUSTOM_MASK:
            
            ### we need to advance to the correct block of our attention matrix that we are ###
            ### currently processing inside our attention mask ###
            ### again the offs_q tells us the block of queries we are processing and the 
            ### kv_indices tell us which keys/values we are processing in this iter of the loop
            mask_offset = (
                index_batch * stride_mask_batch + 
                index_head * stride_mask_head +
                offs_q[:, None] * stride_mask_q + 
                kv_indices[None, :] * stride_mask_kv
            )

            ### Grab this block of our mask ###
            custom_mask = tl.load(attn_mask_ptr + mask_offset, 
                                  mask=(offs_q[:, None] < SEQ_LEN_Q) & (kv_indices[None, :] < SEQ_LEN_KV),
                                  other=False)
        
            ### Add -inf to the masked out positions, so it becomes 0 after softmax ###
            QK_block += tl.where(custom_mask, 0, float("-inf"))

        ### Update our current estimate for the maximum
        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
        QK_block -= m_ij[:, None]

        ### We subtracted the max (and masked if needed) now we exponentiate ###
        ### But remember we will use exp2 instead of exp for float16 stability ###
        ### and we already adjusted for this earlier! ###
        P_block = tl.math.exp2(QK_block)

        ### Compute the sum of the rows for this block ###
        l_ij = tl.sum(P_block, 1)

        ### Correction factor from the previous block so we can do an online softmax ###
        alpha = tl.math.exp2(m_i - m_ij)

        ### Apply the correction factor ###
        l_i = l_i * alpha + l_ij

        ### Make sure our Dtype matches our flag By default it will be float32 ###
        ### but we need to fast to fp16 incase thats the precision type we have ###
        P_block = P_block.to(tl.float32 if DTYPE_FLAG==0 else tl.float16)

        ### Use our formuala to iteratively update our outputs O_new = PV + O_old * alpha ###
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, acc=O_block)
        
        ### Update Estimate for Next Iter ###
        m_i = m_ij

        ### Advance to the next block ###
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        
    return O_block, l_i, m_i

@triton.autotune(
    configs=get_fwd_autotune_configs(),
    key=["SEQ_LEN_Q", "SEQ_LEN_KV", "HEAD_DIM"]
)
@triton.jit
def _attn_fwd(
    Q,  
    K, 
    V,  
    attn_mask,
    softmax_scale: tl.constexpr,
    M,  
    O, 
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_seq,
    stride_V_dim,
    stride_O_seq,
    stride_O_dim,
    stride_mask_batch, 
    stride_mask_head, 
    stride_mask_q, 
    stride_mask_kv,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN_Q,
    SEQ_LEN_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16
    USE_CUSTOM_MASK: tl.constexpr
):  
    """
    Main forward method for Flash Attention, where for a block of queries
    we iteratively compute attention by looping over blocks of Keys/Values
    """

    ### When we do Q @ K, we use the tl.dot method to do this ###
    ### So the inner product loads a row/column of K and Q into ###
    ### registers for the actual computation where each row has HEAD_DIM elements ###
    ### So although we are chunking our sequence into BLOCK_SIZE_KV, we still need ###
    ### to load the entire embeddings. We want to make sure this isnt too large for ###
    ### efficiency. So we place a restriction here that our BLOCK_SIZE cannot be any ###
    ### larger than our HEAD_DIM. Id rather have more blocks scheduled to do less work ###
    ### than have fewer blocks each processing massive matricies for better GPU utilization ###
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    INV_LN2: tl.constexpr = 1.442695040888 # approx 1 / ln(2)

    ### We have to multiply by our 1/sqrt(head_dim) too, so just add the additional multipler to here for later! 
    softmax_scale *= INV_LN2

    #### This is the block index of Q that we will process ###
    block_index_q = tl.program_id(0)

    ### Cast our Pointers to the right type ### 
    Q = tl.cast(Q, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    K = tl.cast(K, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    V = tl.cast(V, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    O = tl.cast(O, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))

    ### Intermediate buffer M is always a float32 for maintaining precision in the backward pass ###
    M = tl.cast(M, tl.pointer_type(tl.float32))

    if USE_CUSTOM_MASK: 
        attn_mask = tl.cast(attn_mask, tl.pointer_type(tl.int1))
        
    ### our index batch head is just a flattened vector of our batch_size * number of heads ###
    ### this means if we want what batch we are on, we can divide by num heads ###
    ### if we want which head we are on we can use modulo ###
    index_batch_head = tl.program_id(1) 
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    ### Compute our offset of where a particular batch and head starts ###
    q_offset = (
    index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head
    )
    kv_offset = (
        index_batch.to(tl.int64) * stride_K_batch + index_head.to(tl.int64) * stride_K_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,                   
        shape=(SEQ_LEN_Q, HEAD_DIM),                
        strides=(stride_Q_seq, stride_Q_dim),     
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),     
        order=(1,0)                               
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,                      
        shape=(SEQ_LEN_KV, HEAD_DIM),                
        strides=(stride_V_seq, stride_V_dim),     
        offsets=(0,0),                           
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),              
        order=(1,0)                               
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,                      
        shape=(HEAD_DIM, SEQ_LEN_KV),               
        strides=(stride_K_dim, stride_K_seq),       
        offsets=(0,0),                           
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),              
        order=(0,1)                    
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O + q_offset,                      
        shape=(SEQ_LEN_Q, HEAD_DIM),               
        strides=(stride_O_seq, stride_O_dim),     
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),     
        order=(1,0)                               
    )

    ### Lets grab offsets to tell us which indexes of Queries we are processing ###
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    ### We also need our offsets for the kv, of how many kv vectors are we processing with 
    ### Every one of our query blocks? 
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    ### Intermediate data we store will be in a higher precision for efficiency ###
    ### Running max initialized with -inf ###
    m_i = tl.full(shape=[BLOCK_SIZE_Q], value=float("-inf"), dtype=tl.float32)

    ### Running sum for our denominator (sum e^x) ###
    ### We initialize with 1, exponentiate in a bit and so e^1 is 0, the start of our sum ###
    l_i = tl.full(shape=[BLOCK_SIZE_Q], value=1.0, dtype=tl.float32) 

    ### Accumulation of our final qk^T v for our specific block of queries/keys/values ###
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    
    ### Load our Query Block ###
    ### Now a super cool ability for block pointers. It can automagically check ###
    ### for invalid indexes (like if our Query we are indexing is greater than SEQ_LEN) ###
    ### And it will fill it with the padding option we give it! ###
    Q_block = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    
    ### Prescale our Queries here so we dont need to do it in our inner loop later ###
    Q_block *= softmax_scale
    Q_block = Q_block.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16)

    O_block, l_i, m_i = _attn_fwd_inner(
        O_block, 
        l_i, 
        m_i, 
        Q_block, 
        K_block_ptr, 
        V_block_ptr, 
        attn_mask,
        BLOCK_SIZE_KV, 
        offs_q,
        offs_kv, 
        SEQ_LEN_Q,
        SEQ_LEN_KV,
        DTYPE_FLAG,
        USE_CUSTOM_MASK, 
        stride_mask_batch, 
        stride_mask_head, 
        stride_mask_q, 
        stride_mask_kv, 
        index_batch, 
        index_head
    )

    ### Store this as we need it for logsumexp in the backward pass ###
    ### this is the main trick we use so in our backward pass we can just ###
    ### use this to quickly recompute our softmax values, without storing ###
    ### a giant N^2 Softmax matrix in memory! ###
    m_i += tl.math.log2(l_i)

    ### We also now have our true sum along each row of attention, we can divide ###
    ### by them to get our actual normalized outputs ###
    O_block = O_block / (l_i[:, None] + 1e-6)
    
    ### Store M w/ a boundary check###
    m_ptrs = M + index_batch_head * SEQ_LEN_Q + offs_q
    q_padding_mask = offs_q < SEQ_LEN_Q
    tl.store(m_ptrs, m_i, mask=q_padding_mask)

    ### Store Q (again with a boundary check) ###
    tl.store(O_block_ptr, O_block.to(O.type.element_ty), boundary_check=(0,))

@triton.autotune(
    configs=get_preprocess_autotune_configs(),
    key=["SEQ_LEN_Q", "SEQ_LEN_KV", "HEAD_DIM"],
)
@triton.jit
def attn_backward_preprocess(
    O_ptr, 
    dO_ptr,
    D_ptr,
    stride_O_heads, 
    stride_O_len, 
    stride_O_embed,
    stride_dO_heads, 
    stride_dO_len, 
    stride_dO_embed,
    stride_D_head,
    SEQ_LEN_Q,
    EMBED_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr # 0 for float32, 1 for float16
):  
    
    """
    Just a fancy way to do sum(dO * O, axis=-1)
    """
    
    row = tl.program_id(0)
    index_batch_head = tl.program_id(1)

    ### Cast Pointers to Correct type ##
    O_ptr = tl.cast(O_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dO_ptr = tl.cast(dO_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))

    ### Our intermediate D is always float32 ###
    D_ptr = tl.cast(D_ptr, tl.pointer_type(tl.float32))

    ### Mask to not grab invalid rows along our sequence length ###
    row_offsets = row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, EMBED_DIM)
    mask = row_offsets < SEQ_LEN_Q

    ### Grab our Output values ###
    O_ptr += index_batch_head * stride_O_heads
    O_offsets = row_offsets[:, None] * stride_O_len + col_offsets[None, :] * stride_O_embed
    O = tl.load(O_ptr + O_offsets, mask = mask[:, None], other=0.)
    
    ### Grab our output grads ###
    dO_ptr += index_batch_head * stride_dO_heads
    dO_offsets = row_offsets[:, None] * stride_dO_len + col_offsets[None, :] * stride_dO_embed
    dO = tl.load(dO_ptr + dO_offsets, mask = mask[:, None], other=0.)

    ### Multiply and store them ###
    Delta = tl.sum(dO.to(tl.float32) * O.to(tl.float32), axis=1) 
    D_ptr += index_batch_head * stride_D_head
    tl.store(D_ptr + row_offsets, Delta, mask = mask)

@triton.autotune(
    configs=get_bwd_dq_autotune_configs(),
    key=["SEQ_LEN_Q", "SEQ_LEN_KV", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dq(
    Q_ptr, 
    K_ptr, 
    V_ptr, 
    dO_ptr, 
    dQ_ptr, 
    M_ptr, 
    D_ptr, 
    attn_mask_ptr,
    softmax_scale: tl.constexpr,
    stride_Q_batch, 
    stride_Q_head, 
    stride_Q_len, 
    stride_Q_embed,
    stride_K_batch, 
    stride_K_head, 
    stride_K_len, 
    stride_K_embed,
    stride_V_batch, 
    stride_V_head, 
    stride_V_len, 
    stride_V_embed,
    stride_mask_batch,
    stride_mask_head,
    stride_mask_q,
    stride_mask_kv,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN_Q: tl.constexpr,
    SEQ_LEN_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr 
):
    
    ln2: tl.constexpr = 0.693147182464
    rln2: tl.constexpr = 1.442695040888
    
    # Cast pointers
    Q_ptr = tl.cast(Q_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    K_ptr = tl.cast(K_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    V_ptr = tl.cast(V_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dO_ptr = tl.cast(dO_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dQ_ptr = tl.cast(dQ_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    M_ptr = tl.cast(M_ptr, tl.pointer_type(tl.float32))
    D_ptr = tl.cast(D_ptr, tl.pointer_type(tl.float32))

    if USE_CUSTOM_MASK:
        attn_mask_ptr = tl.cast(attn_mask_ptr, tl.pointer_type(tl.int1))

    # Which block of Q are we processing?
    pid = tl.program_id(0)
    
    # Which batch/head?
    index_batch_head = tl.program_id(1)
    idx_batch = index_batch_head // NUM_HEADS
    idx_head = index_batch_head % NUM_HEADS

    offset_batch_head_4d_Q = idx_batch * stride_Q_batch + idx_head * stride_Q_head # for (B x H x L x E) Tensors
    offset_batch_head_4d_K = idx_batch * stride_K_batch + idx_head * stride_K_head # for (B x H x L x E) Tensors
    offset_batch_head_4d_V = idx_batch * stride_V_batch + idx_head * stride_V_head # for (B x H x L x E) Tensors
    offset_batch_head_3d = index_batch_head * SEQ_LEN_Q                            # for (B x H x L) Tensors

    Q_ptr += offset_batch_head_4d_Q
    K_ptr += offset_batch_head_4d_K
    V_ptr += offset_batch_head_4d_V
    dO_ptr += offset_batch_head_4d_Q
    dQ_ptr += offset_batch_head_4d_Q
    M_ptr += offset_batch_head_3d
    D_ptr += offset_batch_head_3d 

    # Compute Q block offsets
    offs_q = pid * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_embed = tl.arange(0, HEAD_DIM)
    q_mask = offs_q < SEQ_LEN_Q
    
    # Load Q block
    Q_offsets = offs_q[:, None] * stride_Q_len + offs_embed[None, :] * stride_Q_embed
    Q_block = tl.load(Q_ptr + Q_offsets, mask=q_mask[:, None], other=0.)
    Q_block *= softmax_scale * rln2
    Q_block = Q_block.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16)

    # Load dO block
    dO_block = tl.load(dO_ptr + Q_offsets, mask=q_mask[:, None], other=0.)
    
    # Load M (logsumexp) for this Q block
    M_block = tl.load(M_ptr + offs_q, mask=q_mask, other=0.)[:, None]
    
    # Load D for this Q block
    D_block = tl.load(D_ptr + offs_q, mask=q_mask, other=0.)[:, None]
    
    # Initialize dQ accumulator
    dQ_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    
    # Loop over all K/V blocks
    num_kv_blocks = tl.cdiv(SEQ_LEN_KV, BLOCK_SIZE_KV)

    for kv_block_idx in range(num_kv_blocks):
        start_kv = kv_block_idx * BLOCK_SIZE_KV
        offs_kv = start_kv + tl.arange(0, BLOCK_SIZE_KV)
        kv_mask = offs_kv < SEQ_LEN_KV
        
        # Load K^T and V^T blocks
        K_offsets = offs_embed[:, None] * stride_K_embed + offs_kv[None, :] * stride_K_len
        K_T_block = tl.load(K_ptr + K_offsets, mask=kv_mask[None, :], other=0.)

        V_offsets = offs_embed[:, None] * stride_V_embed + offs_kv[None, :] * stride_V_len
        V_T_block = tl.load(V_ptr + V_offsets, mask=kv_mask[None, :], other=0.)
        
        # Compute Q @ K^T
        S = tl.dot(Q_block, K_T_block)
        
        # Recompute softmax: P = exp2(S - M)
        P = tl.math.exp2(S - M_block)
        
        # No causal masking needed for cross-attention
        # Just apply padding mask
        P = tl.where(kv_mask[None, :], P, 0.)

        if USE_CUSTOM_MASK:

            ### Pointers to block of mask we want ###
            mask_offset = (
                idx_batch * stride_mask_batch + 
                idx_head * stride_mask_head + 
                offs_q[:, None] * stride_mask_q +
                offs_kv[None, :] * stride_mask_kv
            )

            ### Get block of mask ###
            custom_mask = tl.load(attn_mask_ptr + mask_offset, 
                                  mask=(offs_q[:, None] < SEQ_LEN_Q) & (offs_kv[None, :] < SEQ_LEN_KV),
                                  other=False)
            
            ### Set grads to 0 for masked positions ###
            P = tl.where(custom_mask, P, 0.)
        
        # Compute dP = dO @ V^T
        dP = tl.dot(dO_block, V_T_block)
        
        # Compute dS = P * (dP - D) * ln2
        dS = P * (dP - D_block) * ln2
        
        # Accumulate dQ: dQ += dS @ K
        dQ_block = tl.dot(dS.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16), tl.trans(K_T_block), acc=dQ_block)
    
    # Scale dQ
    dQ_block *= softmax_scale * rln2
    
    # Store dQ
    tl.store(dQ_ptr + Q_offsets, dQ_block.to(dQ_ptr.type.element_ty), mask=q_mask[:, None])

@triton.autotune(
    configs=get_bwd_dkdv_autotune_configs(),
    key=["SEQ_LEN_Q", "SEQ_LEN_KV", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd_dk_dv(
    Q_ptr, 
    K_ptr, 
    V_ptr, 
    dO_ptr, 
    dK_ptr, 
    dV_ptr,
    M_ptr, 
    D_ptr, 
    attn_mask_ptr,
    softmax_scale: tl.constexpr,
    stride_Q_batch, 
    stride_Q_head, 
    stride_Q_len, 
    stride_Q_embed,
    stride_K_batch, 
    stride_K_head, 
    stride_K_len, 
    stride_K_embed,
    stride_V_batch, 
    stride_V_head, 
    stride_V_len, 
    stride_V_embed,
    stride_mask_batch, 
    stride_mask_head, 
    stride_mask_q, 
    stride_mask_kv,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN_Q: tl.constexpr,
    SEQ_LEN_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr
):
    
    ln2: tl.constexpr = 0.693147182464
    rln2: tl.constexpr = 1.442695040888
    
    # Cast pointers
    Q_ptr = tl.cast(Q_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    K_ptr = tl.cast(K_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    V_ptr = tl.cast(V_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dO_ptr = tl.cast(dO_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dK_ptr = tl.cast(dK_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dV_ptr = tl.cast(dV_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    M_ptr = tl.cast(M_ptr, tl.pointer_type(tl.float32))
    D_ptr = tl.cast(D_ptr, tl.pointer_type(tl.float32))

    if USE_CUSTOM_MASK:
        attn_mask_ptr = tl.cast(attn_mask_ptr, tl.pointer_type(tl.int1))
        
    # Which block of KV are we processing?
    pid = tl.program_id(0)
    
    # Which batch/head?
    index_batch_head = tl.program_id(1)
    idx_batch = index_batch_head // NUM_HEADS
    idx_head = index_batch_head % NUM_HEADS

    offset_batch_head_4d_Q = idx_batch * stride_Q_batch + idx_head * stride_Q_head # for (B x H x L x E) Tensors
    offset_batch_head_4d_K = idx_batch * stride_K_batch + idx_head * stride_K_head # for (B x H x L x E) Tensors
    offset_batch_head_4d_V = idx_batch * stride_V_batch + idx_head * stride_V_head # for (B x H x L x E) Tensors
    offset_batch_head_3d = index_batch_head * SEQ_LEN_Q                            # for (B x H x L) Tensors

    Q_ptr += offset_batch_head_4d_Q
    K_ptr += offset_batch_head_4d_K
    V_ptr += offset_batch_head_4d_V
    dO_ptr += offset_batch_head_4d_Q
    dK_ptr += offset_batch_head_4d_K
    dV_ptr += offset_batch_head_4d_V    
    M_ptr += offset_batch_head_3d
    D_ptr += offset_batch_head_3d 

    ### Compute K/V Block Offsets ###
    offs_kv = pid * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
    offs_embed = tl.arange(0, HEAD_DIM)
    kv_mask = offs_kv < SEQ_LEN_KV

    ### Load K/V Blocks ###
    KV_Offsets = offs_kv[:, None] * stride_K_len + offs_embed[None, :] * stride_K_embed
    K = tl.load(K_ptr + KV_Offsets, mask=kv_mask[:, None], other=0.)
    V = tl.load(V_ptr + KV_Offsets, mask=kv_mask[:, None], other=0.)

    ### Prescale our Inputs ###
    K *= softmax_scale * rln2
    K = K.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16)

    ### Create empty tensors (in higher precision) to store our grads in ###
    dK_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)
    dV_block = tl.zeros([BLOCK_SIZE_KV, HEAD_DIM], dtype=tl.float32)

    ### Loop over Q Blocks ###
    num_q_blocks = tl.cdiv(SEQ_LEN_Q, BLOCK_SIZE_Q)

    for q_block_idx in range(num_q_blocks):
        
        start_q = q_block_idx * BLOCK_SIZE_Q
        offs_q = start_q + tl.arange(0, BLOCK_SIZE_Q)
        q_mask = offs_q < SEQ_LEN_Q

        ### Get Offsets for Q^T ###
        Q_T_offsets = offs_embed[:, None] * stride_Q_embed + offs_q[None, :] * stride_Q_len
        dO_offsets = offs_q[:, None] * stride_Q_len + offs_embed[None, :] * stride_Q_embed

        ### Load Q^T Block ###
        Q_T_block = tl.load(Q_ptr + Q_T_offsets, mask=q_mask[None, :], other=0.)

        ### Load Corresponding logsumexp, grads, and Ds ###
        M_block = tl.load(M_ptr + offs_q, mask=q_mask, other=0.)
        dO_block = tl.load(dO_ptr + dO_offsets, mask=q_mask[:, None], other=0.)
        D_block = tl.load(D_ptr + offs_q, mask=q_mask, other=0.)

        ### Compute our Block of Attention Matrix ###
        S_T_block = tl.dot(K, Q_T_block)

        ### Get our Softmax Back ###
        P_T_block = tl.math.exp2(S_T_block - M_block[None, :])
        
        ### If we had a custom attention mask we want to make sure we zero out any grads ###
        ### coming in from those masked positions ###
        if USE_CUSTOM_MASK:

            mask_offset_T = (
                idx_batch * stride_mask_batch + 
                idx_head * stride_mask_head + 
                offs_kv[:, None] * stride_mask_kv +
                offs_q[None, :] * stride_mask_q
            )

            custom_mask_T = tl.load(
                attn_mask_ptr + mask_offset_T,
                mask=(offs_kv[:, None] < SEQ_LEN_KV) & (offs_q[None, :] < SEQ_LEN_Q),
                other=False
            )

            P_T_block = tl.where(custom_mask_T, P_T_block, 0.)

        ### Compute dV which is P^T @ dO ###
        dV_block = tl.dot(P_T_block.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16), dO_block, acc=dV_block)

        ### dP = dO @ V^T, but we want dP^T so we transpose the right side and get [dO @ V^T]^T = V @ dO^T
        dP_T_block = tl.dot(V, tl.trans(dO_block))      

        ### Then our dS = P*(dP - D) but we again have all transposes so we just use our transpoed P and dP
        ### D is just a row vector that is the broadcasted over, so we add an extra dimension to make it (1 x Micro)
        dS_T_block = P_T_block * (dP_T_block - D_block[None, :]) * ln2
        dK_block = tl.dot(dS_T_block.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16), tl.trans(Q_T_block), acc=dK_block)

    ### Scale dK ###
    dK_block *= softmax_scale * rln2

    ### Store ###
    tl.store(dK_ptr + KV_Offsets, dK_block.to(dK_ptr.type.element_ty), mask=kv_mask[:, None])
    tl.store(dV_ptr + KV_Offsets, dV_block.to(dK_ptr.type.element_ty), mask=kv_mask[:, None])
    
def fused_cross_sdpa_forward(Q, K, V, 
                             attn_mask=None,
                             softmax_scale=None, 
                             use_dlpack=True):
    
    HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
    HEAD_DIM_V = V.shape[-1]
    BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM = Q.shape
    SEQ_LEN_KV = K.shape[2]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert Q.dtype == K.dtype and K.dtype == V.dtype, "Expect all Q,K,V Tensors to have the same data type"
    assert K.shape[2] == V.shape[2], "Keys and Values must have the same Sequence Length!"

    if softmax_scale is None:
        softmax_scale = 1 / HEAD_DIM**0.5

    if not DLPACK_DISABLE and use_dlpack:
        
        ### USE DLPACK to Convert our Cupy Arrays to Torch Tensors ###
        Q = torch.utils.dlpack.from_dlpack(Q)
        K = torch.utils.dlpack.from_dlpack(K)
        V = torch.utils.dlpack.from_dlpack(V)
        
        ### Check if we have Attention Mask ###
        use_custom_mask = attn_mask is not None

        if use_custom_mask:
            attn_mask = torch.utils.dlpack.from_dlpack(attn_mask)

        ### Make sure there is contiguous memory layout ####
        if not Q.is_contiguous():
            Q = Q.contiguous()
        if not K.is_contiguous():
            K = K.contiguous()
        if not V.is_contiguous():
            V = V.contiguous()
        if use_custom_mask and not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        # Create output tensors
        O = torch.empty_like(Q)
        M = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, dtype=torch.float32, device=Q.device)
        grid = lambda args: (triton.cdiv(SEQ_LEN_Q, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            attn_mask=attn_mask,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            stride_mask_batch=attn_mask.stride(0) if use_custom_mask else 0,
            stride_mask_head=attn_mask.stride(1) if use_custom_mask else 0,
            stride_mask_q=attn_mask.stride(2) if use_custom_mask else 0,
            stride_mask_kv=attn_mask.stride(3) if use_custom_mask else 0,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN_Q=SEQ_LEN_Q,
            SEQ_LEN_KV=SEQ_LEN_KV,
            HEAD_DIM=HEAD_DIM_Q,
            DTYPE_FLAG=0 if Q.dtype == torch.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

        ### Convert back to cupy ###
        Q = cp.from_dlpack(Q)
        K = cp.from_dlpack(K)
        V = cp.from_dlpack(V)
        O = cp.from_dlpack(O)
        M = cp.from_dlpack(M)

    else:

        ### Check if using attention mask ###
        use_custom_mask = attn_mask is not None
            
        ### Make sure there is contiguous memory layout ####
        if not Q.flags.c_contiguous:
            Q = cp.ascontiguousarray(Q)
        if not K.flags.c_contiguous:
            K = cp.ascontiguousarray(K)
        if not V.flags.c_contiguous:
            V = cp.ascontiguousarray(V)
        if use_custom_mask and not attn_mask.flags.c_contiguous:
            attn_mask = cp.ascontiguousarray(attn_mask)

        O = cp.empty_like(Q)
        grid = lambda args: (triton.cdiv(SEQ_LEN_Q, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

        # M is the logsumexp for the backward pass, one for each query
        # Make sure to create it on the right device as we are not using empty_like
        with cp.cuda.Device(Q.device.id):
            M = cp.empty(
                (BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q), dtype=cp.float32
            )

        _attn_fwd[grid](
            Q=Q.data.ptr,
            K=K.data.ptr,
            V=V.data.ptr,
            attn_mask=attn_mask.data.ptr,
            softmax_scale=softmax_scale,
            M=M.data.ptr,
            O=O.data.ptr,
            stride_Q_batch=Q.strides[0] // Q.itemsize,
            stride_Q_head=Q.strides[1] // Q.itemsize,
            stride_Q_seq=Q.strides[2] // Q.itemsize,
            stride_Q_dim=Q.strides[3] // Q.itemsize,
            stride_K_batch=K.strides[0] // Q.itemsize,
            stride_K_head=K.strides[1] // Q.itemsize,
            stride_K_seq=K.strides[2] // K.itemsize,
            stride_K_dim=K.strides[3] // K.itemsize,
            stride_V_seq=V.strides[2] // V.itemsize,
            stride_V_dim=V.strides[3] // V.itemsize,
            stride_O_seq=O.strides[2] // O.itemsize,
            stride_O_dim=O.strides[3] // O.itemsize,
            stride_mask_batch=attn_mask.strides[0] // attn_mask.itemsize if use_custom_mask else 0,
            stride_mask_head=attn_mask.strides[1] // attn_mask.itemsize if use_custom_mask else 0,
            stride_mask_q=attn_mask.strides[2] // attn_mask.itemsize if use_custom_mask else 0,
            stride_mask_kv=attn_mask.strides[3] // attn_mask.itemsize if use_custom_mask else 0,
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN_Q=SEQ_LEN_Q,
            SEQ_LEN_KV=SEQ_LEN_KV,
            HEAD_DIM=HEAD_DIM_Q,
            DTYPE_FLAG=0 if Q.dtype == cp.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

    return Q, K, V, O, M

def fused_cross_sdpa_backward(dO, 
                              Q, K, V, 
                              O, M, 
                              attn_mask=None,
                              softmax_scale=None,
                              use_dlpack=True):

    HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
    HEAD_DIM_V = V.shape[-1]
    BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, HEAD_DIM = Q.shape
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert Q.dtype == K.dtype and K.dtype == V.dtype and V.dtype == O.dtype, "Expect all Q,K,V,O Tensors to have the same data type"
    SEQ_LEN_KV = K.shape[2]

    ### Default softmax scale if not provided ###    
    if softmax_scale is None:
        softmax_scale = 1 / HEAD_DIM**0.5
        
    if not DLPACK_DISABLE and use_dlpack:

        dO = torch.utils.dlpack.from_dlpack(dO)
        Q = torch.utils.dlpack.from_dlpack(Q)
        K = torch.utils.dlpack.from_dlpack(K)
        V = torch.utils.dlpack.from_dlpack(V)
        O = torch.utils.dlpack.from_dlpack(O)
        M = torch.utils.dlpack.from_dlpack(M)

        ### Check if we have Attention Mask ###
        use_custom_mask = attn_mask is not None
        if use_custom_mask:
            attn_mask = torch.utils.dlpack.from_dlpack(attn_mask)

        ### Ensure our grads are contiguous ###
        if not dO.is_contiguous():
            dO = dO.contiguous()
        if use_custom_mask and not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        ### Ensure grads have the same dtype
        if not dO.dtype == Q.dtype:
            dO = dO.to(Q.dtype)

        ### Create Empty Grads to populate ###
        dQ = torch.zeros_like(Q, dtype=Q.dtype)
        dK = torch.zeros_like(K, dtype=K.dtype)
        dV = torch.zeros_like(V, dtype=V.dtype)
        D = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN_Q, dtype=torch.float32, device=Q.device)
    
        preprocess_grid = lambda meta: (triton.cdiv(SEQ_LEN_Q, meta["BLOCK_SIZE"]), BATCH_SIZE * NUM_HEADS)

        # Compute all the elements Di
        attn_backward_preprocess[preprocess_grid](
            O_ptr=O, 
            dO_ptr=dO,
            D_ptr=D,
            stride_O_heads=O.stride(1), 
            stride_O_len=O.stride(2), 
            stride_O_embed=O.stride(3),
            stride_dO_heads=dO.stride(1), 
            stride_dO_len=dO.stride(2), 
            stride_dO_embed=dO.stride(3),
            stride_D_head=D.stride(1),
            SEQ_LEN_Q=SEQ_LEN_Q,
            EMBED_DIM=HEAD_DIM,
            DTYPE_FLAG=0 if dO.dtype == torch.float32 else 1
        )
      
        grid_dq = lambda meta: (triton.cdiv(SEQ_LEN_Q, meta["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS)
        _attn_bwd_dq[grid_dq](
            Q_ptr=Q, K_ptr=K, V_ptr=V, dO_ptr=dO, dQ_ptr=dQ, M_ptr=M, D_ptr=D, attn_mask_ptr=attn_mask,
            softmax_scale=softmax_scale,
            stride_Q_batch=Q.stride(0), 
            stride_Q_head=Q.stride(1), 
            stride_Q_len=Q.stride(2), 
            stride_Q_embed=Q.stride(3),
            stride_K_batch=K.stride(0), 
            stride_K_head=K.stride(1), 
            stride_K_len=K.stride(2), 
            stride_K_embed=K.stride(3),
            stride_V_batch=V.stride(0), 
            stride_V_head=V.stride(1), 
            stride_V_len=V.stride(2), 
            stride_V_embed=V.stride(3),
            stride_mask_batch=attn_mask.stride(0) if attn_mask is not None else 0,
            stride_mask_head=attn_mask.stride(1) if attn_mask is not None else 0,
            stride_mask_q=attn_mask.stride(2) if attn_mask is not None else 0,
            stride_mask_kv=attn_mask.stride(3) if attn_mask is not None else 0,
            NUM_HEADS=NUM_HEADS, SEQ_LEN_Q=SEQ_LEN_Q, SEQ_LEN_KV=SEQ_LEN_KV, HEAD_DIM=HEAD_DIM,
            DTYPE_FLAG=0 if Q.dtype == torch.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

        grid_dk_dv = lambda meta: (triton.cdiv(SEQ_LEN_KV, meta["BLOCK_SIZE_KV"]), BATCH_SIZE * NUM_HEADS, 1)
        _attn_bwd_dk_dv[grid_dk_dv](
            Q_ptr=Q, 
            K_ptr=K, 
            V_ptr=V, 
            dO_ptr=dO, 
            dK_ptr=dK, 
            dV_ptr=dV, 
            M_ptr=M, 
            D_ptr=D,
            attn_mask_ptr=attn_mask,
            softmax_scale=softmax_scale,
            stride_Q_batch=Q.stride(0), 
            stride_Q_head=Q.stride(1), 
            stride_Q_len=Q.stride(2), 
            stride_Q_embed=Q.stride(3),
            stride_K_batch=K.stride(0), 
            stride_K_head=K.stride(1), 
            stride_K_len=K.stride(2), 
            stride_K_embed=K.stride(3),
            stride_V_batch=V.stride(0), 
            stride_V_head=V.stride(1), 
            stride_V_len=V.stride(2), 
            stride_V_embed=V.stride(3),
            stride_mask_batch=attn_mask.stride(0) if attn_mask is not None else 0,
            stride_mask_head=attn_mask.stride(1) if attn_mask is not None else 0,
            stride_mask_q=attn_mask.stride(2) if attn_mask is not None else 0,
            stride_mask_kv=attn_mask.stride(3) if attn_mask is not None else 0,
            NUM_HEADS=NUM_HEADS, SEQ_LEN_Q=SEQ_LEN_Q, SEQ_LEN_KV=SEQ_LEN_KV, HEAD_DIM=HEAD_DIM,
            DTYPE_FLAG=0 if Q.dtype == torch.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

        # Convert back to CuPy if needed
        dQ = cp.from_dlpack(dQ)
        dK = cp.from_dlpack(dK)
        dV = cp.from_dlpack(dV)
    
    else:

        use_custom_mask = attn_mask is not None

        ### Ensure our grads are contiguous ###
        if not dO.flags.c_contiguous:
            dO = cp.ascontiguousarray(dO)
        if use_custom_mask and not attn_mask.flags.c_contiguous:
            attn_mask = cp.ascontiguousarray(attn_mask)

        ### Ensure grads have the same dtype
        if not dO.dtype == Q.dtype:
            dO = dO.astype(Q.dtype)

        ### Create Empty Grads to populate ###
        dQ = cp.zeros_like(Q, dtype=Q.dtype)
        dK = cp.zeros_like(K, dtype=K.dtype)
        dV = cp.zeros_like(V, dtype=V.dtype)
        
        ### Default softmax scale if not provided ###    
        if softmax_scale is None:
            softmax_scale = 1 / HEAD_DIM**0.5

        # preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        preprocess_grid = lambda meta: (triton.cdiv(SEQ_LEN_Q, meta["BLOCK_SIZE"]), BATCH_SIZE * NUM_HEADS)

        ### This will contain our sum(O * dO, axis=-1) intermediate computation ###
        ### Do we need a kernel for this? probably not, but thats fine! ###
        D = cp.empty_like(M, dtype=cp.float32)  # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN)

        # Compute all the elements Di
        attn_backward_preprocess[preprocess_grid](
            O_ptr=O.data.ptr, 
            dO_ptr=dO.data.ptr,
            D_ptr=D.data.ptr,
            stride_O_heads=O.strides[1] // O.itemsize, 
            stride_O_len=O.strides[2] // O.itemsize, 
            stride_O_embed=O.strides[3] // O.itemsize,
            stride_dO_heads=dO.strides[1] // dO.itemsize, 
            stride_dO_len=dO.strides[2] // dO.itemsize, 
            stride_dO_embed=dO.strides[3] // dO.itemsize,
            stride_D_head=D.strides[1] // D.itemsize, 
            SEQ_LEN_Q=SEQ_LEN_Q,
            EMBED_DIM=HEAD_DIM,
            DTYPE_FLAG=0 if dO.dtype == cp.float32 else 1
        )
        
        grid_dq = lambda meta: (triton.cdiv(SEQ_LEN_Q, meta["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS)
        _attn_bwd_dq[grid_dq](
            Q_ptr=Q.data.ptr, 
            K_ptr=K.data.ptr, 
            V_ptr=V.data.ptr, 
            dO_ptr=dO.data.ptr, 
            dQ_ptr=dQ.data.ptr, 
            M_ptr=M.data.ptr, 
            D_ptr=D.data.ptr, 
            attn_mask_ptr=attn_mask.data.ptr,
            softmax_scale=softmax_scale,
            stride_Q_batch=Q.strides[0] // Q.itemsize, 
            stride_Q_head=Q.strides[1] // Q.itemsize, 
            stride_Q_len=Q.strides[2] // Q.itemsize, 
            stride_Q_embed=Q.strides[3] // Q.itemsize, 
            stride_K_batch=K.strides[0] // K.itemsize,  
            stride_K_head=K.strides[1] // K.itemsize,  
            stride_K_len=K.strides[2] // K.itemsize,  
            stride_K_embed=K.strides[3] // K.itemsize,  
            stride_V_batch=V.strides[0] // V.itemsize,  
            stride_V_head=V.strides[1] // V.itemsize,  
            stride_V_len=V.strides[2] // V.itemsize,  
            stride_V_embed=V.strides[3] // V.itemsize,  
            stride_mask_batch=attn_mask.strides[0] // attn_mask.itemsize if attn_mask is not None else 0,
            stride_mask_head=attn_mask.strides[1] // attn_mask.itemsize if attn_mask is not None else 0,
            stride_mask_q=attn_mask.strides[2] // attn_mask.itemsize if attn_mask is not None else 0,
            stride_mask_kv=attn_mask.strides[3] // attn_mask.itemsize if attn_mask is not None else 0,
            NUM_HEADS=NUM_HEADS, SEQ_LEN_Q=SEQ_LEN_Q, SEQ_LEN_KV=SEQ_LEN_KV, HEAD_DIM=HEAD_DIM,
            DTYPE_FLAG=0 if Q.dtype == torch.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

        grid_dk_dv = lambda meta: (triton.cdiv(SEQ_LEN_KV, meta["BLOCK_SIZE_KV"]), BATCH_SIZE * NUM_HEADS, 1)
        _attn_bwd_dk_dv[grid_dk_dv](
            Q_ptr=Q.data.ptr, 
            K_ptr=K.data.ptr, 
            V_ptr=V.data.ptr, 
            dO_ptr=dO.data.ptr, 
            dK_ptr=dK.data.ptr, 
            dV_ptr=dV.data.ptr, 
            M_ptr=M.data.ptr, 
            D_ptr=D.data.ptr,
            attn_mask_ptr=attn_mask.data.ptr,
            softmax_scale=softmax_scale,
            stride_Q_batch=Q.strides[0] // Q.itemsize, 
            stride_Q_head=Q.strides[1] // Q.itemsize, 
            stride_Q_len=Q.strides[2] // Q.itemsize, 
            stride_Q_embed=Q.strides[3] // Q.itemsize, 
            stride_K_batch=K.strides[0] // K.itemsize,  
            stride_K_head=K.strides[1] // K.itemsize,  
            stride_K_len=K.strides[2] // K.itemsize,  
            stride_K_embed=K.strides[3] // K.itemsize,  
            stride_V_batch=V.strides[0] // V.itemsize,  
            stride_V_head=V.strides[1] // V.itemsize,  
            stride_V_len=V.strides[2] // V.itemsize,  
            stride_V_embed=V.strides[3] // V.itemsize,  
            stride_mask_batch=attn_mask.strides[0] // attn_mask.itemsize if attn_mask is not None else 0,
            stride_mask_head=attn_mask.strides[1] // attn_mask.itemsize if attn_mask is not None else 0,
            stride_mask_q=attn_mask.strides[2] // attn_mask.itemsize if attn_mask is not None else 0,
            stride_mask_kv=attn_mask.strides[3] // attn_mask.itemsize if attn_mask is not None else 0,
            NUM_HEADS=NUM_HEADS, SEQ_LEN_Q=SEQ_LEN_Q, SEQ_LEN_KV=SEQ_LEN_KV, HEAD_DIM=HEAD_DIM,
            DTYPE_FLAG=0 if Q.dtype == torch.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

    return dQ, dK, dV


if __name__ == "__main__":
    import torch
    import math
    from torch.utils.dlpack import from_dlpack


    q = torch.randn((2,2,128,128), device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn((2,2,256,128), device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn((2,2,256,128), device="cuda", dtype=torch.float16, requires_grad=True)
    attn_mask = torch.ones((2,2,128,256)).bool().to("cuda")
    attn_mask[0, :, :, -20:] = False
    attn_mask[1, :, :, -4:] = False

    attn_mask_cp = cp.array(attn_mask.detach().cpu().numpy())
    o_grad = torch.randn_like(q)
    o_grad_cp = cp.array(o_grad.detach().cpu().numpy())
    out = torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask=attn_mask)
    out.backward(o_grad)
    q_grad_ref = cp.array(q.grad.detach().cpu().numpy())
    k_grad_ref = cp.array(k.grad.detach().cpu().numpy())
    v_grad_ref = cp.array(v.grad.detach().cpu().numpy())
    out_ref = cp.array(out.detach().cpu().numpy())


    q_cp = cp.array(q.detach().cpu().numpy())
    k_cp = cp.array(k.detach().cpu().numpy())
    v_cp = cp.array(v.detach().cpu().numpy())
    q_cp, k_cp, v_cp, O, M = fused_cross_sdpa_forward(q_cp,k_cp,v_cp, attn_mask_cp, use_dlpack=True)
    dQ_cp, dK_cp, dV_cp = fused_cross_sdpa_backward(o_grad_cp, q_cp, k_cp, v_cp, O, M, attn_mask_cp, use_dlpack=True)
    print(cp.max(cp.abs(O-out_ref)))
    print(cp.max(cp.abs(dQ_cp-q_grad_ref)))
    print(cp.max(cp.abs(dK_cp-k_grad_ref)))
    print(cp.max(cp.abs(dV_cp-v_grad_ref)))
