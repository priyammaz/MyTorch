"""
FlashAttention2 Kernel (Online-Softmax Applied to Attention)

Some really awesome resources that this was largely based off of!
1) Umar Jamil: https://github.com/hkproj/triton-flash-attention
2) Evintunador: https://github.com/evintunador/triton_docs_tutorials

And of course this is also based off of the official implementation provided by Triton!
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py

This adapts the existing code with Cupy!

1) NO Dropout on our attention scores! Maybe another feature to come?

"""
import os
import torch
import cupy as cp
import triton
import triton.language as tl
from .flags import DLPACK_DISABLE, AUTOTUNE_MODE

def get_fwd_autotune_configs():

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

def get_bwd_autotune_configs():

    mode = AUTOTUNE_MODE
    
    # Single config for "none" mode (no autotuning, fixed configuration)
    if mode == "none":
        return [
            triton.Config(
                {"BLOCK_SIZE_MACRO": 64, "BLOCK_SIZE_MICRO": 32},
                num_stages=2,
                num_warps=4,
            )
        ]
    
    # Reduced configs for "medium" mode (fewer configurations for faster tuning)
    if mode == "medium":
        return [
            triton.Config(
                {"BLOCK_SIZE_MACRO": BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_MACRO in [32, 64]  # Reduced set
            for BLOCK_SIZE_MICRO in [16, 32]  # Reduced set
            for num_stages in [2, 3]         # Reduced set
            for num_warps in [4, 8]          # Reduced set
            if BLOCK_SIZE_MICRO < BLOCK_SIZE_MACRO
        ]
    
    # Comprehensive configs for "max" mode (full search space)
    else:  # mode == "max" or any other value defaults to max
        return [
            triton.Config(
                {"BLOCK_SIZE_MACRO": BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO},
                num_stages=num_stages,
                num_warps=num_warps,
            )
            for BLOCK_SIZE_MACRO in [16, 32, 64, 128]
            for BLOCK_SIZE_MICRO in [16, 32, 64]
            for num_stages in [2, 3, 4]
            for num_warps in [4, 8, 16]
            if BLOCK_SIZE_MICRO < BLOCK_SIZE_MACRO
        ]

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    attn_mask_ptr, # Masks are the shape (B x H x L x L)
    block_index_q,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_KV,
    PASS_TYPE: tl.constexpr, # 0: pre_diag, 1: diag, 2: full
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN,
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

    if PASS_TYPE == 0:
        ### When in Causal Mode we need to first compute the prediagonal:
        ### I want indexes (for my K,V) that are upto the ###
        ### index of my queries. This way my queries only attend to ###
        ### Keys and values that are before it. 

        ### This applies to all K,V before the diagonal. These are all blocks 
        ### of queries, as long as we are before the diagonal i know for sure 
        ### that every KV must be less that my query. Lets say we have the 
        ### following output from our blockes QKT

        ### [qk00 qk01 qk02 qk03]
        ### [qk10 qk11 qk12 qk13]
        ### [qk20 qk21 qk22 qk23]
        ### [qk30 qk31 qk32 qk33]

        ### And each qk00 is a block of values (lets say 3 x 3). I know for sure
        ### that every value in qk10, qk20, qk21, qk30, qk31, qk32 dont break any 
        ### causality. every value in those specific blocks that queries are ###
        ### looking at k/v that are <= in index 

        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif PASS_TYPE == 1:

        ### When in causal we need to also process the diagonal but they have a transition
        ### Lets say we grab the top left corner (qk00) and each block is processing 
        ### 3 queries and 3 keys. In our output:

        ### [x00 x01 x02]
        ### [x10 x11 x12]
        ### [x20 x21 x22]

        ### x01 x02 x12 are invalid positions as that would mean the query vector
        ### is attending to a vector after it So we just need to remove these extra 
        ### ones. This is just more post processing!

        ### The block we want is the one at the end of our completely valid Q blocks and the 
        ### next one after. So we essentialyl have a low/high containing onyl the single block 
        ### where Q ends! Its just in that diagonal we have to mask out half that block. 

        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:

        ### When not causal we just want to process the entire sequence as we attend to everything. 
        lo, hi = 0, SEQ_LEN

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
        kv_padding_mask = kv_indices < SEQ_LEN

        ### Load our K and V Blocks ###
        K_block = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        ### We have a (Q_BLOCK_SIZE x E) and (E x KV_BLOCK_SIZE) matricies ###
        ### we can just use dot to do our dot product computation ### 
        QK_block = tl.dot(Q_block, K_block)

        if PASS_TYPE == 1:

            # Post process the diagonal 
            # off_q is the indexes of the queries we are processing
            # offs_kv is the indexes of the keys/values we are processing
            # we can offset our offs_kv for this specific iteration in the loop
            # and do a broadcast check to see for what spots every q is greater than our 
            # k positions.
            # 
            # [0]   [0 1 2 3] -> [True False False False]
            # [1]                [True True  False False]
            # [2]                [True True  True  False]
            # [3]                [True True  True  True]
            # and then we can just fill the False with a large negative number!

            causal_mask = offs_q[:, None] >= kv_indices[None, :]
            mask = causal_mask & kv_padding_mask[None, :]
            QK_block += tl.where(mask, 0, float("-inf"))

        else:

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
                                  mask=(offs_q[:, None] < SEQ_LEN) & (kv_indices[None, :] < SEQ_LEN),
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
    key=["SEQ_LEN", "HEAD_DIM"],
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
    SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    ATTN_MODE: tl.constexpr, # 0 for non_causal, 1 for causal
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16,
    USE_CUSTOM_MASK: tl.constexpr,
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

    ### In the original softmax implementation we use e^(x), but for mixed-precision stability ###
    ### and efficiency we will instead use 2^(x). But this will obviously give a different value ###
    ### So we need to preadjust our values before we compute our 2^(x). ###
    ### recall your log rules! ###
    ### a^(log_a(b)) = b, so taking a=2 and b=e we get:
    ### 2^(log_2(e)) = e
    ### now log_2(e) = log_e(e) / log_e(2) = ln(e) / ln(2) = 1 / ln(2)
    ### Thus we can write that e = 2 ^ (1 / ln(2))
    ### and raising both sides by an exponential:
    ### e^x = 2^(x/ln(2))
    ### Thus, if we predivide all our values by ln(2), then when we do 2^(x) thats the same as doing e^(x)!
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
    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head
    )

    ### Who likes pointer arithmetic? Remember, my Q data is:
    ### Q.shape = (BATCH x HEADS x SEQ_LEN x EMBED_DIM)
    ### Each thread will process a specific BATCH and HEAD as well as a BLOCK of our SEQ_LEN
    ### So I need ot basically do a Q[batch_idx, head_idx, start_q_idx:end_q_idx, :]
    ### To do this with pointer arithmetic it would kind of look like:

    # row_offset = block_index_q * BLOCK_SIZE_Q                ### starting query vector idx in block
    # col_offset = 0                                           ### no column offset as we want the entire embedding vector

    # for i in range(BLOCK_SIZE_Q):                            ### for every query index I want
    #     for j in range(HEAD_DIM):                            ### for the head I am in
    #         ptr = (Q + qkv_offset                            ### We want to start art the right batch/head starting point and then move over by the row/col offset
    #                 + (row_offset + i) * stride_Q_seq
    #                 + (col_offset + j) * stride_Q_dim)
    #         val = tl.load(ptr)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,                      ### Offset to the batch/head we are processing
        shape=(SEQ_LEN, HEAD_DIM),                ### Because we already indexed batch/head the shape that is left is just (SEQ_LEN, HEAD_DIM)
        strides=(stride_Q_seq, stride_Q_dim),     ### What are the strides of the remaining dimensions
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),### Indexes of the Block of queries we are processing
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),     ### What is the shape our our block of queries?
        order=(1,0)                               ### Memory coalescing. We make our HEAD DIM in contiguous memory addresses for fast access over the embeddings
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,                      
        shape=(SEQ_LEN, HEAD_DIM),                
        strides=(stride_V_seq, stride_V_dim),     
        offsets=(0,0),                            ### When loading values we dont skip anything, as we will for loop over this in a bit
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),              
        order=(1,0)                               
    )

    ### Switching our strides transposes our matrix. Take for example
    ### [A B C]
    ### [D E F]
    
    ### This has a strides[0] of 3 and strides[1] of 1. Now in memory, its actually
    ### stored as [A B C D E F]
    
    ### So what if we make our stride[0] = 1 and stride[1] = 3?

    ### Starting at A, to get to the next column, we have to move over 3. So A next 
    ### to A you have D. To get to the next row you move over 1, so from A next next
    ### row would be B. And that is exactly the transpose if you keep going!
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,                      
        shape=(HEAD_DIM, SEQ_LEN),                ### Set shape to transpose dimension
        strides=(stride_K_dim, stride_K_seq),     ### invert the stride     
        offsets=(0,0),                           
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),              
        order=(0,1)                               ### We want contiguous memory along the HEAD_DIM first
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,                      
        shape=(SEQ_LEN, HEAD_DIM),               
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

    ### If we are causal (ATTN_MODE==1) then we only process the pre-diagonal stuff (pass_type==0) ###
    ### otherwise we process everything (full attention) ###
    pass_type = 0 if ATTN_MODE == 1 else 2
    O_block, l_i, m_i = _attn_fwd_inner(
        O_block, 
        l_i, 
        m_i, 
        Q_block, 
        K_block_ptr, 
        V_block_ptr, 
        attn_mask,
        block_index_q, 
        BLOCK_SIZE_Q, 
        BLOCK_SIZE_KV, 
        pass_type,
        offs_q, 
        offs_kv, 
        SEQ_LEN,
        DTYPE_FLAG,
        USE_CUSTOM_MASK, 
        stride_mask_batch, 
        stride_mask_head, 
        stride_mask_q, 
        stride_mask_kv, 
        index_batch, 
        index_head
    )

    ### IF we are causal, we need to separately handle the diagonal ###
    ### so that is taken care of here with pass_type==1 ###
    if ATTN_MODE == 1:

        ### If we are doing causal attention, the blocks on the diagonal contain values that contain a transition. ###
        ### for example lets say we look at the top left block, and lets say each block is 4x4 ###

        ### [qk_00, qk_01, qk_02, qk_03]
        ### [qk_10, qk_11, qk_12, qk_13]
        ### [qk_20, qk_21, qk_22, qk_23]
        ### [qk_30, qk_31, qk_32, qk_33]

        ### The issue is we are processing entire blocks at a time, but the elements of this block are not all valid. ###
        ### If we are causal, qk_01 for example, means query at time 0 is attending to a key in a future time 1. ###
        ### This breaks causality. So in the previous part, we already computed all the blocks upto the diagonal (if causal) ###

        ### Lets look at it at the block level, remember each block has 4x4 elements inside them:

        ### [B_00, B_01, B_02, B_03]
        ### [B_10, B_11, B_12, B_13]
        ### [B_20, B_21, B_22, B_23]
        ### [B_30, B_31, B_32, B_33]

        ### An in this case our B_00 block is the top left block from above.

        ### In the first stage (if causal) we can directly compute everything in 
        ### B_10, B_20, B_21, B_30, B_31, B_32

        ### Because its guaranteed that every query in that block is attending to keys that are before it
        ### But along our diagonal B_00, B_11, B_22, B_33, we have a transition inside the block
        
        ### Again for B_00 we had 

        ### [qk_00, qk_01, qk_02, qk_03]
        ### [qk_10, qk_11, qk_12, qk_13]
        ### [qk_20, qk_21, qk_22, qk_23]
        ### [qk_30, qk_31, qk_32, qk_33]

        ### We need to compute the diagonal and this is easiest if we just do it separately and then 
        ### make sure to mask out the top triangle portion so we get 

        ### [qk_00, -inf , -inf , -inf ]
        ### [qk_10, qk_11, -inf , -inf ]
        ### [qk_20, qk_21, qk_22, -inf ]
        ### [qk_30, qk_31, qk_32, qk_33]

        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, 
            l_i, 
            m_i, 
            Q_block, 
            K_block_ptr, 
            V_block_ptr, 
            attn_mask,
            block_index_q, 
            BLOCK_SIZE_Q, 
            BLOCK_SIZE_KV, 
            1,
            offs_q, 
            offs_kv, 
            SEQ_LEN,
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
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    q_padding_mask = offs_q < SEQ_LEN
    tl.store(m_ptrs, m_i, mask=q_padding_mask)

    ### Store Q (again with a boundary check) ###
    tl.store(O_block_ptr, O_block.to(O.type.element_ty), boundary_check=(0,))

@triton.autotune(
    configs=get_preprocess_autotune_configs(),
    key=["SEQ_LEN", "HEAD_DIM"],
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
    SEQ_LEN,
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
    mask = row_offsets < SEQ_LEN

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

@triton.jit
def _attn_bwd_dk_dv(
    K, 
    V, 
    dK, 
    dV, 
    Q_ptr, 
    dO_ptr, 
    M_ptr, 
    D_ptr, 
    attn_mask_ptr,
    stride_len, 
    stride_embed, 
    stride_mask_q, 
    stride_mask_kv,
    SEQ_LEN, 
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_ROW: tl.constexpr, 
    BLOCK_SIZE_COL: tl.constexpr,
    start_row, 
    start_col, 
    num_steps, 
    ln2, 
    MASK: tl.constexpr,
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16
    USE_CUSTOM_MASK: tl.constexpr, 
    mask_offset_base,
):
    """
    Main method to compute the grads for dK,dV in blocks. This basically
    assumes for some K,V we loop through blocks of queries
    """
    ### Fet Offset for starting row/col ###
    offsets_row = start_row  + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_col = start_col + tl.arange(0, BLOCK_SIZE_COL)
    offsets_embed = tl.arange(0, HEAD_DIM)

    ### Load Transposed Q because we are computing the transpose of our softmax outputs ###
    Q_T_offsets = offsets_embed[:, None] * stride_embed + offsets_row[None, :] * stride_len
    dO_offsets = offsets_row[:, None] * stride_len + offsets_embed[None, :] * stride_embed

    for _ in range(num_steps):
        
        ### Dont grab invalid queries ###
        mask_Q = offsets_row < SEQ_LEN

        ### Load our transpose queries ###
        Q_T_block = tl.load(Q_ptr + Q_T_offsets, mask=mask_Q[None, :], other=0.)

        ### Load the corresponding logsumexps, grads and Ds ###
        M_block = tl.load(M_ptr + offsets_row, mask=mask_Q, other=0.)
        dO_block = tl.load(dO_ptr + dO_offsets, mask=mask_Q[:, None], other=0.)
        D_block = tl.load(D_ptr + offsets_row, mask=mask_Q, other=0.)

        ### We can compute our block of the attention matrix now ###
        S_T_block = tl.dot(K, Q_T_block) #(Macro x E) @ (E x Micro) -> Macro x Micro block

        ### Now lets do softmax without actually doing softmax ! ###
        ### This is one of the most important parts of the implementation! 
        ### What we want is the softmaxed output in this block. But that would
        ### require us to store the entire N x N matrix in memory. So can we instead
        ### Compute it on the fly? In the forward pass we did online softmax, but we 
        ### can avoid that too

        ### remember we have stored m, our absolute max + log(denominator) 
        ### for every row of our softmax These were computed in the forward 
        ### pass so we can avoid doing it again. 

        ### Recall softmax: P_ij = exp(S_ij) / sum(exp(S_i))
        ### but we want stable softmax so instead we do
        ### softmax: P_ij = exp(S_ij - max(S_i)) / sum_j(exp(S_ij - max(S_i))
        ### and we already have m as our max so we can say:
        ### softmax: P_ij = exp(S_ij - m_i) / sum_j(exp(S_ij - m_i))

        ### So, what happens if we do this:

        ### exp(QK^T - m) = exp(QK^T - max - log(denominator))
        ### = exp(QK^T - max) / exp(log(denominator))
        ### Isnt that just our softmax? yes! So we can get our softmax back really
        ### easily with this trick!!!
        P_T_block = tl.math.exp2(S_T_block - M_block[None, :])

        if MASK:
            ### Our P is transposed here. If causal, in the forward pass ###
            ### only the lower triangle matters. This means in the backward ###
            ### pass only the lower triangle matters, but because its transposed ###
            ### we want the upper triangle instead! ###
            mask_block = (offsets_col[:, None] <= offsets_row[None, :])

            ### Set our invalid positions to 0 ###
            P_T_block = tl.where(mask_block, P_T_block, 0.)

        ### If we had an attention mask, we want to make sure we 0 out ###
        ### any of the grads are coming from these masked positions! ###
        if USE_CUSTOM_MASK:
            
            ### Compute the indexes for the block of mask we want ###
            mask_offset = (
                mask_offset_base + 
                offsets_row[None, :] * stride_mask_q + 
                offsets_col[:, None] * stride_mask_kv
            )

            ### Grab that that block of mask ###
            custom_mask = tl.load(attn_mask_ptr + mask_offset, 
                                  mask=(offsets_row[None, :] < SEQ_LEN) & (offsets_col[:, None] < SEQ_LEN),
                                  other=False)

            ### Fill invalid positions with 0! ###
            P_T_block = tl.where(custom_mask, P_T_block, 0.)

        ### Now we start to accumulate grads. Each block of the output contribute to our 
        ### gradient for dV. dV is P^T @ dO
        ### But we are not processing all of our sequence length at once, only chunks of it
        ### and our dV is dependent on contributions from the entire length so we can 
        ### just accumulate as we go for the correct positions we are processing
        dV = tl.dot(P_T_block.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16), dO_block, acc=dV)

        ### dP = dO @ V^T, but we want dP^T so we transpose the right side and get [dO @ V^T]^T = V @ dO^T
        dP_T_block = tl.dot(V, tl.trans(dO_block))        

        ### Then our dS = P*(dP - D) but we again have all transposes so we just use our transpoed P and dP
        ### D is just a row vector that is the broadcasted over, so we add an extra dimension to make it (1 x Micro)
        dS_T_block = P_T_block * (dP_T_block - D_block[None, :]) * ln2
        dK = tl.dot(dS_T_block.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16), tl.trans(Q_T_block), acc=dK)

        ### Advance to the next query block 
        offsets_row += BLOCK_SIZE_ROW  
        Q_ptr += BLOCK_SIZE_ROW * stride_len
        dO_ptr += BLOCK_SIZE_ROW * stride_len

    return dK, dV

@triton.jit
def _attn_bwd_dq(
    dQ, 
    Q, 
    dO, 
    M, 
    K_ptr, 
    V_ptr, 
    D_ptr, 
    attn_mask_ptr,
    stride_len, 
    stride_embed, 
    stride_mask_q, 
    stride_mask_kv,
    SEQ_LEN, 
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr, 
    start_row, 
    start_col, 
    num_steps, 
    ln2: tl.constexpr, 
    MASK: tl.constexpr,
    DTYPE_FLAG: tl.constexpr, # 0 for float32, 1 for float16
    USE_CUSTOM_MASK: tl.constexpr, 
    mask_offset_base
):
    """
    Nearly identical for _attn_bwd_dk_dv but now we have a block of Q and are 
    looping through blocks of K,V to compute out dQ. And instead of computing 
    some transpose of our blocks of the attention matrix, we compute the normal
    non-transposed version as thats all we need
    """
    offsets_row = start_row + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_col = start_col + tl.arange(0, BLOCK_SIZE_COL)
    offsets_embed = tl.arange(0, HEAD_DIM)

    K_V_T_offsets = offsets_embed[:, None] * stride_embed + offsets_col[None, :] * stride_len
    D_block = tl.load(D_ptr + offsets_row, mask=offsets_row<SEQ_LEN, other=0.)

    for _ in range(num_steps):
        
        ### Dont grab invalid masks ###
        mask_kv = offsets_col < SEQ_LEN

        K_T_block = tl.load(K_ptr + K_V_T_offsets, mask=mask_kv[None, :], other=0.)
        V_T_block = tl.load(V_ptr + K_V_T_offsets, mask=mask_kv[None, :], other=0.)

        ### Compute our standard QK^T
        S = tl.dot(Q, K_T_block)

        ### Logsumexp trick to get our softmax values back 
        P = tl.exp2(S - M)
        
        ### Mask for causality 
        if MASK:
            mask = offsets_row[:, None] >= offsets_col[None, :]
            P = tl.where(mask, P, 0.)

        ### Custom Attention Mask ###
        if USE_CUSTOM_MASK:

            mask_offset = (
                mask_offset_base + 
                offsets_row[:, None] * stride_mask_q + 
                offsets_col[None, :] * stride_mask_kv
            )

            custom_mask = tl.load(attn_mask_ptr + mask_offset,
                                 mask=(offsets_row[:, None] < SEQ_LEN) & (offsets_col[None, :] < SEQ_LEN),
                                 other=False)

            P = tl.where(custom_mask, P, 0.)

        ### Same formulation just for dQ now ###
        dP = tl.dot(dO, V_T_block)
        dS = P * (dP - D_block[:, None]) * ln2
        dQ = tl.dot(dS.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16), tl.trans(K_T_block), acc=dQ)

        ### Advance to the next block of Keys/Values ###
        offsets_col += BLOCK_SIZE_COL
        K_ptr += BLOCK_SIZE_COL * stride_len
        V_ptr += BLOCK_SIZE_COL * stride_len
    
    return dQ

@triton.autotune(
    configs=get_bwd_autotune_configs(),
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_bwd(
    Q_ptr, 
    K_ptr, 
    V_ptr, 
    dO_ptr, 
    dQ_ptr, 
    dK_ptr, 
    dV_ptr, 
    M_ptr, 
    D_ptr, 
    attn_mask_ptr,
    softmax_scale, 
    stride_batch, 
    stride_head, 
    stride_len, 
    stride_embed, 
    stride_mask_batch, 
    stride_mask_head, 
    stride_mask_q, 
    stride_mask_kv,
    NUM_HEADS, 
    SEQ_LEN, 
    HEAD_DIM: tl.constexpr, 
    BLOCK_SIZE_MICRO: tl.constexpr,
    BLOCK_SIZE_MACRO: tl.constexpr,
    CAUSAL: tl.constexpr, # 1 for causal, 0 for noncausal 
    DTYPE_FLAG: tl.constexpr, # 0 for float32 1 for float16,
    USE_CUSTOM_MASK: tl.constexpr
):
    
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    ### Store our Contants for scaling due to exp2 vs exp difference ###
    ln2: tl.constexpr = 0.693147182464
    rln2: tl.constexpr = 1.442695040888

    ### Cast all our pointers ###
    Q_ptr = tl.cast(Q_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    K_ptr = tl.cast(K_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    V_ptr = tl.cast(V_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dO_ptr = tl.cast(dO_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dQ_ptr = tl.cast(dQ_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dK_ptr = tl.cast(dK_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    dV_ptr = tl.cast(dV_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else tl.float16))
    M_ptr = tl.cast(M_ptr, tl.pointer_type(tl.float32))
    D_ptr = tl.cast(D_ptr, tl.pointer_type(tl.float32))

    if USE_CUSTOM_MASK:
        attn_mask_ptr = tl.cast(attn_mask_ptr, tl.pointer_type(tl.int1))

    ### What Block are we processing? ###
    pid = tl.program_id(0)

    ### What Batch/Head are we on? ###
    index_batch_head = tl.program_id(1)
    offsets_embed = tl.arange(0, HEAD_DIM)
    idx_batch = index_batch_head // NUM_HEADS
    idx_head = index_batch_head % NUM_HEADS

    ### Offset everything to our current Batch x Head ##
    offset_batch_head_4d = idx_batch * stride_batch + idx_head * stride_head # for (B x H x L x E) Tensors
    offset_batch_head_3d = index_batch_head * SEQ_LEN                        # for (B x H x L) Tensors

    Q_ptr += offset_batch_head_4d
    K_ptr += offset_batch_head_4d
    V_ptr += offset_batch_head_4d
    dO_ptr += offset_batch_head_4d
    dQ_ptr += offset_batch_head_4d
    dK_ptr += offset_batch_head_4d
    dV_ptr += offset_batch_head_4d
    M_ptr += offset_batch_head_3d
    D_ptr += offset_batch_head_3d

    ### If we have an attention mask, we can compute which batch/head of our mask we want to index later ###
    mask_offset_base = 0
    if USE_CUSTOM_MASK:
        mask_offset_base = (
            idx_batch * stride_mask_batch +
            idx_head * stride_mask_head
        )

    ###################### dK dV #####################

    ### Rows are the number of queries in every block we loop over 
    ### Cols are the number of Keys/Values in our block that we hold constant in this specific thread
    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO

    ### STAGE 1: Process the Diagonal Block ###
    ### Just like in the forward pass our diagonal block has a ###
    ### Transition from causal to non-causal positions. ###
    ### Lets process that first!
    if CAUSAL == 1:

        ### Index of the starting column (starting key/value index)
        start_col = pid * BLOCK_SIZE_COL_1

        ### The diagonal starts where our starting query index matches the starting key/value
        start_row = start_col

        ### Incase our blocks are not sqaure, it can take multiple micro iterations of our queries
        ### to cover everything. For example, if a block of keys/values contain 64 timesteps, but
        ### each block of queries we loop over has only 16 timesteps, it will take 4 steps to 
        ### get through the full 64 x 64 block
        num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1
    
    ### If we are not causal then there isnt really anything to do, 
    ### we loop over all possible Queries for this specific block of Keys/Values
    else:

        ### go from start to end for queries ###
        start_row = 0

        ### Processing this specific block of keys/values ###
        start_col = pid * BLOCK_SIZE_COL_1

        ### Just go however many blocks worth of queries it takes to cover the 
        ### entire sequence length 
        num_steps = tl.cdiv(SEQ_LEN, BLOCK_SIZE_ROW_1)

    ### Load K/V ###
    ### Instead of QK^T we will do KQ^T, giving us a transposed ###
    ### output. This is because our dV is P^T @ dO, so might as well ###
    ### just transpose it now rather than grab it normally and transpose after ###
    offsets_col_1 = start_col + tl.arange(0, BLOCK_SIZE_COL_1)

    ### Ensure we dont grab any invalid KV positions ###
    KV_offsets = offsets_col_1[:, None] * stride_len + offsets_embed[None, :] * stride_embed
    KV_mask = (offsets_col_1 < SEQ_LEN)

    ### Load our data ! ###
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask[:, None], other=0.)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask[:, None], other=0.)
    
    ### Prescale our inputs like we did in our forward pass ###
    K *= softmax_scale * rln2
    K = K.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16)

    ### Create empty tensors (in higher precision) to store our grads in ###
    dK_block = tl.zeros([BLOCK_SIZE_COL_1, HEAD_DIM], dtype=tl.float32)
    dV_block = tl.zeros([BLOCK_SIZE_COL_1, HEAD_DIM], dtype=tl.float32)

    ### Run our backward pass for that block on the diagonal (if we are in causal mode) 
    ### or for everything (if we are in non causal) ###
    ### If we are in causal model then we will Mask as a part of that diagonal is invalid 
    dK_block, dV_block = _attn_bwd_dk_dv(
        K, V, dK_block, dV_block, 
        Q_ptr, dO_ptr, M_ptr, D_ptr, attn_mask_ptr, 
        stride_len, stride_embed, 
        stride_mask_q, stride_mask_kv,
        SEQ_LEN, HEAD_DIM, 
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_row, start_col, num_steps, 
        ln2, 
        MASK=(CAUSAL==1),
        DTYPE_FLAG=DTYPE_FLAG,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        mask_offset_base=mask_offset_base
    )

    ### STAGE 2: Process Under the Diagonal Block for Causal###
    ### If we are in causal mode we need to do all the other off diagonal positions. ###
    ### Now lets say we had the following block setup. Remember, each block has more ###
    ### values inside it but we are processing at the block level

    ### [B_00, B_01, B_02, B_03]
    ### [B_10, B_11, B_12, B_13]
    ### [B_20, B_21, B_22, B_23]
    ### [B_30, B_31, B_32, B_33]

    ### and lets say we processed B_11 just now (a diagonal block with transition from causal to non causal positions) 
    ### Each thread here processes a column, as we picked a specific Key/Value, and we can loop through our queries. 
    ### This means if we are causal, we need to also process B_21, and B_31! So lets move our starting row forward 
    ### however many values there were in our keys/values and keep iterating downwards. 
    if CAUSAL == 1:
        
        ### Push our starting point for the queries forward 1 blocks worth of Keys/Values 
        start_row += BLOCK_SIZE_COL_1

        ### keys/values block size worth of blocks do I need to cover the entire sequence?
        ### And then multiply by by the blocks size to get the total number of timesteps. 
        ### this may spill over the edge but thats ok we handle it with masking later!
        N_adj = tl.cdiv(SEQ_LEN, BLOCK_SIZE_COL_1) * BLOCK_SIZE_COL_1

        ### Take the total number of steps we need to cover the entire sequence, subtract out
        ### the number of steps we have already taken, and then get how many blocks of size queries we 
        ### need to cover that distance! 
        num_steps = (N_adj - start_row) // BLOCK_SIZE_ROW_1

        ### Backward pass again on these blocks underneath that diagonal block for the Causal Case 
        dK_block, dV_block = _attn_bwd_dk_dv(
            K, V, dK_block, dV_block, 
            Q_ptr, dO_ptr, M_ptr, D_ptr, attn_mask_ptr, 
            stride_len, stride_embed, 
            stride_mask_q, stride_mask_kv,
            SEQ_LEN, HEAD_DIM, 
            BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
            start_row, start_col, num_steps, 
            ln2, 
            MASK=False,
            DTYPE_FLAG=DTYPE_FLAG,
            USE_CUSTOM_MASK=USE_CUSTOM_MASK,
            mask_offset_base=mask_offset_base
        )

    ### We didnt apply this scaling in our loop (as its just a constant) ###
    ### but if we had some scaling on our input, then the backprop will also have ###
    ### exactly the same scale (y = aX => dy/dx = a) 
    dK_block *= softmax_scale * rln2

    ### Store it with the mask! ###
    tl.store(dK_ptr + KV_offsets, dK_block, mask=KV_mask[:, None])
    tl.store(dV_ptr + KV_offsets, dV_block, mask=KV_mask[:, None])

    ###################### dQ ##################### 

    ### Now we are grabbing some Q and looping over K,Vs. So we will 
    ### have Macro blocks of Q and loop through micro blocks of KVs
    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO

    ### STAGE 1: Process the Diagonal Block ###
    ### Same setup as before, just now we go the other direction ###
    ### our pid sets which block of rows of queries we grab ###
    ### and our diagnal K/V will have the same starting point along the cols 
    if CAUSAL == 1:
        start_row = pid * BLOCK_SIZE_ROW_2
        start_col = start_row
        num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2
    
    ### In non causal we just process everything ###
    else:
        start_col = 0
        start_row = pid * BLOCK_SIZE_ROW_2
        num_steps = tl.cdiv(SEQ_LEN, BLOCK_SIZE_COL_2)

    ### Compute offsets to grab our queries (also applies to our outputs) ###
    offsets_row = start_row + tl.arange(0, BLOCK_SIZE_ROW_2)
    Q_offsets = offsets_row[:, None] * stride_len + offsets_embed[None, :] * stride_embed

    ### Mask out any invalid queries we grabbed ###
    mask_row = offsets_row < SEQ_LEN

    ### Load our Queries ###
    Q_block = tl.load(Q_ptr + Q_offsets, mask=mask_row[:, None], other=0.)
    
    ### Prescale our Queries ###
    Q_block *= softmax_scale * rln2
    Q_block = Q_block.to(tl.float32 if DTYPE_FLAG == 0 else tl.float16)

    ### Load our gradients for this specific block ###
    dO_block = tl.load(dO_ptr + Q_offsets, mask=mask_row[:, None], other=0.)

    ### These were the logsumexp values along the rows of queries we had in our attention matrix ###
    ### this means we can just grab the corresponding block right here and pass it in rather than ###
    ### grabing them a block at a time like we did earlier in our dKdV computation ###
    M_block = tl.load(M_ptr + offsets_row, mask=mask_row, other=0.)[:, None]

    ### Create a tensor for grad storage ###
    dQ_block = tl.zeros([BLOCK_SIZE_ROW_2, HEAD_DIM], dtype=tl.float32)

    ### First pass ###
    dQ_block = _attn_bwd_dq(
        dQ_block, Q_block, dO_block, M_block, 
        K_ptr, V_ptr, D_ptr, attn_mask_ptr,
        stride_len, stride_embed,
        stride_mask_q, stride_mask_kv,
        SEQ_LEN, HEAD_DIM, 
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2, 
        start_row, start_col, num_steps, 
        ln2, MASK=(CAUSAL==1),
        DTYPE_FLAG=DTYPE_FLAG,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        mask_offset_base=mask_offset_base,
    )

    ### Second pass (only for causal models) ###
    if CAUSAL == 1:
        end_col = start_col
        start_col = 0
        num_steps = end_col // BLOCK_SIZE_COL_2
        dQ_block = _attn_bwd_dq(
            dQ_block, Q_block, dO_block, M_block, 
            K_ptr, V_ptr, D_ptr, attn_mask_ptr,
            stride_len, stride_embed,
            stride_mask_q, stride_mask_kv,
            SEQ_LEN, HEAD_DIM, 
            BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2, 
            start_row, start_col, num_steps, 
            ln2, MASK=False,
            DTYPE_FLAG=DTYPE_FLAG,
            USE_CUSTOM_MASK=USE_CUSTOM_MASK,
            mask_offset_base=mask_offset_base,
        )

    ### Scale our grads with the same factor ###
    dQ_block *= softmax_scale * rln2

    tl.store(dQ_ptr + Q_offsets, dQ_block, mask=mask_row[:, None])

def fused_sdpa_forward(Q, K, V, 
                       attn_mask=None,
                       causal=False, 
                       softmax_scale=None, 
                       use_dlpack=True):
    
    HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
    HEAD_DIM_V = V.shape[-1]
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert Q.dtype == K.dtype and K.dtype == V.dtype, "Expect all Q,K,V Tensors to have the same data type"

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
        M = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, dtype=torch.float32, device=Q.device)
        grid = lambda args: (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

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
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM_Q,
            ATTN_MODE=1 if causal else 0,
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
  
        ### Check for custom mask ###
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
        grid = lambda args: (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)

        # M is the logsumexp for the backward pass, one for each query
        # Make sure to create it on the right device as we are not using empty_like
        with cp.cuda.Device(Q.device.id):
            M = cp.empty(
                (BATCH_SIZE, NUM_HEADS, SEQ_LEN), dtype=cp.float32
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
            SEQ_LEN=SEQ_LEN,
            HEAD_DIM=HEAD_DIM_Q,
            ATTN_MODE=1 if causal else 0,
            DTYPE_FLAG=0 if Q.dtype == cp.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

    return Q, K, V, O, M

def fused_sdpa_backward(dO, 
                        Q, K, V, 
                        O, M, 
                        attn_mask=None,
                        causal=False,
                        softmax_scale=None,
                        use_dlpack=True):

    HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
    HEAD_DIM_V = V.shape[-1]
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert Q.dtype == K.dtype and K.dtype == V.dtype and V.dtype == O.dtype, "Expect all Q,K,V,O Tensors to have the same data type"

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
        D = torch.empty(BATCH_SIZE, NUM_HEADS, SEQ_LEN, dtype=torch.float32, device=Q.device)
    
        preprocess_grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE"]), BATCH_SIZE * NUM_HEADS)

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
            SEQ_LEN=SEQ_LEN,
            EMBED_DIM=HEAD_DIM,
            DTYPE_FLAG=0 if dO.dtype == torch.float32 else 1
        )

        grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_MACRO"]), BATCH_SIZE * NUM_HEADS)
        _attn_bwd[grid](
            Q_ptr=Q, 
            K_ptr=K, 
            V_ptr=V, 
            dO_ptr=dO, 
            dQ_ptr=dQ, 
            dK_ptr=dK, 
            dV_ptr=dV, 
            M_ptr=M, 
            D_ptr=D, 
            attn_mask_ptr=attn_mask,
            softmax_scale=softmax_scale, 
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_len=Q.stride(2),
            stride_embed=Q.stride(3), 
            stride_mask_batch=attn_mask.stride(0) if use_custom_mask else 0,
            stride_mask_head=attn_mask.stride(1) if use_custom_mask else 0,
            stride_mask_q=attn_mask.stride(2) if use_custom_mask else 0,
            stride_mask_kv=attn_mask.stride(3) if use_custom_mask else 0,
            NUM_HEADS=NUM_HEADS, 
            SEQ_LEN=SEQ_LEN, 
            HEAD_DIM=HEAD_DIM, 
            CAUSAL=1 if causal else 0, 
            DTYPE_FLAG=0 if Q.dtype == torch.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

        # Convert back to CuPy if needed
        dQ = cp.from_dlpack(dQ)
        dK = cp.from_dlpack(dK)
        dV = cp.from_dlpack(dV)

    else:  

        ### Check if we have Attention Mask ###
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
        preprocess_grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE"]), BATCH_SIZE * NUM_HEADS)

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
            SEQ_LEN=SEQ_LEN,
            EMBED_DIM=HEAD_DIM,
            DTYPE_FLAG=0 if dO.dtype == cp.float32 else 1
        )


        grid = lambda meta: (triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_MACRO"]), BATCH_SIZE * NUM_HEADS)
        _attn_bwd[grid](
            Q_ptr=Q.data.ptr, 
            K_ptr=K.data.ptr, 
            V_ptr=V.data.ptr, 
            dO_ptr=dO.data.ptr, 
            dQ_ptr=dQ.data.ptr, 
            dK_ptr=dK.data.ptr, 
            dV_ptr=dV.data.ptr, 
            M_ptr=M.data.ptr, 
            D_ptr=D.data.ptr, 
            attn_mask_ptr=attn_mask.data.ptr,
            softmax_scale=softmax_scale, 
            stride_batch=Q.strides[0] // Q.itemsize,
            stride_head=Q.strides[1] // Q.itemsize,
            stride_len=Q.strides[2] // Q.itemsize,
            stride_embed=Q.strides[3] // Q.itemsize, 
            stride_mask_batch=attn_mask.strides[0] // attn_mask.itemsize if use_custom_mask else 0,
            stride_mask_head=attn_mask.strides[1] // attn_mask.itemsize if use_custom_mask else 0,
            stride_mask_q=attn_mask.strides[2] // attn_mask.itemsize if use_custom_mask else 0,
            stride_mask_kv=attn_mask.strides[3] // attn_mask.itemsize if use_custom_mask else 0,
            NUM_HEADS=NUM_HEADS, 
            SEQ_LEN=SEQ_LEN, 
            HEAD_DIM=HEAD_DIM, 
            CAUSAL=1 if causal else 0, 
            DTYPE_FLAG=0 if Q.dtype == cp.float32 else 1,
            USE_CUSTOM_MASK=use_custom_mask
        )

    return dQ, dK, dV

if __name__ == "__main__":

    q = torch.randn((2,2,64,128), device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn((2,2,64,128), device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn((2,2,64,128), device="cuda", dtype=torch.float16, requires_grad=True)
    attn_mask = torch.ones((2,2,64,64)).bool().to("cuda")
    attn_mask[0, :, :, -20:] = False
    attn_mask[1, :, :, -4:] = False

    attn_mask_cp = cp.array(attn_mask.detach().cpu().numpy())
    o_grad = torch.randn_like(q)
    o_grad_cp = cp.array(o_grad.detach().cpu().numpy())
    out = torch.nn.functional.scaled_dot_product_attention(q,k,v, attn_mask=attn_mask, is_causal=True)
    out.backward(o_grad)
    q_grad_ref = cp.array(q.grad.detach().cpu().numpy())
    k_grad_ref = cp.array(k.grad.detach().cpu().numpy())
    v_grad_ref = cp.array(v.grad.detach().cpu().numpy())
    out_ref = cp.array(out.detach().cpu().numpy())


    q_cp = cp.array(q.detach().cpu().numpy())
    k_cp = cp.array(k.detach().cpu().numpy())
    v_cp = cp.array(v.detach().cpu().numpy())
    q_cp, k_cp, v_cp, O, M = fused_sdpa_forward(q_cp,k_cp,v_cp, attn_mask_cp, causal=True)
    dQ_cp, dK_cp, dV_cp = fused_sdpa_backward(o_grad_cp, q_cp, k_cp, v_cp, O, M, attn_mask_cp, causal=True)
    print(cp.max(cp.abs(O-out_ref)))

    print(cp.max(cp.abs(dQ_cp-q_grad_ref)))
    print(cp.max(cp.abs(dK_cp-k_grad_ref)))
    print(cp.max(cp.abs(dV_cp-v_grad_ref)))
