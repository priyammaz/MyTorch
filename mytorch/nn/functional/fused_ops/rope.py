# """
# Awesome implementation from Liger Kernels!
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py
# """
# import torch
# import triton
# import triton.language as tl

# @triton.jit
# def triton_rope_forward_backward(
#     q_ptr,  # B x L x H x D
#     q_row_stride, 
#     k_ptr, # B x L x H x D
#     k_row_stride, 
#     cos, # (1, seq_len, head_dim) or (b, seq_len, head_dim)
#     cos_row_stride, 
#     sin, # (1, seq_len, head_dim) or (b, seq_len, head_dim)
#     sin_row_stride, 
#     seq_len, 
#     cos_batch_size, 
#     num_q_heads: tl.constexpr, 
#     num_kv_heads: tl.constexpr, 
#     head_dim: tl.constexpr, 
#     pad_n_q_head: tl.constexpr, # num heads of our queries + extra to be power of 2 (to be masked out)
#     pad_n_kv_head: tl.constexpr, # num heads of our keys + extra to be power of 2 (to be masked out)
#     pad_head_dim: tl.constexpr, # head dim of our sin/cos + extra to be power of 2 (to be masked out)
#     BLOCK_SIZE, 
#     BACKWARD_PASS
# ):
    
#     ### Each pid will process a single timestep per batch
#     pid = tl.program_id(0)

#     ### Advance our pointers to the correct timestep ###
#     q_ptr = q_ptr + pid * q_row_stride
#     k_ptr = k_ptr + pid * k_row_stride

#     ### There are a total of batch_size * seq_len tokens we need to process
#     ### The pid flattened these two dims together, but lets get the exact batch
#     ### we are currently on
#     batch_idx = pid // seq_len

#     ### Then we need which timestep we are on inside this batch, which we can get as
#     row_idx = pid % seq_len

#     ### Now we load our sin and cos values. Both sin/cos are (1,seq_len,head_dim) or (b, seq_len, head_dim)
#     ### but remember this from our construction of the frequencies?
#     ### freqs = mytorch.concatenate([freqs, freqs], dim=-1)  # [max_pos, head_dim]
#     ### We have a duplicate, the second half of the embed dimension is the same as the 
#     ### first half of the embed dimension. So because the right half is just a clone of
#     ### the left half, it is wasteful to load all of it so we just grab the left half!

#     ### To index correctly though, we need to check our potential batch dim in the cos/sin as it could be 1 
#     ### or it can just be a batch dimension. So we can handle it like so:
    
#     cos = cos + tl.where(
#         cos_batch_size == 1,
#         row_idx * cos_row_stride,  # <- if batch size is 1, then we just need to advance to the correct timestep
#         (batch_idx * seq_len * cos_row_stride) + row_idx * cos_row_stride # <- if batch isnt 1 then we advance to 
#                                                                           # correct batch and then the correct timestep
#     )

#     sin = sin + tl.where(
#         cos_batch_size == 1,
#         row_idx * sin_row_stride,  # <- if batch size is 1, then we just need to advance to the correct timestep
#         (batch_idx * seq_len * sin_row_stride) + row_idx * sin_row_stride # <- if batch isnt 1 then we advance to 
#                                                                           # correct batch and then the correct timestep
#     )

#     ### Get the offset for the first half of the head dim (as our second half is again just a copy!)
#     cos_offset = tl.arange(0, pad_head_dim // 2)
#     cos_mask = cos_offset < head_dim//2
#     cos_row = tl.load(cos + cos_offset, mask=cos_mask, other=0.)
#     sin_row = tl.load(sin + cos_offset, mask=cos_mask, other=0.)

#     ### Now we can load our queries and keys. Remember from our implementation:

#     # def rotate_half(x):
#     #     """
#     #     [a1, a2, a3, a4] -> [-a3, -a4, a1, a2]
#     #     """
#     #     x1 = x[..., :x.shape[-1]//2]
#     #     x2 = x[..., x.shape[-1]//2:]
#     #     return mytorch.concatenate([-x2,x1], dim=-1)

#     ### So all we have to do first is grab the first half and second half of our queries and keys along the embed dim 
#     ### But we do this across ALL the heads of attention, so we have here a grid in the shape of:
#     ### [num_heads x 1] + [1 x head_dim//2] -> gives a matrix of indexes [num_heads x head_dim//2] to the first half
#     ### of the embed dim across all heads of attention
#     first_half_q_offsets = tl.arange(0, pad_n_q_head)[:, None] * head_dim + tl.arange(0, pad_head_dim // 2)[None, :]
#     first_half_k_offsets = tl.arange(0, pad_n_kv_head)[:, None] * head_dim + tl.arange(0, pad_head_dim // 2)[None, :]

#     ### Now as everything is padded we have our masks as well
#     first_q_mask = (tl.arange(0, pad_n_q_head)[:, None] < num_q_heads) & (tl.arange(0, pad_head_dim // 2)[None, :] < head_dim // 2)
#     first_k_mask = (tl.arange(0, pad_n_kv_head)[:, None] < num_kv_heads) & (tl.arange(0, pad_head_dim // 2)[None, :] < head_dim // 2)

#     ### And finally we load our data ###
#     q_tile_left = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
#     k_tile_left = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

#     ### In the same way we load the right half now
#     ### first we advance our pointers to the second half
#     second_half_q_offsets = first_half_q_offsets + (head_dim // 2)
#     second_half_k_offsets = first_half_k_offsets + (head_dim // 2)
    
#     ### The masks are the same
#     second_q_mask = first_q_mask
#     second_k_mask = first_k_mask

#     ### Load the right half 
#     q_tile_right = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
#     k_tile_right = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

#     ### Now the tricky part, the forward and backward pass are technically the same thing. Lets start with the
#     ### forward pass first!

#     if not BACKWARD_PASS:

#         ### In the forward pass what we want to do is the followiung:
#         ### q_embed = (q * cos) + (rotate_half(q) * sin)
#         ### k_embed = (k * cos) + (rotate_half(k) * sin)
#         ### Lets just look at the first one:
#         ### q_embed = (q * cos) + (rotate_half(q) * sin)

#         ### Now we dont have q, what we have is the left and right sides of q, so 
#         ### lets use this bad notation to make it easier to understand

#         ### q_embed = [q_left | q_right] * cos + (rotate_half([q_left | q_right]) * sin)
#         ### lets perform the rotate_half operation:
#         ### q_embed = [q_left | q_right] * cos + [-q_right | q_left] * sin
#         ### So now technically we can do the left and right ops separately and then 
#         ### Just save them in the correct place, so we can write:
#         ### [q_embed_left | q_embed_right] = [q_left | q_right] * cos + [-q_right | q_left] * sin
#         ### where:
#         ### q_embed_left = q_left * cos - q_right * sin
#         ### q_embed_right = q_right * cos + q_left * sin
#         ### also notice that we dont have a left/right in our sin/cos. We could but the left/right 
#         ### of our sin/cos are the same (copies of each other) so we just need it once!

#         q_embed_left = q_tile_left * cos_row - q_tile_right * sin_row
#         q_embed_right = q_tile_right * cos_row + q_tile_left * sin_row

#         ### We can then directly store these value in the existing memory locations of our queries and keys
#         ### just to save us from doing an additional memory allocation! The backward pass doenst need any of 
#         ### this information so we can just replace it!
#         tl.store(q_ptr + first_half_q_offsets, q_embed_left, mask=first_q_mask)
#         tl.store(q_ptr + second_half_q_offsets, q_embed_right, mask=second_q_mask)

#         ### Same thing applies to K 
#         k_embed_left = k_tile_left * cos_row - k_tile_right * sin_row
#         k_embed_right = k_tile_right * cos_row + k_tile_left * sin_row
#         tl.store(k_ptr + first_half_k_offsets, k_embed_left, mask=first_k_mask)
#         tl.store(k_ptr + second_half_k_offsets, k_embed_right, mask=second_k_mask)

#     else:

#         ### Now for the backward pass, nothing really changes but a sign flip. Right now
#         ### what we have done in the forward pass is effectively:
#         ### [y_1]   [cos(θ)  -sin(θ)] [x_1]
#         ### [y_2] = [sin(θ)   cos(θ)] [x_1]
        
#         ### For simplicity lets just say Y = R(θ)@X
#         ### During backprop we will recieve our grads w.r.t to the outputs, so 
#         ### we will have have something like:
#         ### [dy_1]
#         ### [dy_2]
#         ### And like normal in a matmul we need to compute the derivative w.r.t x
#         ### Well if Y = R(θ)@X, then dL/dX = R(θ).T @ dL/dY

#         ### And if R(θ) is just 
#         ### [cos(θ)  -sin(θ)]
#         ### [sin(θ)   cos(θ)]

#         ### then R(θ).T is just:
#         ### [cos(θ)    sin(θ)]
#         ### [-sin(θ)   cos(θ)]

#         ### So really all we need to do is flip the sign of our sin and we are good to go!
#         ### In the backward pass, our q_ptr/k_ptr will now be pointing to dQ and dK, our 
#         ### upstream grads from the rotary embeddings. So that is basically our Y in this case. 
#         ### We are doing the same rotary operation as before, just with a transposed rotation matrix

#         q_embed_left = q_tile_left * cos_row + q_tile_right * sin_row
#         q_embed_right = q_tile_right * cos_row - q_tile_left * sin_row
#         tl.store(q_ptr + first_half_q_offsets, q_embed_left, mask=first_q_mask)
#         tl.store(q_ptr + second_half_q_offsets, q_embed_right, mask=second_q_mask)

#         ### Same thing applies to K 
#         k_embed_left = k_tile_left * cos_row + k_tile_right * sin_row
#         k_embed_right = k_tile_right * cos_row - k_tile_left * sin_row
#         tl.store(k_ptr + first_half_k_offsets, k_embed_left, mask=first_k_mask)
#         tl.store(k_ptr + second_half_k_offsets, k_embed_right, mask=second_k_mask)

# def fused_rope_forward(q, k, cos, sin, use_dlpack=True):
    
#     """
#     The forward method expects q,k to be [B x L x H x D]
#     and cos/sin to be [1 x L x D] or [B x L x D]
#     """

#     if use_dlpack:
        
#         q = torch.utils.dlpack.from_dlpack(q)
#         k = torch.utils.dlpack.from_dlpack(k)
#         cos = torch.utils.dlpack.from_dlpack(cos)
#         sin = torch.utils.dlpack.from_dlpack(sin)

#         batch_size, seq_len, n_q_heads, head_dim = q.shape
#         n_kv_heads = k.shape[2]

#         pad_head_dim = triton.next_power_of_2(head_dim)
#         pad_num_q_heads = triton.next_power_of_2(n_q_heads)
#         pad_num_kv_heads = triton.next_power_of_2(n_kv_heads)
#         BLOCK_SIZE = max(pad_num_q_heads, pad_num_kv_heads)

#         ### Total number of tokens to process 
#         n_row = batch_size * seq_len

#         ### Quick contiguous check ###
#         if not q.is_contiguous():
#             q = q.contiguous()
#         if not k.is_contiguous():
#             k = k.contiguous()
#         if not cos.is_contiguous():
#             cos = cos.contiguous()
#         if not sin.is_contiguous():
#             sin = sin.contiguous()

#         triton_rope_forward_backward[(n_row, )](
#             q, 
#             q.stride(1), 
#             k, 
#             k.stride(1), 
#             cos, 
#             cos.stride(-2),
#             sin, 
#             sin.stride(-2),
#             seq_len, 
#             cos.shape[0], 
#             n_q_heads, 
#             n_kv_heads, 
#             head_dim, 
#             pad_num_q_heads, 
#             pad_num_kv_heads, 
#             pad_head_dim,
#             BLOCK_SIZE=BLOCK_SIZE, 
#             BACKWARD_PASS=False
#         )

#     return q, k, cos, sin

# def fused_rope_backward(dq, dk, cos, sin, use_dlpack=True):

#     """
#     The backward method expects dq,dk to be [B x L x H x D]
#     and cos/sin to be [1 x L x D] or [B x L x D]
#     """

#     if use_dlpack:
        
#         dq = torch.utils.dlpack.from_dlpack(dq)
#         dk = torch.utils.dlpack.from_dlpack(dk)
#         cos = torch.utils.dlpack.from_dlpack(cos)
#         sin = torch.utils.dlpack.from_dlpack(sin)

#         batch_size, seq_len, n_q_heads, head_dim = dq.shape
#         n_kv_heads = dk.shape[2]

#         pad_head_dim = triton.next_power_of_2(head_dim)
#         pad_num_q_heads = triton.next_power_of_2(n_q_heads)
#         pad_num_kv_heads = triton.next_power_of_2(n_kv_heads)
#         BLOCK_SIZE = max(pad_num_q_heads, pad_num_kv_heads)

#         ### Total number of tokens to process 
#         n_row = batch_size * seq_len

#         ### Quick contiguous check ###
#         if not dq.is_contiguous():
#             dq = dq.contiguous()
#         if not dk.is_contiguous():
#             dk = dk.contiguous()

#         triton_rope_forward_backward[(n_row, )](
#             dq, 
#             dq.stride(1), 
#             dk, 
#             dk.stride(1), 
#             cos, 
#             cos.stride(-2),
#             sin, 
#             sin.stride(-2),
#             seq_len, 
#             cos.shape[0], 
#             n_q_heads, 
#             n_kv_heads, 
#             head_dim, 
#             pad_num_q_heads, 
#             pad_num_kv_heads, 
#             pad_head_dim,
#             BLOCK_SIZE=BLOCK_SIZE, 
#             BACKWARD_PASS=False
#         )

#     return dq, dk

"""
Awesome implementation from Liger Kernels!
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py
"""
import cupy as cp
import torch
import triton
import triton.language as tl

@triton.jit
def triton_rope_forward_backward(
    x_ptr,  # B x L x H x D
    x_row_stride, 
    cos, # (1, seq_len, head_dim) or (b, seq_len, head_dim)
    cos_row_stride, 
    sin, # (1, seq_len, head_dim) or (b, seq_len, head_dim)
    sin_row_stride, 
    seq_len, 
    cos_batch_size, 
    num_x_heads: tl.constexpr, 
    head_dim: tl.constexpr, 
    pad_n_x_head: tl.constexpr, # num heads of our queries + extra to be power of 2 (to be masked out)
    pad_head_dim: tl.constexpr, # head dim of our sin/cos + extra to be power of 2 (to be masked out)
    BACKWARD_PASS
):
    
    ### Each pid will process a single timestep per batch
    pid = tl.program_id(0)

    ### Advance our pointers to the correct timestep ###
    x_ptr = x_ptr + pid * x_row_stride

    ### There are a total of batch_size * seq_len tokens we need to process
    ### The pid flattened these two dims together, but lets get the exact batch
    ### we are currently on
    batch_idx = pid // seq_len

    ### Then we need which timestep we are on inside this batch, which we can get as
    row_idx = pid % seq_len

    ### Now we load our sin and cos values. Both sin/cos are (1,seq_len,head_dim) or (b, seq_len, head_dim)
    ### but remember this from our construction of the frequencies?
    ### freqs = mytorch.concatenate([freqs, freqs], dim=-1)  # [max_pos, head_dim]
    ### We have a duplicate, the second half of the embed dimension is the same as the 
    ### first half of the embed dimension. So because the right half is just a clone of
    ### the left half, it is wasteful to load all of it so we just grab the left half!

    ### To index correctly though, we need to check our potential batch dim in the cos/sin as it could be 1 
    ### or it can just be a batch dimension. So we can handle it like so:
    
    cos = cos + tl.where(
        cos_batch_size == 1,
        row_idx * cos_row_stride,  # <- if batch size is 1, then we just need to advance to the correct timestep
        (batch_idx * seq_len * cos_row_stride) + row_idx * cos_row_stride # <- if batch isnt 1 then we advance to 
                                                                          # correct batch and then the correct timestep
    )

    sin = sin + tl.where(
        cos_batch_size == 1,
        row_idx * sin_row_stride,  # <- if batch size is 1, then we just need to advance to the correct timestep
        (batch_idx * seq_len * sin_row_stride) + row_idx * sin_row_stride # <- if batch isnt 1 then we advance to 
                                                                          # correct batch and then the correct timestep
    )

    ### Get the offset for the first half of the head dim (as our second half is again just a copy!)
    cos_offset = tl.arange(0, pad_head_dim // 2)
    cos_mask = cos_offset < head_dim//2
    cos_row = tl.load(cos + cos_offset, mask=cos_mask, other=0.)
    sin_row = tl.load(sin + cos_offset, mask=cos_mask, other=0.)

    ### Now we can load our queries and keys. Remember from our implementation:

    # def rotate_half(x):
    #     """
    #     [a1, a2, a3, a4] -> [-a3, -a4, a1, a2]
    #     """
    #     x1 = x[..., :x.shape[-1]//2]
    #     x2 = x[..., x.shape[-1]//2:]
    #     return mytorch.concatenate([-x2,x1], dim=-1)

    ### So all we have to do first is grab the first half and second half of our queries and keys along the embed dim 
    ### But we do this across ALL the heads of attention, so we have here a grid in the shape of:
    ### [num_heads x 1] + [1 x head_dim//2] -> gives a matrix of indexes [num_heads x head_dim//2] to the first half
    ### of the embed dim across all heads of attention
    first_half_x_offsets = tl.arange(0, pad_n_x_head)[:, None] * head_dim + tl.arange(0, pad_head_dim // 2)[None, :]

    ### Now as everything is padded we have our masks as well
    first_x_mask = (tl.arange(0, pad_n_x_head)[:, None] < num_x_heads) & (tl.arange(0, pad_head_dim // 2)[None, :] < head_dim // 2)

    ### And finally we load our data ###
    x_tile_left = tl.load(x_ptr + first_half_x_offsets, mask=first_x_mask, other=0).to(sin_row.dtype)

    ### In the same way we load the right half now
    ### first we advance our pointers to the second half
    second_half_x_offsets = first_half_x_offsets + (head_dim // 2)
    
    ### The masks are the same
    second_x_mask = first_x_mask

    ### Load the right half 
    x_tile_right = tl.load(x_ptr + second_half_x_offsets, mask=second_x_mask, other=0).to(sin_row.dtype)

    ### Now the tricky part, the forward and backward pass are technically the same thing. Lets start with the
    ### forward pass first!

    if not BACKWARD_PASS:

        ### In the forward pass what we want to do is the followiung:
        ### q_embed = (q * cos) + (rotate_half(q) * sin)
        ### k_embed = (k * cos) + (rotate_half(k) * sin)
        ### Lets just look at the first one:
        ### q_embed = (q * cos) + (rotate_half(q) * sin)

        ### Now we dont have q, what we have is the left and right sides of q, so 
        ### lets use this bad notation to make it easier to understand

        ### q_embed = [q_left | q_right] * cos + (rotate_half([q_left | q_right]) * sin)
        ### lets perform the rotate_half operation:
        ### q_embed = [q_left | q_right] * cos + [-q_right | q_left] * sin
        ### So now technically we can do the left and right ops separately and then 
        ### Just save them in the correct place, so we can write:
        ### [q_embed_left | q_embed_right] = [q_left | q_right] * cos + [-q_right | q_left] * sin
        ### where:
        ### q_embed_left = q_left * cos - q_right * sin
        ### q_embed_right = q_right * cos + q_left * sin
        ### also notice that we dont have a left/right in our sin/cos. We could but the left/right 
        ### of our sin/cos are the same (copies of each other) so we just need it once!

        x_embed_left = x_tile_left * cos_row - x_tile_right * sin_row
        x_embed_right = x_tile_right * cos_row + x_tile_left * sin_row

        ### We can then directly store these value in the existing memory locations of our queries and keys
        ### just to save us from doing an additional memory allocation! The backward pass doenst need any of 
        ### this information so we can just replace it!
        tl.store(x_ptr + first_half_x_offsets, x_embed_left, mask=first_x_mask)
        tl.store(x_ptr + second_half_x_offsets, x_embed_right, mask=second_x_mask)

    else:

        ### Now for the backward pass, nothing really changes but a sign flip. Right now
        ### what we have done in the forward pass is effectively:
        ### [y_1]   [cos(θ)  -sin(θ)] [x_1]
        ### [y_2] = [sin(θ)   cos(θ)] [x_1]
        
        ### For simplicity lets just say Y = R(θ)@X
        ### During backprop we will recieve our grads w.r.t to the outputs, so 
        ### we will have have something like:
        ### [dy_1]
        ### [dy_2]
        ### And like normal in a matmul we need to compute the derivative w.r.t x
        ### Well if Y = R(θ)@X, then dL/dX = R(θ).T @ dL/dY

        ### And if R(θ) is just 
        ### [cos(θ)  -sin(θ)]
        ### [sin(θ)   cos(θ)]

        ### then R(θ).T is just:
        ### [cos(θ)    sin(θ)]
        ### [-sin(θ)   cos(θ)]

        ### So really all we need to do is flip the sign of our sin and we are good to go!
        ### In the backward pass, our q_ptr/k_ptr will now be pointing to dQ and dK, our 
        ### upstream grads from the rotary embeddings. So that is basically our Y in this case. 
        ### We are doing the same rotary operation as before, just with a transposed rotation matrix

        x_embed_left = x_tile_left * cos_row + x_tile_right * sin_row
        x_embed_right = x_tile_right * cos_row - x_tile_left * sin_row
        tl.store(x_ptr + first_half_x_offsets, x_embed_left, mask=first_x_mask)
        tl.store(x_ptr + second_half_x_offsets, x_embed_right, mask=second_x_mask)

def fused_rope_forward(x, cos, sin, use_dlpack=True):
    
    """
    The forward method expects x to be [B x L x H x D]
    and cos/sin to be [1 x L x D] or [B x L x D]
    """

    if use_dlpack:
        
        x = torch.utils.dlpack.from_dlpack(x)
        cos = torch.utils.dlpack.from_dlpack(cos)
        sin = torch.utils.dlpack.from_dlpack(sin)

        batch_size, seq_len, num_heads, head_dim = x.shape

        pad_head_dim = triton.next_power_of_2(head_dim)
        pad_num_heads = triton.next_power_of_2(num_heads)

        ### Total number of tokens to process 
        n_row = batch_size * seq_len

        ### Quick contiguous check ###
        if not x.is_contiguous():
            x = x.contiguous()
        if not cos.is_contiguous():
            cos = cos.contiguous()
        if not sin.is_contiguous():
            sin = sin.contiguous()

        triton_rope_forward_backward[(n_row, )](
            x, 
            x.stride(1),  
            cos, 
            cos.stride(-2),
            sin, 
            sin.stride(-2),
            seq_len, 
            cos.shape[0], 
            num_heads, 
            head_dim, 
            pad_num_heads, 
            pad_head_dim,
            BACKWARD_PASS=False
        )

        return cp.from_dlpack(x), cp.from_dlpack(cos), cp.from_dlpack(sin)
    
    else:

        batch_size, seq_len, num_heads, head_dim = x.shape

        pad_head_dim = triton.next_power_of_2(head_dim)
        pad_num_heads = triton.next_power_of_2(num_heads)

        ### Total number of tokens to process 
        n_row = batch_size * seq_len

        ### Quick contiguous check ###
        if not x.flags['C_CONTIGUOUS']:
            x = cp.ascontiguousarray(x)
        if not cos.flags['C_CONTIGUOUS']:
            cos = cp.ascontiguousarray(cos)
        if not sin.flags['C_CONTIGUOUS']:
            sin = cp.ascontiguousarray(sin)

        triton_rope_forward_backward[(n_row, )](
            x.data.ptr,
            x.strides[1] // x.itemsize,  
            cos.data.ptr,
            cos.strides[-2] // cos.itemsize,
            sin.data.ptr,
            sin.strides[-2] // sin.itemsize,
            seq_len, 
            cos.shape[0], 
            num_heads, 
            head_dim, 
            pad_num_heads, 
            pad_head_dim,
            BACKWARD_PASS=False
        )

        return x, cos, sin

def fused_rope_backward(dx, cos, sin, use_dlpack=True):

    """
    The backward method expects dq,dk to be [B x L x H x D]
    and cos/sin to be [1 x L x D] or [B x L x D]
    """

    if use_dlpack:
        
        dx = torch.utils.dlpack.from_dlpack(dx)
        cos = torch.utils.dlpack.from_dlpack(cos)
        sin = torch.utils.dlpack.from_dlpack(sin)

        batch_size, seq_len, num_heads, head_dim = dx.shape

        pad_head_dim = triton.next_power_of_2(head_dim)
        pad_num_heads = triton.next_power_of_2(num_heads)

        ### Total number of tokens to process 
        n_row = batch_size * seq_len

        ### Quick contiguous check ###
        if not dx.is_contiguous():
            dx = dx.contiguous()

        triton_rope_forward_backward[(n_row, )](
            dx, 
            dx.stride(1),  
            cos, 
            cos.stride(-2),
            sin, 
            sin.stride(-2),
            seq_len, 
            cos.shape[0], 
            num_heads, 
            head_dim, 
            pad_num_heads, 
            pad_head_dim,
            BACKWARD_PASS=True
        )


        return cp.from_dlpack(dx)   

    else:
     
        batch_size, seq_len, num_heads, head_dim = dx.shape

        pad_head_dim = triton.next_power_of_2(head_dim)
        pad_num_heads = triton.next_power_of_2(num_heads)

        ### Total number of tokens to process 
        n_row = batch_size * seq_len

        ### Quick contiguous check ###
        if not dx.flags['C_CONTIGUOUS']:
            dx = cp.ascontiguousarray(dx)

        triton_rope_forward_backward[(n_row, )](
            dx.data.ptr,
            dx.strides[1] // dx.itemsize,  
            cos.data.ptr,
            cos.strides[-2] // cos.itemsize,
            sin.data.ptr,
            sin.strides[-2] // sin.itemsize,
            seq_len, 
            cos.shape[0], 
            num_heads, 
            head_dim, 
            pad_num_heads, 
            pad_head_dim,
            BACKWARD_PASS=True
        )

        return dx