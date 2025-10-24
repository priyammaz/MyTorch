"""
Inspired by Unsloth: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py

Also, CrossEntropy is typically unsafe at float16, so these modules expect float32 inputs always!
"""
import torch
import cupy as cp
import triton
import triton.language as tl
from .utils import calc_num_warps
from .flags import DLPACK_DISABLE

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def cross_entropy_forward(
    logits_ptr,            # (N, NUM_CLASSES) matrix
    logits_row_stride,     
    loss_ptr,              # (N,) dim vector to hold the loss
    logsumexp_ptr,         
    labels_ptr, 
    NUM_CLASSES: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Our Logits will be some (N x NUM_CLASSES)
    Our Targets will be some (N, ) where each value in the target 
    is between 0 and NUM_CLASSES-1

    Cross Entropy Formula w. Softmax together was just:

    CE = log(sum(e^x)) = x_{correct}

    And we know the index of correct, its just our label the cooresponding labels. 
    So lets just write a kernel that processes one row at a time. We will grab the 
    NUM_CLASSES length vector of logits and the single label

    The difficulty is that to do this operation, we need to compute the max over the 
    ENTIRE row, and compute a sum over the ENTIRE row. This means, even for large vocabs
    we need to load the full row to memory. This is why unsloth has a more efficient
    chunked cross entropy here:
    https://github.com/unslothai/unsloth/blob/67ea5e422d65b9afa15748e56d2b1495e5ac06e5/unsloth/kernels/cross_entropy_loss.py#L108
    
    Where they chunk over the NUM_CLASSES dimension. We will keep it simple!
    """

    row_idx = tl.program_id(0)

    ### Cast Pointers ###
    logits_ptr = tl.cast(logits_ptr, tl.pointer_type(tl.float32))
    loss_ptr = tl.cast(loss_ptr, tl.pointer_type(tl.float32))
    logsumexp_ptr = tl.cast(logsumexp_ptr, tl.pointer_type(tl.float32))
    labels_ptr = tl.cast(labels_ptr, tl.pointer_type(tl.int64))
    
    ### Get starting pointer of the row we want ###
    logits_ptr += row_idx * logits_row_stride

    ### Labels are just a vector so move over to the N index we want ###
    labels_ptr += row_idx

    ### logsumexp_ptr and loss_ptr are also just vectors to store data for each n in N ###
    ### So just get the index to that as well###
    loss_ptr += row_idx
    logsumexp_ptr += row_idx

    ### Load the logits ###
    col_offsets = tl.arange(0, BLOCK_SIZE)

    ### Incase our block spills over the edge ###
    mask = col_offsets < NUM_CLASSES

    ### Its pretty common to compute loss in 32 bit precision just incase ###
    ### We dont want any over/underflows at this stage ###
    ### also we are about to do a max as well as an exp, so -inf doesnt effect ###
    ### max and exponentiating it set it to zero ! ###
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))#.to(tl.float32)
    
    ### Compute the logsumexp (stable so subtract the max inside and add it back outside) ###
    c = tl.max(logits, axis=0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), axis=0))

    ### Load Label Index ###
    label_idx = tl.load(labels_ptr)#.to(tl.int64)

    ### -100 is our default ignore index ###
    if label_idx != -100:

        ### Load logit at the label idx ###
        x_label = tl.load(logits_ptr + label_idx)#.to(tl.float32)
        loss = logsumexp - x_label

    else:
        loss = 0.0

    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)

@triton.heuristics({"num_warps": lambda args: calc_num_warps(args["BLOCK_SIZE"])})
@triton.jit
def cross_entropy_backward(
    logits_ptr,            # Original logits for loading (N x NUM_CLASSES)
    logits_row_stride,     # Element stride for rows
    grad_ptr,              # Gradient output buffer (N x NUM_CLASSES)
    logsumexp_ptr,         # (N,) precomputed logsumexp
    labels_ptr,            # (N,) labels (int64)
    NUM_CLASSES: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):  
    """
    The backward pass formula is Pj - yj, where yj is 1 for the correct
    label and 0 for the incorrect one. 

    Now to do this, all we need is the probability of a specific index. Well, we computed
    earlier our logsumexp. So to get the probability of any step all we need to do is 
    compute our softmax like so:
    
    Compute Softmax (e^{x - log(sum(x)) = e^x / e^{log(sum(x)) = e^x / sum(e^x)}}

    Notice here, we juse our logsumexp from ealier. We dont need to recompute the entire
    row again! So lets save some memory. We can tile our operation into small chunks of the
    matrix and use a 2d setup of blocks to be more efficient! Each threadblock doesnt need to
    load the entire row anymore, we will cut each row into some chunks and allow for more
    parallelism!
    """

    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # Cast pointers
    logits_ptr = tl.cast(logits_ptr, tl.pointer_type(tl.float32))
    grad_ptr = tl.cast(grad_ptr, tl.pointer_type(tl.float32))
    logsumexp_ptr = tl.cast(logsumexp_ptr, tl.pointer_type(tl.float32))
    labels_ptr = tl.cast(labels_ptr, tl.pointer_type(tl.int64))
    
    # Advance to correct row
    logits_ptr += row_idx * logits_row_stride
    grad_ptr += row_idx * logits_row_stride  
    
    # Load scalars for this row
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)
    
    # Get column offsets for this block
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < NUM_CLASSES
    
    # Load logits
    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    
    # Compute softmax
    y = tl.exp(x - logsumexp)
    
    # Subtract 1 for correct label
    mask_correct_label = (col_offsets == label_idx)
    y = tl.where(mask_correct_label, y - 1.0, y)
    
    # Zero out ignored labels
    y = tl.where(label_idx != -100, y, 0.0)
    
    # Store to correct location
    tl.store(grad_ptr + col_offsets, y, mask=mask)

def fused_cross_entropy_forward(logits, labels, use_dlpack=True):
    N, C = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(C)

    if not DLPACK_DISABLE and use_dlpack:

        logits = torch.utils.dlpack.from_dlpack(logits)
        labels = torch.utils.dlpack.from_dlpack(labels)
    
        N, C = logits.shape
        loss = torch.zeros(N, dtype=torch.float32, device=logits.device)
        logsumexp = torch.zeros(N, dtype=torch.float32, device=logits.device)

        BLOCK_SIZE = triton.next_power_of_2(C)
        row_stride = logits.stride(0)

        grid = (N,)
        cross_entropy_forward[grid](
            logits, 
            row_stride, 
            loss, 
            logsumexp, 
            labels, 
            C, 
            BLOCK_SIZE
        )
        
        # Convert back to CuPy
        loss = cp.from_dlpack(loss)
        logsumexp = cp.from_dlpack(logsumexp)
        return loss, logsumexp
    
    else:

        loss = cp.zeros(N, dtype=cp.float32)
        logsumexp = cp.zeros(N, dtype=cp.float32)

        row_stride = logits.strides[0] // logits.itemsize

        grid = (N,)
        cross_entropy_forward[grid](
                logits.data.ptr, 
                row_stride, 
                loss.data.ptr, 
                logsumexp.data.ptr, 
                labels.data.ptr, 
                C, 
                BLOCK_SIZE
        )       
        
        return loss, logsumexp

def fused_cross_entropy_backward(
        logits, labels, logsumexp, BLOCK_SIZE=128, use_dlpack=True
):
    
    """
    Block size should definitely be tuned, but i just pick something
    reasonable here for now! We will chunk our NUM_CLASSES len vector
    into chunks of BLOCK_SIZE

    Also, this is the last thing in our model. There is no more
    ops after Cross Entropy. So our dloss is just a bunch of ones!
    We can create that in here manually as our "upstream" grad. We 
    already do this in our .backward() in our tensor, but that only
    returns a single 1
    """
    N, C = logits.shape

    if not DLPACK_DISABLE and use_dlpack:
        logits = torch.utils.dlpack.from_dlpack(logits)
        labels = torch.utils.dlpack.from_dlpack(labels)
        logsumexp = torch.utils.dlpack.from_dlpack(logsumexp)
        
        row_stride = logits.stride(0)
        grad = torch.zeros_like(logits, dtype=torch.float32)

        grid = (N, triton.cdiv(C, BLOCK_SIZE))
        cross_entropy_backward[grid](
            logits,
            row_stride,
            grad, 
            logsumexp, 
            labels, 
            C, 
            BLOCK_SIZE
        )

        return grad

    else:

        row_stride = logits.strides[0] // logits.itemsize
        grad = cp.zeros_like(logits , dtype=cp.float32)

        grid = (N, triton.cdiv(C, BLOCK_SIZE))
        cross_entropy_backward[grid](
            logits.data.ptr,
            row_stride,
            grad.data.ptr, 
            logsumexp.data.ptr, 
            labels.data.ptr, 
            C, 
            BLOCK_SIZE
        )

        return grad