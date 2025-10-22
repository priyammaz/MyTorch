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
        logits, labels, logsumexp, BLOCK_SIZE=128
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
    row_stride = logits.strides[0] // logits.itemsize
    
    grad = cp.zeros_like(logits, dtype=cp.float32)

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

# ### TESTING DLPACK ###
# import cupy as cp
# import torch
# from torch.utils.dlpack import from_dlpack
# def fused_cross_entropy_forward(logits, labels):
#     # Detect tensor type
#     is_torch = isinstance(logits, torch.Tensor)
#     is_cupy = not is_torch and hasattr(logits, '__cuda_array_interface__')
    
    # # Convert CuPy to PyTorch for better Triton performance
    # if is_cupy:
    #     logits = from_dlpack(logits)
    #     labels = from_dlpack(labels)
    
    # N, C = logits.shape
    # loss = torch.zeros(N, dtype=torch.float32, device=logits.device)
    # logsumexp = torch.zeros(N, dtype=torch.float32, device=logits.device)

    # BLOCK_SIZE = triton.next_power_of_2(C)
    # row_stride = logits.stride(0)

    # grid = (N,)
    # cross_entropy_forward[grid](
    #     logits, 
    #     row_stride, 
    #     loss, 
    #     logsumexp, 
    #     labels, 
    #     C, 
    #     BLOCK_SIZE
    # )
    
    # # Convert back to CuPy if needed
    # if is_cupy:
    #     import cupy as cp
    #     loss = cp.from_dlpack(loss)
    #     logsumexp = cp.from_dlpack(logsumexp)
    
#     return loss, logsumexp

# def fused_cross_entropy_backward(
#         logits, labels, logsumexp, BLOCK_SIZE=128
# ):
#     """
#     Block size should definitely be tuned, but i just pick something
#     reasonable here for now! We will chunk our NUM_CLASSES len vector
#     into chunks of BLOCK_SIZE

#     Also, this is the last thing in our model. There is no more
#     ops after Cross Entropy. So our dloss is just a bunch of ones!
#     We can create that in here manually as our "upstream" grad. We 
#     already do this in our .backward() in our tensor, but that only
#     returns a single 1
#     """
    
#     # Detect tensor type
#     is_torch = isinstance(logits, torch.Tensor)
#     is_cupy = not is_torch and hasattr(logits, '__cuda_array_interface__')
    
#     # Convert CuPy to PyTorch for better Triton performance
#     if is_cupy:
        
#         logits = from_dlpack(logits)
#         labels = from_dlpack(labels)
#         logsumexp = from_dlpack(logsumexp)
    
#     N, C = logits.shape
#     row_stride = logits.stride(0)
    
#     grad = torch.zeros_like(logits, dtype=torch.float32)

#     grid = (N, triton.cdiv(C, BLOCK_SIZE))
#     cross_entropy_backward[grid](
#         logits,
#         row_stride,
#         grad, 
#         logsumexp, 
#         labels, 
#         C, 
#         BLOCK_SIZE
#     )

#     # Convert back to CuPy if needed
#     if is_cupy:
#         import cupy as cp
#         grad = cp.from_dlpack(grad)

#     return grad

if __name__ == "__main__":

    import torch
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    def run_torch_triton_forward(logits, labels):
        N, NUM_CLASSES = logits.shape
        logits_ptr = logits.data_ptr()
        labels_ptr = labels.data_ptr()
        loss = torch.zeros(N, dtype=torch.float32, device=logits.device)
        logsumexp = torch.zeros(N, dtype=torch.float32, device=logits.device)
        
        ### Each block will process a full row ###
        BLOCK_SIZE = triton.next_power_of_2(NUM_CLASSES)

        grid = (N,)
        cross_entropy_forward[grid](
            logits_ptr,
            NUM_CLASSES,
            loss.data_ptr(),
            logsumexp.data_ptr(),
            labels_ptr,
            NUM_CLASSES,
            BLOCK_SIZE
        )
        return loss, logsumexp

    def run_torch_triton_backward(original_logits, 
                                  grad, 
                                  labels, 
                                  dloss, 
                                  logsumexp, 
                                  BLOCK_SIZE=128):
        """
        I hardcode the blocksize to somethign reasonable here, but it should
        be tuned! 
        """

        N, NUM_CLASSES = original_logits.shape
        orig_logits_ptr = original_logits.data_ptr()
        labels_ptr = labels.data_ptr()
        
        grid = (N, (NUM_CLASSES + BLOCK_SIZE - 1) // BLOCK_SIZE)
        cross_entropy_backward[grid](
            orig_logits_ptr,       # For loading x (original)
            NUM_CLASSES,           # Row stride in elements
            grad.data_ptr(),       # For storing y (grad output)
            dloss.data_ptr(),
            logsumexp.data_ptr(),
            labels_ptr,
            NUM_CLASSES,
            BLOCK_SIZE
        )
        return grad  # Already modified in-place


    def benchmark_cross_entropy_forward(N, C, n_trials=100):
        print(f"=== Forward N={N}, C={C} ===")
        
        # --------------------------
        # PyTorch setup
        # --------------------------
        logits_torch = torch.randn(N, C, device="cuda", dtype=torch.float32)
        labels_torch = torch.randint(0, C, (N,), device="cuda", dtype=torch.long)
        
        # FLOPs estimate (softmax + logsumexp + subtract label)
        flops = 5 * N * C
        
        # PyTorch forward
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_trials):
            loss_torch = torch.nn.functional.cross_entropy(logits_torch, labels_torch, reduction='none')
        torch.cuda.synchronize()
        t_pt = (time.time() - t0) / n_trials
        flops_pt = flops / t_pt / 1e9  # GFLOPS

        # --------------------------
        # CuPy + Triton setup
        # --------------------------
        logits_cp = cp.array(logits_torch.detach().cpu().numpy(), dtype=cp.float32)
        labels_cp = cp.array(labels_torch.detach().cpu().numpy(), dtype=cp.int64)

        # Warmup Triton
        _ = fused_cross_entropy_forward(logits_cp, labels_cp)
        cp.cuda.runtime.deviceSynchronize()

        # Triton forward
        t0 = time.time()
        for _ in range(n_trials):
            loss_cp, _ = fused_cross_entropy_backward(logits_cp, labels_cp)
        cp.cuda.runtime.deviceSynchronize()
        t_triton = (time.time() - t0) / n_trials
        flops_triton = flops / t_triton / 1e9

        # --------------------------
        # Compare correctness
        # --------------------------
        max_diff = cp.max(cp.abs(loss_cp - cp.array(loss_torch.detach().cpu().numpy()))).item()
        print(f"PyTorch: {t_pt*1000:.2f} ms, {flops_pt:.2f} GFLOPS")
        print(f"Triton:  {t_triton*1000:.2f} ms, {flops_triton:.2f} GFLOPS, max diff {max_diff:.3e}")
        print("-"*50)
        
        return t_pt, flops_pt, t_triton, flops_triton, max_diff


    def benchmark_cross_entropy_backward(N, C, n_trials=100):
        print(f"=== Backward N={N}, C={C} ===")

        # --------------------------
        # PyTorch setup
        # --------------------------
        logits_torch = torch.randn(N, C, device="cuda", dtype=torch.float32, requires_grad=True)
        labels_torch = torch.randint(0, C, (N,), device="cuda", dtype=torch.long)
        dloss = torch.ones(N, device="cuda", dtype=torch.float32)

        # Single forward pass for correctness comparison
        loss_torch = torch.nn.functional.cross_entropy(logits_torch, labels_torch, reduction='none')
        loss_torch_sum = loss_torch.sum()
        logits_torch.grad = None
        loss_torch_sum.backward()
        grad_pt = logits_torch.grad.detach().clone()

        # --------------------------
        # CuPy + Triton setup
        # --------------------------
        logits_cp = cp.array(logits_torch.detach().cpu().numpy(), dtype=cp.float32)
        labels_cp = cp.array(labels_torch.detach().cpu().numpy(), dtype=cp.int64)
        dloss_cp = cp.ones(N, dtype=cp.float32)
        grad_cp = cp.zeros((N, C), dtype=cp.float32)

        # Compute logsumexp once using forward kernel
        _, logsumexp_cp = fused_cross_entropy_forward(logits_cp, labels_cp)

        # Warmup Triton
        _ = fused_cross_entropy_backward(logits_cp, grad_cp, labels_cp, dloss_cp, logsumexp_cp)
        cp.cuda.runtime.deviceSynchronize()

        # Timing loop
        t0 = time.time()
        for _ in range(n_trials):
            grad_cp = fused_cross_entropy_backward(logits_cp, grad_cp, labels_cp, dloss_cp, logsumexp_cp)
        cp.cuda.runtime.deviceSynchronize()
        t_triton = (time.time() - t0) / n_trials

        # FLOPs estimate
        flops = 5 * N * C
        flops_pt = flops / (t_triton * n_trials) / 1e9  # GFLOPS (optional)
        flops_triton = flops / t_triton / 1e9

        # Compare correctness
        grad_triton_torch = torch.from_numpy(cp.asnumpy(grad_cp)).to("cuda")
        max_diff = (grad_triton_torch - grad_pt).abs().max().item()

        print(f"Triton Backward: {t_triton*1000:.2f} ms, {flops_triton:.2f} GFLOPS, max diff {max_diff:.3e}")
        print("-"*50)

        return t_triton, flops_triton, max_diff

    # Lists to hold results
    sizes = [(4096, 128), (4096, 256), (4096, 512), (4096, 1024), 
            (4096, 2048), (4096, 4096), (4096, 8192)]

    forward_times, forward_flops, forward_diffs = [], [], []
    backward_times, backward_flops, backward_diffs = [], [], []

    # Run forward benchmarks
    print("BENCHMARKING FORWARD")
    for N, C in sizes:
        t_pt, fl_pt, t_tr, fl_tr, max_diff = benchmark_cross_entropy_forward(N, C, n_trials=10)
        forward_times.append((t_pt*1000, t_tr*1000))  # ms
        forward_flops.append((fl_pt, fl_tr))         # GFLOPS
        forward_diffs.append(max_diff)

    # Run backward benchmarks
    print("BENCHMARKING BACKWARD")
    for N, C in sizes:
        t_tr, fl_tr, max_diff = benchmark_cross_entropy_backward(N, C, n_trials=10)
        backward_times.append((0, t_tr*1000))  # PyTorch timing not measured for consistency, set 0
        backward_flops.append((0, fl_tr))      # PyTorch GFLOPS not measured
        backward_diffs.append(max_diff)

    # Convert to NumPy arrays for plotting
    C_values = [C for _, C in sizes]
    forward_times = np.array(forward_times)
    forward_flops = np.array(forward_flops)
    backward_times = np.array(backward_times)
    backward_flops = np.array(backward_flops)

    plt.figure(figsize=(12,5))

    # Forward time
    plt.subplot(1,2,1)
    plt.plot(C_values, forward_times[:,0], "-o", label="PyTorch Forward", linewidth=3)
    plt.plot(C_values, forward_times[:,1], "-o", label="Triton Forward", linewidth=3)
    plt.xlabel("Number of Classes (C)")
    plt.ylabel("Time (ms)")
    plt.title("Forward Cross-Entropy Runtime")
    plt.legend()
    plt.grid(True)

    # Forward GFLOPS
    plt.subplot(1,2,2)
    plt.plot(C_values, forward_flops[:,0], "-o", label="PyTorch Forward", linewidth=3)
    plt.plot(C_values, forward_flops[:,1], "-o", label="Triton Forward", linewidth=3)
    plt.xlabel("Number of Classes (C)")
    plt.ylabel("Throughput (GFLOPS)")
    plt.title("Forward Cross-Entropy GFLOPS")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark/cross_entropy.png")
    plt.show()