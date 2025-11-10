import numpy as np
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_cross_entropy_forward, fused_cross_entropy_backward

def auto_cross_entropy(logits, targets, ignore_index=-100):
    
    *dims, num_classes = logits.shape
    flattened_dim = np.prod(dims)
    logits = logits.reshape(flattened_dim, num_classes)

    ### Make sure targets are always int32 ###
    targets = targets.astype("int32", copy=False)

    ### Flatten Targets ###
    targets = targets.reshape(flattened_dim)

    ### Mask out our ignore index (-100 by default) ###
    mask = (targets != ignore_index)
    logits = logits[mask]
    targets = targets[mask]

    ### Get number of valid labels so we can compute the avg over them ###
    valid_count = mask.sum()

    ### Stable Log-Softmax ###
    logits_shifted = logits - logits.max(dim=1, keepdims=True)

    ### Log Sum Exp ###
    logsumexp = (logits_shifted.exp()).sum(dim=1, keepdims=True).log()

    ### Log Softmax ###
    log_softmax = logits_shifted - logsumexp

    ### Negative Log Likelihood For Correct Class ###
    nll = -log_softmax[np.arange(len(targets)), targets] / valid_count

    ### Mean Loss ###
    loss = nll.sum()

    return loss

def manual_cross_entropy(logits, targets, ignore_index=-100):

    *dims, num_classes = logits.shape
    flattened_dim = np.prod(dims)
    
    logits_data = get_inner_array(logits).reshape(flattened_dim, num_classes).astype("float32", copy=False)
    logits_data = logits_data.reshape(flattened_dim, num_classes).astype("float32", copy=False)
    targets_data = get_inner_array(targets).reshape(flattened_dim)

    mask = (targets_data != ignore_index)
    valid_counts = mask.sum()

    # Stable logsumexp per row
    logits_max = np.max(logits_data, axis=1, keepdims=True)
    exp_shifted = np.exp(logits_data - logits_max)  # shape (B, C)
    logsumexp = np.log(np.sum(exp_shifted, axis=1, keepdims=True)) + logits_max  # shape (B, 1)

    # Negative log-likelihood only for valid rows
    nll = (logsumexp.flatten() - logits_data[np.arange(flattened_dim), targets_data]) * mask
    loss_value = np.sum(nll) / valid_counts

    loss_value = loss_value.astype(logits.dtype, copy=False)
    
    def _cross_entropy_backward(grad_output):

        if logits.requires_grad:
        
            # Compute softmax probabilities for all rows
            softmax = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)  # shape (B, C)
            
            # Initialize gradient
            grad_input = softmax.copy()
            grad_input[np.arange(flattened_dim), targets_data] -= 1  # softmax - one_hot

            # Scale by grad_output and divide by valid counts
            grad_input *= (grad_output / valid_counts)
            
            # Zero out ignored rows
            grad_input *= mask.reshape(-1,1)

            # Reshape back to original logits shape and dtype
            grad_input = grad_input.reshape(logits.shape).astype(logits.dtype, copy=False)

            if logits.grad is None:
                logits.grad = grad_input
            else:
                logits.grad += grad_input

    requires_grad = logits.requires_grad
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        loss_value,
        requires_grad=requires_grad,
        grad_fn=_cross_entropy_backward if requires_grad else None,
        grad_fn_name="<CrossEntropyBackward>" if requires_grad else None
    )
    
    if requires_grad:
        out._add_parents(logits)

    return out

def fused_cross_entropy(logits, targets, ignore_index=-100):

    assert (ignore_index == -100), "Fused Cross entorpy hardcoded ignore_index to -100, use the same here!"

    *dims, num_classes = logits.shape
    flattened_dim = np.prod(dims)

    logits_data = get_inner_inner_array(logits).reshape(flattened_dim, num_classes)
    targets_data = get_inner_inner_array(targets).reshape(flattened_dim).astype("int32", copy=False)

    ### -100 is hard coded into the fused kernel, so we hardcode it here as well! ###
    targets_flat = targets_data
    mask = (targets_flat != -100)
    valid_counts = mask.sum().get().item()
    
    # Triton kernel forward
    loss_cp, logsumexp_cp = fused_cross_entropy_forward(logits_data, targets_data)
    
    ### Average by the number of valid elements in the array ###
    loss_value = loss_cp.sum() / valid_counts

    ### Cast back to the wanted dtype ###
    loss_value = loss_value.astype(logits.dtype, copy=False)

    def _cross_entropy_backward(grad_output):
        
        ### The grad output is just a single value. It will be 1 for fp32 training but some scale in ###
        ### fp16 training (dynamic grad scaling!) Our valid_Counts is also just some constant so ###
        ### instead of multiplying the output of grad_cp by this scale, we pass it into our kernel to ###
        ### just do it all at once! ###
        if logits.requires_grad:

            grad_cp = fused_cross_entropy_backward(
                logits_data,
                targets_data,
                logsumexp_cp,
                scale=(grad_output.get().item() / valid_counts)
            ).reshape(*logits.shape).astype(logits.dtype, copy=False)

            if logits.grad is None:
                logits.grad = grad_cp
            else:
                logits.grad += grad_cp

    requires_grad = logits.requires_grad
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        loss_value,
        requires_grad=requires_grad,
        grad_fn=_cross_entropy_backward if requires_grad else None,
        grad_fn_name="<CrossEntropyBackward>" if requires_grad else None
    )

    if requires_grad:
        out._add_parents(logits)

    return out

def cross_entropy(logits, targets, ignore_index=-100, auto=False, fused=False):
    if auto:
        return auto_cross_entropy(logits, targets, ignore_index)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        op = fused_cross_entropy if _use_fused else manual_cross_entropy
        if fused and op is manual_cross_entropy:
            CHECKS.warn_triton_missing()
        return op(logits, targets, ignore_index)