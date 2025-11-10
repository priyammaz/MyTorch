from mytorch import Tensor

def averagepool2d(input, kernel_size, stride=None, padding=0):
    """
    AveragePool2d using im2col. Supports manual backward.

    Args:
        input: Tensor of shape (B, C, H, W)
        kernel_size: int or tuple
        stride: int or tuple
        padding: int
    """

    ###################
    ### MANUAL GRAD ###
    ###################

    ### Get Backend ###
    xp = input.xp

    ### Get all of our sizes ###
    B, C, H, W = input.data.shape
    K_h = K_w = kernel_size
    S_h = S_w = stride or kernel_size  # Stride = Kernel size unless provided
    P = padding

    ### Get output size ###
    H_out = (H + 2*P - K_h) // S_h + 1
    W_out = (W + 2*P - K_w) // S_w + 1

    ### Pad input if needed ###
    if P > 0:
        x_padded = xp.pad(input.data, ((0,0),(0,0),(P,P),(P,P)), mode='constant', constant_values=0)
    else:
        x_padded = input.data

    ### Im2Col Algorithm ###
    shape = (B, C, K_h, K_w, H_out, W_out)
    strides = (
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2],
        x_padded.strides[3],
        S_h * x_padded.strides[2],
        S_w * x_padded.strides[3],
    )
    cols = xp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    cols_flat = cols.reshape(B, C, K_h*K_w, H_out*W_out)

    ### Forward: Compute the mean over the kernel window ###
    out = xp.mean(cols_flat, axis=2).reshape(B, C, H_out, W_out)

    def _averagepool2d_backward(grad_outputs):
        """
        Backward pass for AveragePool2d.
        
        In MaxPool2d, the gradient is assigned to the max index only. For AveragePool2d,
        the gradient is distributed equally across all positions in the kernel window (K_h*K_w).
        Each position gets grad_output / (K_h * K_w). We use the same im2col structure to
        distribute gradients back to the input tensor.

        Steps:
        1. Create a grad_cols array of shape (B, C, K_h*K_w, H_out*W_out).
        2. For each position in the kernel window, assign grad_output / (K_h * K_w).
        3. Use col2im to accumulate gradients into the input shape, handling overlaps.
        """
        ### Create grad_cols with gradients distributed equally ###
        grad_cols = xp.zeros_like(cols_flat, dtype=grad_outputs.dtype)
        grad_cols += grad_outputs.reshape(B, C, 1, H_out*W_out) / (K_h * K_w)

        ### Col2im to accumulate into grad_input ###
        grad_input = xp.zeros_like(x_padded, dtype=grad_outputs.dtype)

        i0 = xp.repeat(xp.arange(K_h), K_w)
        i0 = xp.tile(i0, C)
        i1 = S_h * xp.repeat(xp.arange(H_out), W_out)
        j0 = xp.tile(xp.arange(K_w), K_h * C)
        j1 = S_w * xp.tile(xp.arange(W_out), H_out)
        i = i0.reshape(-1,1) + i1.reshape(1,-1)
        j = j0.reshape(-1,1) + j1.reshape(1,-1)
        k = xp.repeat(xp.arange(C), K_h*K_w).reshape(-1,1)

        ### Vectorized over batch ###
        num_k = C * K_h * K_w
        num_p = H_out * W_out

        # Reshape grad_cols to (B, num_k, num_p)
        grad_values = grad_cols.reshape(B, num_k, num_p)

        # Values flattened in row-major order
        values_flat = grad_values.reshape(-1)

        # Batch indices: repeat each b for num_k * num_p times
        bb_flat = xp.repeat(xp.arange(B), num_k * num_p)

        # Channel indices: repeat each k for num_p times, then tile over B
        kk_per_batch = xp.repeat(k.reshape(-1), num_p)
        kk_flat = xp.tile(kk_per_batch, B)

        # Row and col indices: flatten row-major, then tile over B
        ii_flat = xp.tile(i.reshape(-1), B)
        jj_flat = xp.tile(j.reshape(-1), B)

        ### Single vectorized addition ###
        xp.add.at(grad_input, (bb_flat, kk_flat, ii_flat, jj_flat), values_flat)

        if P > 0:
            grad_input = grad_input[:, :, P:-P, P:-P]

        if input.grad is None:
            input.grad = grad_input
        else:
            input.grad += grad_input

    requires_grad = input.requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        out,
        requires_grad=requires_grad,
        grad_fn=_averagepool2d_backward if requires_grad else None,
        grad_fn_name="<AveragePool2dBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input)

    return output