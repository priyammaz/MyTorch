from mytorch import Tensor

def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0):
    """
    Exactly the same as the conv_transpose2d, just with less dimensions!
    """
    
    ###################
    ### MANUAL GRAD ###
    ###################

    xp = input.xp
    input_cp = input.data
    weight_cp = weight.data
    bias_cp = bias.data if bias is not None else None

    B, C_in, L_in = input_cp.shape
    C_in_w, C_out, K = weight_cp.shape
    assert C_in_w == C_in, f"Input channel mismatch. Expecting {C_in_w}, got {C_in}"
    S, P, OP = stride, padding, output_padding

    # Temporary output length before cropping
    L_temp = (L_in - 1) * S + K
    # Final output length
    L_out = (L_in - 1) * S - 2*P + K + OP

    # Flatten input for matmul
    input_flat = input_cp.transpose(0, 2, 1).reshape(B*L_in, C_in)  # (B*L_in, C_in)
    weights_flat = weight_cp.reshape(C_in, C_out*K)                  # (C_in, C_out*K)

    # Forward matmul
    cols_flat = xp.matmul(input_flat, weights_flat)

    # Reshape to (B, L_in, C_out*K)
    cols = cols_flat.reshape(B, L_in, C_out*K)

    # Padded output
    output_padded = xp.zeros((B, C_out, L_temp))

    # Flatten values for add.at
    num_k = C_out * K
    num_p = L_in
    values_flat = cols.transpose(0, 2, 1).reshape(-1)

    # Batch indices
    bb_flat = xp.repeat(xp.arange(B), num_k * num_p)

    # Compute indices for 1D
    i0 = xp.repeat(xp.arange(K), C_out)
    k = xp.tile(xp.arange(C_out), K)
    i1 = S * xp.repeat(xp.arange(L_in), 1)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    kk_per_batch = xp.repeat(k, num_p)
    kk_flat = xp.tile(kk_per_batch, B)
    ii_flat = xp.tile(i.reshape(-1), B)

    # Accumulate into output
    xp.add.at(output_padded, (bb_flat, kk_flat, ii_flat), values_flat)

    # Crop to final output
    output_data = output_padded[:, :, P:P+L_out]

    if bias is not None:
        output_data += bias_cp.reshape(1, -1, 1)

    # Backward function
    def _conv_transpose1d_backward(grad_output):
        grad_output_data = grad_output.reshape(B, C_out, L_out)

        grad_output_padded = xp.zeros((B, C_out, L_temp), dtype=grad_output.dtype)
        grad_output_padded[:, :, P:P+L_out] += grad_output_data

        values_flat_grad = grad_output_padded[bb_flat, kk_flat, ii_flat]
        grad_cols = values_flat_grad.reshape(B, num_k, num_p).transpose(0, 2, 1)
        grad_cols_flat = grad_cols.reshape(B*L_in, C_out*K)

        if weight.requires_grad:
            grad_weights_flat = xp.matmul(input_flat.T, grad_cols_flat)
            grad_W = grad_weights_flat.reshape(C_in, C_out, K)
            if weight.grad is None:
                weight.grad = grad_W
            else:
                weight.grad += grad_W

        if bias is not None and bias.requires_grad:
            grad_b = grad_output_data.sum((0, 2))
            if bias.grad is None:
                bias.grad = grad_b
            else:
                bias.grad += grad_b

        if input.requires_grad:
            grad_input_flat = xp.matmul(grad_cols_flat, weights_flat.T)
            grad_input = grad_input_flat.reshape(B, L_in, C_in).transpose(0, 2, 1)
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input

    requires_grad = input.requires_grad or weight.requires_grad or \
                    (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()

    output = Tensor(
        output_data,
        requires_grad=requires_grad,
        grad_fn=_conv_transpose1d_backward if requires_grad else None,
        grad_fn_name="<ConvTranspose1dBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)

    return output
