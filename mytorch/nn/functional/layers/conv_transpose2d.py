from mytorch import Tensor

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0):

    """

    ### CONVOLUTION RECAP ###
    The setup for transpose convolution is very similar to standard convolutions, 
    but with an important difference in its purpose. In a standard convolution, 
    the goal is often to compress the spatial dimensions of the input. Each output pixel 
    is computed from multiple input pixels, and because the convolutional filters slide over the 
    input with overlap, a single input pixel can contribute to multiple output pixels.

    To make standard convolutions efficient, we use the im2col algorithm. This rearranges 
    all the sliding patches of the input into columns, allowing the convolution to be 
    expressed as a fast matrix multiplication (GEMM) instead of slow nested loops. During 
    backpropagation, we need to propagate gradients back to the input. Each upstream gradient 
    affects all the input pixels that contributed to it, which is handled using col2im. 
    Col2im scatters and accumulates the flattened gradients back into the original input shape, 
    ensuring that overlapping contributions are summed correctly.

    ### TRANSPOSE CONVS ARE BASICALLY THE SAME JUST BACKWARDS! ###

    Transpose convolutions, on the other hand, have the goal of increasing spatial dimensions. 
    Here, a single input pixel contributes to multiple output pixels, which is conceptually 
    the inverse of standard convolution. Interestingly, this is very similar to the backward pass
    of a regular convolution. In both cases, a single value spreads its influence across 
    multiple positions: in backpropagation, a gradient spreads to all contributing inputs, 
    and in transpose convolution, an input pixel spreads to multiple outputs.

    This similarity allows us to reuse the same computational trick. Just like in 
    backpropagation, we can flatten the input and use a col2im-like accumulation to distribute 
    each input pixel’s contribution to the larger output space. Each “column” of the input, 
    representing flattened channel and kernel dimensions, maps to multiple columns in the expanded output.
    This makes transpose convolution efficient and easy to implement using the same underlying principles 
    as standard convolution and its backward pass.

    Backward pass of a transpose conv is pretty easy too then. A single input contributed to multiple outputs
    so just accumulate all those contributions up and send them back! 

    """

    ### Set Backend ###
    xp = input.xp
    input_cp = input.data
    weight_cp = weight.data
    bias_cp = bias.data

    ### Get Input/Output Shapes ###
    B, C_in, H_in, W_in = input_cp.shape
    C_in_w, C_out, K, _ = weight.data.shape
    assert C_in_w == C_in, f"Input channel mismatch. Expecting {C_in_w} channels, got {C_in}"
    S, P, OP = stride, padding, output_padding


    ### Compute Temporary Output Shape (prepadding/cropping) (https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) ###
    H_temp = (H_in - 1) * S + K
    W_temp = (W_in - 1) * S + K

    ### Final Output Shape ###
    H_out = (H_in - 1) * S - 2*P + K + OP
    W_out = (W_in - 1) * S - 2*P + K + OP

    ### Flatten Data for MatMul (B x C_in x H x W) -> (B*H*W, C_in) ###
    input_flat = input_cp.transpose(0,2,3,1).reshape(B*H_in*W_in, C_in)
    
    ### Flatten Weights for Op ###
    weights_flat = weight_cp.reshape(C_in, C_out*K*K)
    
    ### Forward MatMul ###
    cols_flat = xp.matmul(input_flat, weights_flat)

    ### Reshape Cols back to (B, H_in*W_in, C_out*K*K) ###
    cols = cols_flat.reshape(B, H_in*W_in, C_out*K*K)

    ### Create padded output tensor ###
    output_padded = xp.zeros((B, C_out, H_temp, W_temp))

    ### Compute Flattened Value and Indices for add.at ###
    num_k = C_out * K * K
    num_p = H_in * W_in

    ### Flatten values in order of batch x kernels x patches ###
    values_flat = cols.transpose(0,2,1).reshape(-1)
    
    ### Batch indexes ###
    bb_flat = xp.repeat(xp.arange(B), num_k*num_p)

    ### Indexing Op ###
    i0 = xp.repeat(xp.arange(K), K)
    i0 = xp.tile(i0, C_out)
    j0 = xp.tile(xp.arange(K), K * C_out)
    i1 = S * xp.repeat(xp.arange(H_in), W_in)
    j1 = S * xp.tile(xp.arange(W_in), H_in)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C_out), K * K)

    ### Channel indices ###
    kk_per_batch = xp.repeat(k, num_p)
    kk_flat = xp.tile(kk_per_batch, B)

    ### Tile row and col indices over batch ###
    ii_flat = xp.tile(i.reshape(-1), B)
    jj_flat = xp.tile(j.reshape(-1), B)

    ### Accumulate into output_padded ###
    xp.add.at(output_padded, (bb_flat, kk_flat, ii_flat, jj_flat), values_flat)
    
    ### Add extra padding if we want to match a specific output shape ###
    if OP > 0:
        output_padded = xp.pad(output_padded, ((0,0), (0,0), (0,OP), (0,OP)), mode="constant")

    ### Crop off any padding to remove effect of padding ### 
    ### If we had added extra padding from before, thats would ###
    ### have made the overal size larger after this than if we ###
    ### didnt include it ###
    output_data = output_padded[:, :, P:P+H_out, P:P+W_out]
    
    if bias is not None:
        output_data += bias_cp.reshape(1,-1,1,1)
    
    def _conv_transpose2d_backward(grad_output):

        ### Flatten grad output to match output_data shape ###
        grad_output_data = grad_output.reshape(B, C_out, H_out, W_out)

        ### Create a grad_output_padded_full to store the values of our grads ###
        H_padded = H_temp + OP
        W_padded = W_temp + OP
        grad_output_padded_full = xp.zeros((B, C_out, H_padded, W_padded), dtype=grad_output.dtype)

        ### Add gradient contributions to the pre padded portion ###
        grad_output_padded_full[:, :, P:P+H_out, P:P+W_out] += grad_output_data

        ### Extract the pre-padded portion ###
        grad_output_padded = grad_output_padded_full[:, :, :H_temp, :W_temp]

        ### Gather the values_flat_grad from grad_output_padded using the same indexes from earlier ###
        values_flat_grad = grad_output_padded[bb_flat, kk_flat, ii_flat, jj_flat]

        ### Reshape back to grad_cols ###
        grad_cols = values_flat_grad.reshape(B, num_k, num_p).transpose(0,2,1)
        grad_cols_flat = grad_cols.reshape(B*H_in*W_in, C_out*K*K)

        if weight.requires_grad:
            ### grad_weights_flat = input_flat.T @ grad_cols_flat -> (C_in, C_out*K*K) ###
            grad_weights_flat = xp.matmul(input_flat.T, grad_cols_flat)

            ### Reshape to (C_in, C_out, K, K) ###
            grad_W = grad_weights_flat.reshape(C_in, C_out, K, K)

            if weight.grad is None:
                weight.grad = grad_W
            else:
                weight.grad += grad_W

        if bias is not None and bias.requires_grad:
            grad_b = grad_output_data.sum((0, 2, 3))

            if bias.grad is None:
                bias.grad = grad_b
            else:
                bias.grad += grad_b

        if input.requires_grad:
            ### grad_input_flat = grad_cols_flat @ weights_flat.T -> (B*H_in*W_in, C_in) ###
            grad_input_flat = xp.matmul(grad_cols_flat, weights_flat.T)

            ### Reshape back, accounting for the transpose ###
            grad_input_reshaped = grad_input_flat.reshape(B, H_in, W_in, C_in)
            grad_input = grad_input_reshaped.transpose(0, 3, 1, 2)  # back to (B, C_in, H_in, W_in)

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
        grad_fn=_conv_transpose2d_backward if requires_grad else None,
        grad_fn_name="<ConvTranspose2dBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output