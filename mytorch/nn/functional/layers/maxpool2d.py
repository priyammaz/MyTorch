from mytorch import Tensor

def maxpool2d(input, kernel_size, stride=None, padding=0):
    
    """
    MaxPool2d using im2col + argmax. Supports manual backward.

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
    S_h = S_w = stride or kernel_size # Stride = Kernel size unless provided
    P = padding

    ### Get output size ###
    H_out = (H + 2*P - K_h)//S_h + 1
    W_out = (W + 2*P - K_w)//S_w + 1

    ### Pad input if needed ###
    if P > 0:
        x_padded = xp.pad(input.data, ((0,0),(0,0),(P,P),(P,P)), mode='constant', constant_values=0)
    else:
        x_padded = input.data

    ### Im2Col Algorithm ###
    # --- im2col ---
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

    ### Forward: Get the max values and their indexes ###
    ### Basically, out of the K_h*K_w possibilities, just ###
    ### grab the largest one and its index! ###
    max_idx = xp.argmax(cols_flat, axis=2)
    out = xp.max(cols_flat, axis=2).reshape(B, C, H_out, W_out)

    def _maxpool2d_backward(grad_outputs):
        
        """
        All we have to do is copy grads from the output to its index back in the original 
        tensor! For example, lets say we had the tensor:

        [1,5,2]

        if we did Max we would select 5, and its at index 1.

        This means during backprop our gradient will go to the 5 location and be 0 elsewhere
        as the other values didnt contribute to the gradient. So whatever grad flowed back
        to our 5 value, gets copied in with 0 everywhere else. 

        In our case, we have our data in the shape of (B x C x Kh*Kw x H_out*W_out). In that
        set of Kh*Kw, one of them contributed to the gradient (the max one) and the rest is 0. 
        So to do this all we need to do is index every possible B, C, and H_out*W_out, and then 
        also index for the specific K that was max!

        Easiest way is to use meshgrid:
        
        meshgrid is a function used to create a rectangular 
        grid out of two or more one-dimensional arrays representing coordinate values.

        For example. Lets say we have a 2d matrix of size (5 x 5). The height goes from 0 to 4
        and the width goes from 0 to 4, and I want every combination of points that give me the entire
        space. For example [(0,0), (0,1), (0,2), (0,3), ... (4,4)]

        We could use a for loop, or we can just use meshgrid:

        ```
        import cupy as cp

        H_idx, W_idx = cp.meshgrid(
            cp.arange(5), cp.arange(5), indexing='ij'
        )

        print(H_idx)
        
        [[0 0 0 0 0]
        [1 1 1 1 1]
        [2 2 2 2 2]
        [3 3 3 3 3]
        [4 4 4 4 4]]
 
        print(W_idx)

        [[0 1 2 3 4]
        [0 1 2 3 4]
        [0 1 2 3 4]
        [0 1 2 3 4]
        [0 1 2 3 4]]

        ```

        Well in our case we have dimensions: 

        B -> 0 to num samples in batch
        C -> 0 to num channels in inputs
        H_out*W_out -> 0 to num patches in our output

        And all that is left is to index K from K_h*K_w, which 
        we already know is the max k!

        
        """
        ### Create Empty Grad to Populate in the shape of (B, C, K_h*K_w, H_out*W_out)###
        grad_cols = xp.zeros_like(cols_flat, dtype=input.dtype)

        ### Get Indexes ###
        B_idx, C_idx, N_idx = xp.meshgrid(
            xp.arange(B), xp.arange(C), xp.arange(H_out*W_out), indexing="ij"
        )

        ### Use Indexes to Copy all grad_outputs into our grad_cols ###
        ### Our data is (B x C x K_h*K_w x H_out*W_out) but we are indexing ###
        ### for a specific k (the max index) so our shape we are copying into will ###
        ### be (B x C x H_out*W_out), so we need to make sure our grad_outputs are ###
        ### also flattened into that shape! ###
        grad_cols[B_idx, C_idx, max_idx, N_idx] = grad_outputs.reshape(B, C, -1)

        ### Now that we have our gradients, remember that we could have overlapping kernels ###
        ### Just like we did in our Conv2d above, we need to accumulate all of this into our ###
        ### grad input, this code is identical to above ###

        # Col2im to accumulate into grad_input
        grad_input = xp.zeros_like(x_padded, dtype=x_padded.dtype)

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
        grad_fn=_maxpool2d_backward if requires_grad else None,
        grad_fn_name="<MaxPool2dBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input)

    return output