from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_array, get_inner_inner_array
from ..fused_ops import fused_conv2d_forward, fused_conv2d_backward

def manual_conv2d(input, weight, bias=None, stride=1, padding=0, **args):

    """
    Reference: https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py

    Conv2d using im2col + matmul. No Auto mode supported 
    as backprop through this operation would be annoying 
    to do, and its just faster to have dedicated 
    forward/backward methods

    input: Tensor of shape (B, C_in, H, W)
    weight: Tensor of shape (C_out, C_in, K, K)
    bias: Tensor of shape (C_out,)

    To avoid making a giant intermediate array we will use stride tricks:
    
    What are strides? Its the internal memory layout

    ```
    x = np.arange(16).reshape(4,4)
    print(x)
    print(x.strides)

    [[ 0  1  2  3]
    [ 4  5  6  7]
    [ 8  9 10 11]
    [12 13 14 15]]
    
    (32, 8)

    To go from element 0 to 4 (down a row) you have to move 
    32 bits. To go from element 0 to 1 (next column) you have 
    to move 8 bits

    ```

    How can we take advantage of this to make a copy of data? 

    ```
    from numpy.lib.stride_tricks import as_strided
    
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    # Create a 3x3 rolling window view
    window_size = 3
    strided = as_strided(a, 
                        shape=(len(a) - window_size + 1, window_size),
                        strides=(4, 4))  # 4 bytes per int
    print(strided)

    [[1 2 3]
    [2 3 4]
    [3 4 5]]
    
    ```

    """

    input_arr = get_inner_array(input)
    weight_arr = get_inner_array(weight)
    if bias is not None:
        bias_arr = get_inner_array(bias)

    ### Get Input/Output Shapes ###
    B, C_in,  H, W = input_arr.shape
    C_out, _, K, K_w = weight_arr.shape
    S,P = stride, padding

    H_out = (H + 2*P - K)//S + 1
    W_out = (H + 2*P - K)//S + 1
    
    ### Get Backend (np or xp) ###
    xp = input.xp

    ### Pad Data If Padding is set ###
    if P > 0:
        x_padded = xp.pad(input_arr, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
    else:
        x_padded = input_arr

    ### Use stride tricks for efficient im2col ###   
    ### First, each slice is a kxk patch in the image at ###
    ### some position (i,j) of the output ###
    shape = (B, C_in, K, K, H_out, W_out)
    strides = (
        x_padded.strides[0], # Number of bits to move to get to next batch
        x_padded.strides[1], # Number of bits to move to get to next channel
        x_padded.strides[2], # Number of bits to move to get to next row in kernel
        x_padded.strides[3], # Number of bits to move to get to next col in kernel
        S*x_padded.strides[2], # Number of bits to move to get to next row in output
        S*x_padded.strides[3] # Number of bits to move to get to next col in output
    )

    ### Grab Strided View of our Data (no extra copy needed!) ###
    cols = xp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

    ### Flatten to our wanted dim of (B*H_out*W_out x C_in*K*K) ###
    cols = cols.reshape(B, C_in*K*K, H_out*W_out).transpose(0,2,1)
    cols_flat = cols.reshape(B*H_out*W_out, -1)

    ### Flatten Weights for Operation ###
    weights_flat = weight_arr.reshape(C_out, -1).T

    ### Forward ###
    output = xp.matmul(cols_flat, weights_flat)
    if bias is not None:
        output += bias_arr

    #### Reshape back to (B x C_out x H_out x W_out) ###
    output = output.reshape(B, H_out*W_out, C_out).transpose(0,2,1).reshape(B, C_out, H_out, W_out)

    def _conv2d_backward(grad_output):

        """
        Input (1x4x4): (simple one channel input)

        [ a00 a01 a02 a03 ]
        [ a10 a11 a12 a13 ]
        [ a20 a21 a22 a23 ]
        [ a30 a31 a32 a33 ]

        Kernel slides over 2x2 regions, lets pick the first two patches:

        Patch1:
        [ a00 a01 ]
        [ a10 a11 ]

        Patch2: 
        [ a01 a02 ]
        [ a11 a12 ]

        ### Each patch is a 2x2 patch, which is flattened to a vector of length 4. 
        Patch1 -> [ a00, a01, a10, a11 ]
        Patch2 -> [ a01, a02, a11, a12 ]
        ...

        ### Each patch is matrix multipled by a (C*K*K=1*2*2 x C_out=1)
        ### for simplicity lets just say our output also has 1 channel 

        output1 -> [ a00, a01, a10, a11 ] @ (4 x 1) = (1x1) -> o00
        output2 -> [ a01, a02, a11, a12 ] @ (4 x 1) = (1x1) -> o01

        The overall output of convolving our (1x4x4) with a (1x2x2) with a stride of 1
        will be a (1x3x3) that looks like:

        [ o00 o01 o02 ]
        [ o10 o11 o12 ]
        [ o20 o21 o22 ]


        but in actuallity, because our data was flattened, it would look like:

        [[o00],
        [o01],
        [o02]
        [o12],
        [o11],
        [o12],
        [o20],
        [o21],
        [o22]]

        So now during backprop, we will have gradients go to all the input pixels. So this will
        look like:

        Patch1_grad → [ g00, g01, g10, g11 ]
        Patch2_grad → [ g01, g02, g11, g12 ]

        Notice that we have some repeats. g01 occurs in both patches, and thats because
        pixel p01 were a part of both operations. Therefore we have to accumulate the 
        contributions

        And we need to accumulate all these gradients into our original image shape.

        Patch1 Gradient Contrubutions
        [ g00 g01 0 0 ]
        [ g10 g11 0 0 ]
        [  0   0  0 0 ]
        [  0   0  0 0 ]

        Patch2 Gradient Contrubutions
        [  0  g01 g02 0 ]
        [  0  g11 g12 0 ]
        [  0   0   0  0 ]
        [  0   0   0  0 ]

        ...

        We could do this with a for loop like we did in our original implementation:
        
        ```
        https://github.com/priyammaz/PyTorch-Adventures/blob/main/Neural%20Networks%20from%20Scratch/ManualGrad/nn.py
        for i in range(out_h):
            for j in range(out_w):
                ### Extract corresponding column that goes with this patch in the ####
                ### original image. This will be used in our backward pass on our gradients ###
                ### to return our grad tensor from a simple matrix (for matmul in linear) ###
                ### to the normal shape of a convolution. Due to the overlap between patches 
                ### we want to accumulate up all the overlapping contributions to gradients ###
                ### Hence we use += ###
                patch = cols[:, i*out_w + j, :].reshape(B, C, K, K)
                x_padded[:, :, i*S:i*S+K, j*S:j*S+K] += patch
        ```

        But why not make a more efficient version with annoying indexing?

        ### INDEXING 

        Our output grad (w.r.t the input) will be of the same shape as our input data
        we just need to get the indexing right. 

        Input indices (H x W). Because we flatten the spatial (and channel) dimension
                            the positions are treated as a vector from 0 to 15

        [ 0  1  2  3 ]
        [ 4  5  6  7 ]
        [ 8  9 10 11 ]
        [12 13 14 15 ]

        Kernel 2x2 slides across → 3x3 output

        Patch 0 (top-left): [0,1,4,5]
        Patch 1 (top-middle): [1,2,5,6]
        Patch 2 (top-right): [2,3,6,7]
        Patch 3 (mid-left): [4,5,8,9]

        ### Compute i0 and j0 (Kernel-Local Offsets) ###
        i0 = cp.repeat(cp.arange(k), k) -> [0,0,1,1]
        j0 = cp.tile(cp.arange(cp.arange(k), k)) -> [0,1,0,1]
        k = cp.repeat(cp.arange(C_in), K*K) -> [0,0,0,0]

        This tells us the row/column offsets in each patch. For example:
        (i0, j0) = [(0,0), (0,1), (1,0), (1,1)]

        Well those are the 4 possible positions in our kernel. Top left, top right, 
        bottom left, bottom right (as its just a 2x2 kernel). And we have this for 
        every channel, in our case its just the 1 channel (j = 0) 

        ### Compute i1 and j1 for our sliding offsets ##
        The i0,j0 only really give us the positions of the top left patch ###
        but we can move it over by stride amounts so we have this for all patches ###
        
        For example, i1 is our height offset. We have three positions along the 
        height that the kernel can be at (as we have a stride of 1) so lets
        create some indexes that indicate that. But remember, at one height, 
        we have also three possible column indexes we can be at. So we have essentialyl 
        9 offsets:
        
        i1 = stride * cp.repeat(cp.arange(H_out), W_out) -> [0,0,0,1,1,1,2,2,2]

        [0,0,0,1,1,1,2,2,2] tells us the first three patches have a height offset of 0
                                    the second three patches have a height offset of 1
                                    the third three patches have a height offset of 2

        And thats exactly what we are doing!

        In our matrix:

        [ a00 a01 a02 a03 ]
        [ a10 a11 a12 a13 ]
        [ a20 a21 a22 a23 ]
        [ a30 a31 a32 a33 ]

        The first convolution looks at:

        [ a00 a01 ]
        [ a10 a11 ]

        The second convolution looks at 

        [ a01 a02 ]
        [ a11 a12 ]

        The third convolution looks at

        [ a02 a03 ]
        [ a12 a13 ]

        So we are at the same height (same row) but different columns. Thus we need tp 
        do the same thing the other way. We have our height offsets, we also need our
        width (column) offsets:

        j1 = S * cp.tile(cp.arange(W_out), H_out) -> [0,1,2,0,1,2,0,1,2]

        [0,1,2,0,1,2,0,1,2] tells us the first patch has a column offset of 0
                                    the second patch has a column offset of 1
                                    the third patch has a column offset of 2
                                    the fourth path has a column offset of 1 (coming back to the leftmost)

        And thats exactly what we saw above!

        Therefore i1 gives us the row offset (height) and j1 gives us the column offset (width)

        So lets go ahead and apply the offsets to our patch indexes:

        i = i0.reshape(-1,1) + i1.reshape(1,-1)

        [0]     [0 0 0 1 1 1 2 2 2]   [0 0 0 1 1 1 2 2 2]
        [0]  +  [0 0 0 1 1 1 2 2 2] = [0 0 0 1 1 1 2 2 2]
        [1]     [0 0 0 1 1 1 2 2 2]   [1 1 1 2 2 2 3 3 3]
        [1]     [0 0 0 1 1 1 2 2 2]   [1 1 1 2 2 2 3 3 3]

        j = j0.reshape(-1,1) + j1.reshape(1,-1)
                                    
        [0]     [0 1 2 0 1 2 0 1 2]   [0 1 2 0 1 2 0 1 2]
        [1]  +  [0 1 2 0 1 2 0 1 2] = [1 2 3 1 2 3 1 2 3]
        [0]     [0 1 2 0 1 2 0 1 2]   [0 1 2 0 1 2 0 1 2]
        [1]     [0 1 2 0 1 2 0 1 2]   [1 2 3 1 2 3 1 2 3]

        Lets look at our original data 

        [ 0  1  2  3 ]
        [ 4  5  6  7 ]
        [ 8  9 10 11 ]
        [12 13 14 15 ]

        And to make it simpler lets give everying their indices:

        [ (0,0) (0,1) (0,2) (0,3) ]
        [ (1,0) (1,1) (1,2) (1,3) ]
        [ (2,0) (2,1) (2,2) (2,3) ]
        [ (3,0) (3,1) (3,2) (3,3) ]


        Lets look at the first column our our i and j outputs:

        i[:, 0] = [0 0 1 1]
        j[:, 0] = [0 1 0 1] 

        Just like before the combination of these two give:

        i,j = [(0,0), (0,1), (1,0), (1,1)]

        That looks like the top left corner!

        What about the next column?

        i[:, 1] = [0 0 1 1]
        j[:, 1] = [1 2 1 2]

        The combination gives:

        i,j = [(0,1), (0,2), (1,1), (1,2)]

        Thats the next patch! 

        Lets look at the last one:

        j[:, -1] = [2 2 3 3]
        j[:, -1] = [2 3 2 3]

        The combination gives:

        i,j = [(2,2), (2,3), (3,2), (3,3)]

        That is our bottom right patch!

        Of course we dont have just one patch but we have the same patch across the batch dimension.
        Lets pretend our C_in is 3

        For this we create:
        k = cp.repeat(cp.arange(C_in), K*K).reshape(-1,1)

        This tells us for the first 4 positions (in our 2x2 kernel) they go to channel 0, 
        the second 4 positions (in our 2x2 kernel) goes to channel 1, etc...
        [0 0 0 0 1 1 1 1 2 2 2 2]

        This is a tensor that repeats 0 to num_channels, for each value in our kernel, as if our 
        kernel is 2 x 2, and we have C_in channels, that means we have C_in channels for each of 
        those 2 x 2 kernels!

        Next we create:
        kk_per_batch = cp.repeat(k.reshape(-1), num_p)

        We have our k from above for every single patch of 2x2, so repeat it again for all patches. If we had 
        2 patches it would just repeat again:

        This tell us the first 4 positions belong to the first 2x2 kernel patch and is in channel 0. The second
        4 positions belong to the second 2x2 kernel patch and is also in channel 0, etc...
        [ 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 ]

        Lastly, we want to tile this whole operation on the batch dimension, so if we had a batch of 2 we would get:
        kk_flat = cp.tile(kk_per_batch, B)
        [ 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 ]

        Therefore this is our annoying indexing. The reason we care is because we 
        can now use a faster cp.add.at method:

        Similarly, we need to tile our i and j indices to also repeat over the batch dimension, we can use tile for that:
        ii_flat = cp.tile(i.reshape(-1), B)
        jj_flat = cp.tile(j.reshape(-1), B)


        # For every sample in the batch:
        #     grad_input[b] is tensor we want to accumulate grads into
        #     (k,i,j) is the (channel x height x width) index combinations we want to accumulate into
        #     grad_cols[b].T will be the actual values we want to grab to accumulte them
        
        # for b in range(B):
        #     cp.add.at(grad_input[b], (k, i, j), grad_cols[b].T)


        """
        
        ### Put grad output back in original shape that we processed with ###
        grad_output_flat = grad_output.reshape(B, C_out, -1).transpose(0, 2, 1).reshape(B*H_out*W_out, C_out)
        
        if weight.requires_grad:

            ### grad_W is (B*H_out*W_out x C_in*K*K).T @ (B*H_out*W_out, C_out) -> (C_in*K*K x C_out) ###
            grad_W = xp.matmul(cols_flat.T, grad_output_flat)

            ### Remeber our weights are actually in the shape of (C_out x C_in x K x K), so ###
            ### we need a transpose first to (C_in*K*K x C_out) -> (C_out x C_in*K*K) -> (C_out, C_in, K, K) ###
            grad_W = grad_W.T.reshape(C_out, C_in, K, K)

            if weight.grad is None:
                weight.grad = grad_W
            else:
                weight.grad += grad_W
        
        if bias is not None and bias.requires_grad:
            grad_b = grad_output_flat.sum(axis=0)
            
            if bias.grad is None:
                bias.grad = grad_b
            else:
                bias.grad += grad_b

        # Input gradient
        if input.requires_grad:
            
            ### Compute Flattened Values and Indices for Vectorized add.at ###
            num_k = C_in * K * K
            num_p = H_out * W_out

            ### Get our gradients in the original shape ###
            ### (B*H*W, C_out) @ (C_out, C_in*K*K) -> [B*N_patches, C*K*K]
            grad_cols_flat = xp.matmul(grad_output_flat, weights_flat.T) 

            ### Reshape to expose Batch dimension ###
            grad_cols = grad_cols_flat.reshape(B, H_out*W_out, C_in*K*K)

            ### Create empty tensor to accumulate grads into ###
            grad_input = xp.zeros_like(x_padded, dtype=x_padded.dtype)

            ### Flatten Values in order of Batches and then Kernels and then Patches ###
            values_flat = grad_cols.transpose(0,2,1).reshape(-1)

            # Batch indices: repeat each b for all its kernel-patch pairs
            bb_flat = xp.repeat(xp.arange(B), num_k * num_p)

            ### Do the indexing op as described above ###
            i0 = xp.repeat(xp.arange(K), K)
            i0 = xp.tile(i0, C_in)
            i1 = S * xp.repeat(xp.arange(H_out), W_out)
            j0 = xp.tile(xp.arange(K), K * C_in)
            j1 = S * xp.tile(xp.arange(W_out), H_out)
            i = i0.reshape(-1,1) + i1.reshape(1,-1)
            j = j0.reshape(-1,1) + j1.reshape(1,-1)
            k = xp.repeat(xp.arange(C_in), K*K)
        
            ### Channel Indices: Repeat each K for its patches and then tile over batches ###
            kk_per_batch = xp.repeat(k, num_p)
            kk_flat = xp.tile(kk_per_batch, B)

            ### Repeat Row and Col indices for every sample in batch too ###
            ii_flat = xp.tile(i.reshape(-1), B)
            jj_flat = xp.tile(j.reshape(-1), B)

            # print(bb_flat.shape, kk_flat.shape, ii_flat.shape, jj_flat.shape, values_flat.shape, grad_input.shape)
            # (147456,) (147456,) (147456,) (147456,) (147456,) (16, 256, 6, 6) and 16x256x6x6 = 147456
            xp.add.at(grad_input, (bb_flat, kk_flat, ii_flat, jj_flat), values_flat)

            ### Remove padding that didnt exist before the conv ###
            if P > 0:
                grad_input = grad_input[:, :, P:-P, P:-P]

            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input

    requires_grad = input.requires_grad or weight.requires_grad or \
                        (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_conv2d_backward if requires_grad else None,
        grad_fn_name="<Conv2dBackward>" if requires_grad else None
    )
    
    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output     

def fused_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1):

    ### Fused ops need raw cupy arrays not Array ###
    input_arr = get_inner_inner_array(input)
    weight_arr = get_inner_inner_array(weight)
    bias_arr = get_inner_inner_array(bias) if bias is not None else None

    ### Get Input/Output Shapes ###
    B, C_in,  H, W = input_arr.shape
    C_out, _, K, K_w = weight_arr.shape
    S,P = stride, padding

    output = fused_conv2d_forward(input_arr, weight_arr, bias_arr, 
                                  stride, padding, dilation=dilation)

    def _conv2d_backward(grad_output):
        
        grads = fused_conv2d_backward(grad_output, input_arr, weight_arr, 
                                      bias_arr,
                                      H, W, K, K_w, 
                                      stride, padding, dilation=dilation)
        
        if bias is not None:
            dinput, dweight, dbias = grads
        else:
            dinput, dweight = grads

        ### Accumulate Grads ####
        if input.grad is None:
            input.grad = dinput
        else:
            input.grad += dinput

        if weight.grad is None:
            weight.grad = dweight
        else:
            weight.grad += dweight
        
        if bias is not None:
            if bias.grad is None:
                bias.grad = dbias
            else:
                bias.grad += dbias

    requires_grad = input.requires_grad or weight.requires_grad or \
                        (bias is not None and bias.requires_grad)
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        output,
        requires_grad=requires_grad,
        grad_fn=_conv2d_backward if requires_grad else None,
        grad_fn_name="<Conv2dBackward>" if requires_grad else None
    )
    
    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output   

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, fused=False):

    """
    This toggles between the different methods implemented!
    """

    _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
    op = fused_conv2d if _use_fused else manual_conv2d
    if not _use_fused and dilation > 1:
        raise Exception("Non-Fused Conv2d does not support Dilations Greater than 1!")
    if fused and op is manual_conv2d:
        CHECKS.warn_triton_missing()
    return op(input, weight, bias, stride=stride, padding=padding, dilation=dilation) 