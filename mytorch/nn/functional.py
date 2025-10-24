"""
Functional access for all our Ops! This uses 
the topological sort backward method of Tensor

For some operations:
auto=True indicates that we will use our autograd 
engine to compute grads. But for some known (and complex) ops, we 
can manually define them as well which is auto=False 

Although we use np, they get piped through our Array ops
so they remain cpu/gpu agnostic 

"""
import numpy as np
from ..tensor import Tensor
try:
    import triton
    from . import fused_ops as FO
    FUSED_AVAIL = True
except:
    FUSED_AVAIL = False
    FLAG_ONCE = False

import warnings

def linear(input, weight, bias=None, auto=False):

    """
    Standard linear layer operation w/ support for multidim ops:

    y = x@W.T + b

    x: (B, I)
    W: (I,O)
    b: (O,)
    """

    ### Normally data is in the shape of (N x I)
    reshaped = False
    *dims, in_features = input.shape
    out_features = weight.shape[0]

    ### If our data is (*, I) where * is any number of extra dimensions ###
    ### We need to flatten it! ###
    if len(dims) > 1:
        reshaped = True
    
    if auto: # We can only use methods defined in our Tensor class
        
        ### Flatten Data Dimensions to (*, in_features) ###
        if reshaped:
            input = input.reshape(-1, in_features)

        output = input @ weight.transpose(-1,-2)
        if bias is not None:
            output = output + bias.reshape(1,-1)

        if reshaped:
            output = output.reshape(*dims, out_features)

        return output

    else: # Manual forward and backward

        ### FORWARD PASS ###
        input_xp = input.data
        weight_xp = weight.data.T

        if bias is not None:
            bias_xp = bias.data
        
        ### Flatten data to (N x I) if we have more dimensions ###
        if reshaped:
            input_xp = input_xp.reshape(-1, in_features)

        ### Do MatMul Op ###
        output_shape = (np.prod(dims), out_features) if reshaped else (input_xp.shape[0], out_features)

        ### Preallocation Needs to Occur on the Correct Device ###
        output = input.xp.empty(output_shape, dtype=input_xp.dtype)
        np.matmul(input_xp, weight_xp, out=output)

        if bias is not None:
            np.add(output, bias_xp.reshape(1,-1), out=output)

        ### Return output to original shape (*, O) ###
        if reshaped:
            output = output.reshape(*dims, -1)

        ### BACKWARD PASS ###
        def _linear_backward(grad_output):
            
            ### Our gradients are coming in the shape of (*, O) ###
            ### But our operation happened in the shape of (N x O) ###
            ### So change our grad_output shape to that by flattening ###
            if reshaped:
                grad_output = grad_output.reshape(-1, out_features)

            ### Standard Weight Update formula ###
            if weight.requires_grad:
                grad_W = np.matmul(input_xp.T, grad_output)
                if weight.grad is None:
                    weight.grad = grad_W.T
                else:
                    weight.grad += grad_W.T
            
            ### Standard Bias Update Formula ###
            if bias is not None and bias.requires_grad:
                grad_b = grad_output.sum(axis=0)
                if bias.grad is None:
                    bias.grad = grad_b
                else:
                    bias.grad += grad_b
            
            ### Grad to Input ###
            if input.requires_grad:

                grad_input = np.matmul(grad_output, weight_xp.T)

                ### Reshape grad_input back to input feature shape (* x I) ###
                grad_input = grad_input.reshape(*dims, in_features)
                
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
            grad_fn=_linear_backward if requires_grad else None,
            grad_fn_name="<LinearBackward>" if requires_grad else None
        )

        if requires_grad:
            output._add_parents(input, weight, bias)
            
        return output

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, fused=False):

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
    
    if fused and not FUSED_AVAIL:
        if not FLAG_ONCE:
            warnings.warn("Fused Ops not available, defaulting to normal ops, install Triton for Fused Operations!")
            FLAG_ONCE = True
        fused = False
    
    if not fused and dilation != 1:
        raise ValueError("Non-Fused Conv2d, Only Supports Dilation of 1")
        
    ### Get Input/Output Shapes ###
    B, C_in,  H, W = input.data.shape
    C_out, _, K, K_w = weight.data.shape
    S,P = stride, padding

    H_out = (H + 2*P - K)//S + 1
    W_out = (H + 2*P - K)//S + 1
    
    if not fused:

        ### Get Backend ###
        xp = input.xp

        ### Pad Data If Padding is set ###
        if P > 0:
            x_padded = np.pad(input.data, ((0,0), (0,0), (P,P), (P,P)), mode='constant')
        else:
            x_padded = input.data

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
        weights_flat = weight.data.reshape(C_out, -1).T

        ### Forward ###
        output = input.xp.empty((cols_flat.shape[0], weights_flat.shape[1]))
        np.matmul(cols_flat, weights_flat, out=output)
        if bias is not None:
            np.add(output, bias.data, out=output)

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

    else:

        assert "cuda" in input.device, "Fused Operations can only be performed on Cuda Tensors!"
            
        ### Fused ops need raw cupy arrays not Array ###
        if hasattr(input.data, "_array"):
            input_array = input.data._array
        else:
            input_array = input.data

        if hasattr(weight.data, "_array"):
            weight_array = weight.data._array
        else:
            weight_array = weight.data

        if bias is not None:
            if hasattr(bias.data, "_array"):
                bias_array = bias.data._array
            else:
                bias_array = bias.data

        output = FO.fused_conv2d_forward(input_array, weight_array, bias_array, 
                                          stride, padding, dilation=dilation)

        def _conv2d_backward(grad_output):
            
            grads = FO.fused_conv2d_backward(grad_output, input_array, weight_array, 
                                             bias_array, H, W, K, K_w, 
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

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, fused=False):

    """
    Almost identical to conv2d, just reduced a dimension
    """
    
    if fused and not FUSED_AVAIL:
        if not FLAG_ONCE:
            warnings.warn("Fused Ops not available, defaulting to normal ops, install Triton for Fused Operations!")
            FLAG_ONCE = True
        fused = False
    
    if not fused and dilation != 1:
        raise ValueError("Non-Fused Conv2d, Only Supports Dilation of 1")
    
    if not fused:
        ### Get Backend ###
        xp = input.xp

        ### Get Input/Output Shapes ###
        B, C_in,  L_in = input.data.shape
        C_out, _, K = weight.data.shape
        S,P = stride, padding

        L_out = (L_in + 2*P - K)//S + 1

        ### Pad Data If Padding is set ###
        if P > 0:
            x_padded = np.pad(input.data, ((0,0), (0,0), (P,P)), mode='constant')
        else:
            x_padded = input.data

        ### Use stride tricks for efficient im2col ###
        
        shape = (B, C_in, K, L_out)
        strides = (
            x_padded.strides[0], # Number of bits to move to get to next batch
            x_padded.strides[1], # Number of bits to move to get to next channel
            x_padded.strides[2], # Number of bits to move to get to next row in kernel
            S*x_padded.strides[2], # Number of bits to move to get to next col in kernel
        )

        ### Grab Strided View of our Data (no extra copy needed!) ###
        cols = xp.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

        cols_flat = cols.reshape(B*L_out, -1)
        weights_flat = weight.data.reshape(C_out, -1).T

        ### Forward ###
        output = input.xp.empty((cols_flat.shape[0], weights_flat.shape[1]))
        np.matmul(cols_flat, weights_flat, out=output)
        if bias is not None:
            np.add(output, bias.data, out=output)

        #### Reshape back to (B x C_out x H_out x W_out) ###
        output = output.reshape(B, L_out, C_out).transpose(0, 2, 1)

        def _conv1d_backward(grad_output):
            grad_output_flat = grad_output.transpose(0,2,1).reshape(B*L_out, C_out)

            # Gradient w.r.t. weight
            if weight.requires_grad:
                grad_W = xp.matmul(cols_flat.T, grad_output_flat)
                weight.grad = grad_W.T.reshape(C_out, C_in, K) if weight.grad is None else weight.grad + grad_W.T.reshape(C_out, C_in, K)

            # Gradient w.r.t. bias
            if bias is not None and bias.requires_grad:
                grad_b = grad_output_flat.sum(axis=0)
                bias.grad = grad_b if bias.grad is None else bias.grad + grad_b

            # Gradient w.r.t. input
            if input.requires_grad:
                grad_cols_flat = xp.matmul(grad_output_flat, weights_flat.T)  # (B*L_out, C_in*K)
                grad_cols = grad_cols_flat.reshape(B, L_out, C_in*K).transpose(0,2,1)  # (B, C_in*K, L_out)

                grad_input = xp.zeros_like(x_padded)
                # Compute indices for scatter-add
                i0 = xp.repeat(xp.arange(K), C_in)
                k = xp.tile(xp.arange(C_in), K)
                i1 = S * xp.repeat(xp.arange(L_out), 1)

                i = i0.reshape(-1,1) + i1.reshape(1,-1)
                kk = xp.tile(k.reshape(-1,1), (1,L_out))

                # Flatten for batch
                bb = xp.repeat(xp.arange(B), C_in*K*L_out)
                ii = xp.tile(i.flatten(), B)
                kk_flat = xp.tile(kk.flatten(), B)
                vals = grad_cols.flatten()

                xp.add.at(grad_input, (bb, kk_flat, ii), vals)

                if P > 0:
                    grad_input = grad_input[:, :, P:-P]

                if input.grad is None:
                    input.grad = grad_input 
                else:
                    input.grad += grad_input

    else:

        assert "cuda" in input.device, "Fused Operations can only be performed on Cuda Tensors!"
            
        ### Fused ops need raw cupy arrays not Array ###
        if hasattr(input.data, "_array"):
            input_array = input.data._array
        else:
            input_array = input.data

        if hasattr(weight.data, "_array"):
            weight_array = weight.data._array
        else:
            weight_array = weight.data

        if bias is not None:
            if hasattr(bias.data, "_array"):
                bias_array = bias.data._array
            else:
                bias_array = bias.data

        output = FO.fused_conv1d_forward(input_array, weight_array, bias_array, 
                                         stride, padding, dilation=dilation)

        def _conv1d_backward(grad_output):
            
            grads = FO.fused_conv1d_backward(grad_output, input_array, weight_array, 
                                             bias_array, input_array.shape[1], weight_array.shape[1], 
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
        grad_fn=_conv1d_backward if requires_grad else None,
        grad_fn_name="<Conv1dBackward>" if requires_grad else None
    )

    
    if requires_grad:
        output._add_parents(input, weight, bias)
        
    return output       

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
    cols_flat = xp.empty((B*H_in*W_in, C_out*K*K))
    xp.matmul(input_flat, weights_flat, out=cols_flat)

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
            grad_weights_flat = xp.empty((C_in, C_out * K * K), dtype=weight.dtype)
            xp.matmul(input_flat.T, grad_cols_flat, out=grad_weights_flat)

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
            grad_input_flat = xp.empty((B * H_in * W_in, C_in))
            xp.matmul(grad_cols_flat, weights_flat.T, out=grad_input_flat)

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

def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0):
    """
    Exactly the same as the conv_transpose2d, just with less dimensions!
    """

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
    cols_flat = xp.empty((B*L_in, C_out*K))
    xp.matmul(input_flat, weights_flat, out=cols_flat)

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
            grad_weights_flat = xp.empty((C_in, C_out*K), dtype=weight.dtype)
            xp.matmul(input_flat.T, grad_cols_flat, out=grad_weights_flat)
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
            grad_input_flat = xp.empty((B*L_in, C_in))
            xp.matmul(grad_cols_flat, weights_flat.T, out=grad_input_flat)
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

def maxpool2d(input, kernel_size, stride=None, padding=0):
    
    """
    MaxPool2d using im2col + argmax. Supports manual backward.

    input: Tensor of shape (B, C, H, W)
    kernel_size: int or tuple
    stride: int or tuple
    padding: int
    """

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

def averagepool2d(input, kernel_size, stride=None, padding=0):
    """
    AveragePool2d using im2col. Supports manual backward.

    Args:
        input: Tensor of shape (B, C, H, W)
        kernel_size: int or tuple
        stride: int or tuple
        padding: int
    """

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

def embedding(indices, weight):
    """
    Standard indexing op to get embeddings for the indexes we want

    No need for "auto" here, __getitem__ implemented in Tensor class
    """
    return weight[indices]

def dropout(input, dropout_p, training=True, auto=False):

    if not training or dropout_p == 0.0:
        return input

    ### Sample Mask ###
    mask = (input.xp.random.random_sample(input.data.shape) >= dropout_p).astype(input.dtype)
    ratio = 1 / (1 - dropout_p)
    mask *= ratio

    if auto:
        return input * mask

    else:

        out_data = input.data * mask
        
        # Backward function only needs the mask (not full input_tensor)
        def _dropout_backward(input_grad):
            if input.requires_grad:
                self_grad = input_grad * mask
                if input.grad is None:
                    input.grad = self_grad
                else:
                    input.grad += self_grad

        # Attach to computation graph if needed
        requires_grad = input.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_dropout_backward if requires_grad else None,
            grad_fn_name="<DropoutBackward>" if requires_grad else None,
        )

        if requires_grad:
            out._add_parents(input)

        return out
   
def layernorm(input, weight, bias, eps=1e-5, training=True, auto=False, fused=False):

    """
    Standard LayerNorm op with input of the shape (*, E)
    
    gamma -> (E,)
    beta -> (E,)

    """
    
    reshaped = False
    *dims, embed_dim = input.shape

    ### If we have more than 1 dim, we have to flatten ###
    if len(dims) > 1:
        reshaped = True

    if auto:
        
        if reshaped:
            input = input.reshape(-1, embed_dim)
        
        var_x = (input.var(dim=-1, keepdims=True) + eps)
        norm_x = (input - input.mean(dim=-1, keepdims=True)) / var_x**0.5
        scale_shifted_x = norm_x * weight.reshape(1,-1) 
        
        if bias:
            scale_shifted_x = scale_shifted_x + bias.reshape(1,-1)

        if reshaped:
            scale_shifted_x = scale_shifted_x.reshape(*dims, embed_dim)

        return scale_shifted_x

    else:

        if fused and not FUSED_AVAIL:
            if not FLAG_ONCE:
                warnings.warn("Fused Ops not available, defaulting to normal ops, install Triton for Fused Operations!")
                FLAG_ONCE = True
            fused = False

        if not fused:
            input_cp = input.data
            gamma_cp = weight.data

            if bias is not None:
                beta_cp = bias.data

            if reshaped:
                input_cp = input_cp.reshape(-1, embed_dim)
            
            ### Compute Mean and Var Along Last Dimension ###
            mean = np.mean(input_cp, axis=-1, keepdims=True)
            var = np.var(input_cp, axis=-1, keepdims=True)
            inv_std = np.reciprocal(np.sqrt(var + eps))

            ### Store copy of x_hat for the input backward ###
            x_hat = (input_cp - mean) * inv_std
            
            output = np.empty_like(x_hat)
            np.multiply(x_hat, gamma_cp.reshape(1,-1), out=output)
            if bias is not None:
                np.add(output, beta_cp.reshape(1,-1), out=output)

            ### Reshape Back if Needed ###
            output = output.reshape(*dims, embed_dim)

            def _layernorm_backward(grad_output):
                
                ### Reshape Grad Output as its currently (*, I) ###
                if reshaped:
                    grad_output = grad_output.reshape(-1, embed_dim)

                if weight.requires_grad:
                    # y = x_hat * gamma + beta
                    # dL/dgamma = dL/dy * dy/dgamma = grad_output * x_hat
                    # sum up grads over the batch dim
                    grad_gamma = np.sum(grad_output * x_hat, axis=0)

                    if weight.grad is None:
                        weight.grad = grad_gamma
                    else:
                        weight.grad += grad_gamma
                
                if bias is not None:
                    if bias.requires_grad:
                        # y = x_hat * gamma + beta
                        # dL/dbeta = dL/dy * dy/dbeta = grad_output * 1
                        # sum up grads over the batch dim
                        grad_beta = np.sum(grad_output, axis=0)
                        
                        if bias.grad is None:
                            bias.grad = grad_beta
                        else:
                            bias.grad += grad_beta

                if input.requires_grad:
                    # y = x_hat * gamma + beta
                    # where x_hat = (x - mu) / (var + eps)
                    # dL/dx = dL/dy * dy/dx_hat * dx_hat / dx
                    # = inv_std * (grad_output * gamma - mean(grad_output*gamma) - x_hat*mean(grad_output * gamma_cp * x_hat))
                    # sum up grads over the batch dim
                    dx_hat = grad_output * gamma_cp
                    mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
                    mean_mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
                    grad_input = inv_std * (dx_hat - mean_dx_hat - x_hat * mean_mean_dx_hat_x_hat) 

                    ### Put Back into Original Shape ###
                    if reshaped:
                        grad_input = grad_input.reshape(*dims, embed_dim)

                    if input.grad is None:
                        input.grad = grad_input
                    else:
                        input.grad += grad_input

        else:

            assert "cuda" in input.device, "Fused Operations can only be performed on Cuda Tensors!"
            
            ### Fused ops need raw cupy arrays not Array ###
            if hasattr(input.data, "_array"):
                input_array = input.data._array
            else:
                input_array = input.data

            if hasattr(weight.data, "_array"):
                weight_array = weight.data._array
            else:
                weight_array = weight.data

            if bias is not None:
                if hasattr(bias.data, "_array"):
                    bias_array = bias.data._array
                else:
                    bias_array = bias.data

            ### Flatten Input Array ###
            *dims, embed_dim = input_array.shape
            flat_input_array = input_array.reshape(-1, embed_dim)

            outputs = FO.fused_layernorm_forward(flat_input_array,
                                                 gamma=weight_array, 
                                                 beta=bias_array if bias is not None else None, 
                                                 eps=eps, 
                                                 training=training)
            
            ### during training we return intermediate tensors ###
            if training:
                output_flat, x_hat, inv_var = outputs
            else:
                output_flat = outputs

            ### Return y back to (B x S x E) ###
            output = output_flat.reshape(*dims, -1)

            def _layernorm_backward(grad_output):
                
                ### Reshape grad back to (*xE) ###
                grad_flat = grad_output.reshape(-1, embed_dim)

                grads = FO.fused_layernorm_backward(x_hat=x_hat,
                                                    inv_var=inv_var,
                                                    dy=grad_flat,
                                                    gamma=weight_array, 
                                                    bias=True if bias is not None else False)
    
                if bias is not None:
                    dx, dgamma, dbeta = grads
                else:
                    dx, dgamma = grads

                ### Reshape dx back to original shape ###
                dx = dx.reshape(*dims, -1)

                ### Accumulate Grads ####
                if input.grad is None:
                    input.grad = dx
                else:
                    input.grad += dx

                if weight.grad is None:
                    weight.grad = dgamma
                else:
                    weight.grad += dgamma
                
                if bias is not None:
                    if bias.grad is None:
                        bias.grad = dbeta
                    else:
                        bias.grad += dbeta

        requires_grad = input.requires_grad or weight.requires_grad or \
                            (bias is not None and bias.requires_grad)
        requires_grad = requires_grad and Tensor.build_graph_enabled()
        output = Tensor(
            output, 
            requires_grad=requires_grad,
            grad_fn=_layernorm_backward if requires_grad else None, 
            grad_fn_name="<LayerNormBackward>" if requires_grad else None
        )

        if requires_grad:
            output._add_parents(input, weight, bias)

        return output

def batchnorm(input, weight, bias, 
              running_mean, running_var, momentum=0.1, 
              eps=1e-5, training=True):
    
    """
    BatchNorm for input of shape (N, C, *), normalizing per-channel.

    gamma: (C,)
    beta: (C,)
    running_mean: (C,)
    running_var: (C,)
    """

    ### Get Backend ###
    xp = input.xp 

    N, C, *dims = input.shape
    reshaped = len(dims) > 0

    input_xp = input.data 
    weight_xp = weight.data.reshape(1,C,1)
    bias_xp = bias.data.reshape(1,C,1)

    ### Flatten Spatal Dims ###
    x = input_xp.reshape(N,C,-1)

    if training:
        mean = xp.mean(x, axis=(0, 2), keepdims=True)
        var = xp.var(x, axis=(0, 2), keepdims=True)
        running_mean.data[:] = (1 - momentum) * running_mean.data + momentum * mean.squeeze()
        running_var.data[:] = (1 - momentum) * running_var.data + momentum * var.squeeze()

    else:
        mean = running_mean.data.reshape(1, C, 1)
        var = running_var.data.reshape(1, C, 1)

    inv_std = xp.reciprocal(xp.sqrt(var + eps))
    norm_x = (x - mean) * inv_std
    out_data = norm_x * weight_xp + bias_xp

    if reshaped:
        out_data = out_data.reshape(N,C,*dims)

    def _batchnorm_backward(grad_output):

        ### Reshape Grad from (N,C,*) to (N,C,-1) ###
        grad_output = grad_output.reshape(N,C,-1)

        if weight.requires_grad:
            grad_gamma = xp.sum(grad_output * norm_x, axis=(0, 2))
            if weight.grad is None:
                weight.grad = grad_gamma
            else:
                weight.grad += grad_gamma

        if bias.requires_grad:
            grad_beta = xp.sum(grad_output, axis=(0, 2))
            if bias.grad is None:
                bias.grad = grad_beta
            else:
                bias.grad += grad_beta

        if input.requires_grad:
            grad_norm = grad_output * weight_xp

            mean_grad = xp.mean(grad_norm, axis=(0,2), keepdims=True)
            mean_norm_grad = xp.mean(grad_norm * norm_x, axis=(0,2), keepdims=True)
            grad_input = (grad_norm - mean_grad - norm_x * mean_norm_grad) * inv_std

            ### Put Back into Original Shape ###
            if reshaped:
                grad_input = grad_input.reshape(N,C,*dims)
            else:
                grad_input = grad_input.reshape(N,C)
            
            if input.grad is None:
                input.grad = grad_input
            else:
                input.grad += grad_input

    requires_grad = input.requires_grad or weight.requires_grad or bias.requires_grad
    requires_grad = requires_grad and Tensor.build_graph_enabled()
    output = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_batchnorm_backward if requires_grad else None,
        grad_fn_name="<BatchNormBackward>" if requires_grad else None
    )

    if requires_grad:
        output._add_parents(input, weight, bias)
    
    return output

def sigmoid(x, auto=False):

    if auto:
        return 1 / (1 + (-x).exp())
    else:
        x_cp = x.data
        out_data = 1 / (1 + np.exp(-x_cp))

        def _sigmoid_backward(grad_output):
            grad_input = grad_output * out.data * (1 - out.data)
            if x.grad is None:
                x.grad = grad_input
            else:
                x.grad += grad_input
        
        requires_grad = x.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data, 
            requires_grad=requires_grad,
            grad_fn=_sigmoid_backward if requires_grad else None, 
            grad_fn_name="<SigmoidBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(x)

        return out

def relu(input, auto=False):

    if auto:
        mask = Tensor(np.where(input.data < 0, 0, 1).astype(input.dtype))
        return input * mask
    else:
  
        input.data[input.data < 0] = 0

        def _relu_backward(input_grad):
            if input.requires_grad:
                grad_input = input_grad * (input.data > 0)

                if input.grad is None:
                    input.grad = grad_input
                else:
                    input.grad += grad_input

        requires_grad = input.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            input.data,
            requires_grad=requires_grad,
            grad_fn=_relu_backward if requires_grad else None,
            grad_fn_name="<ReLUBackward>" if requires_grad else None,
            device=input.device, 
            dtype=input.dtype
        )

        if requires_grad:
            out._add_parents(input)

        return out
    
def gelu(x):
    
    """
    gelu as described in https://arxiv.org/pdf/2305.12073

    Forward method is Equation 24
    Backward methdo is Equation 42-43
    """

    data = x.data

    # Constants
    sqrt_2_over_pi = 0.7978845 # xp.sqrt(2 / xp.pi).astype(x.data.dtype)
    coeff = 0.44715

    #inner = sqrt_2_over_pi * (x + coeff * x^3)
    x_squared = np.power(data, 2)
    x_cubed = x_squared * data

    inner = sqrt_2_over_pi * (data + coeff * x_cubed)

    ### Tanh out = tanh(inner) ###
    tanh_out = np.tanh(inner)
    out_data = 0.5 * data * (1.0 + tanh_out)

    # Backward
    def _gelu_backward(grad_output):

        if x.requires_grad:
    
            inner_grad = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_squared)

            # derivative of GELU approximation (sech^2(x) = 1 - tanh^2(x))
            sech2 = 1 - np.power(tanh_out, 2)  # derivative of tanh

            grad_input = 0.5 * (1.0 + tanh_out + data * sech2 * inner_grad) * grad_output

            if x.grad is None:
                x.grad = grad_input
            else:
                x.grad += grad_input

    requires_grad = x.requires_grad and Tensor.build_graph_enabled()
    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        grad_fn=_gelu_backward if requires_grad else None,
        grad_fn_name="<GELUBackward>" if requires_grad else None
    )

    if requires_grad:
        out._add_parents(x)

    return out

def softmax(x, dim=-1, auto=False, fused=False):

    if auto:

        max_x = x.max(dim=dim, keepdims=True)
        x_shifted = x - max_x
        exp_x = x_shifted.exp()
        sum_exp = exp_x.sum(dim=dim, keepdims=True)
        return exp_x / sum_exp
    
    else:
        
        if fused and not FUSED_AVAIL:
            if not FLAG_ONCE:
                warnings.warn("Fused Ops not available, defaulting to normal ops, install Triton for Fused Operations!")
                FLAG_ONCE = True
            fused = False

        if not fused:
            # Numerical stability: subtract max along dim
            max_val = np.max(x.data, axis=dim, keepdims=True)
            shifted = x.data - max_val
            exp_x = np.exp(shifted)
            sum_exp = np.sum(exp_x, axis=dim, keepdims=True)
            out_data = exp_x / sum_exp

            # Define manual backward
            def _softmax_backward(grad_output):

                if x.requires_grad:
                    # Softmax derivative: grad_input = s * (grad - sum(grad*s))
                    # s = out_data
                    sum_grad_s = np.sum(grad_output * out_data, axis=dim, keepdims=True)
                    grad_input = out_data * (grad_output - sum_grad_s)
                    
                    if x.grad is None:
                        x.grad = grad_input
                    else:
                        x.grad += grad_input
        else:
            
            assert "cuda" in x.device, "Fused Operations can only be performed on Cuda Tensors!"

            ### Fused Ops Need Access to Raw Arrays and they must be on CUDA ###
            if hasattr(x.data, "_array"):
                array = x.data._array
            else:
                array = x.data
            
            orig_shape = array.shape
            ndim = len(orig_shape)

            # grabs the dim we want. If we have ndim=4 and we want to take softmax #
            # over dim=2, then 2%4 is just 2. But if we say -1, then -1 % 4 is 3. #
            dim = dim % ndim 

            ### Permute so target dim is last ###
            if dim != ndim - 1:

                ### Put all other dimensions first ###
                permute_axis = [i for i in range(ndim) if i != dim] + [dim]
                array_perm = array.transpose(permute_axis)
            
            else:
                array_perm = array

            ### Flatten to (*,I) ###
            n_rows = int(np.prod(array_perm.shape[:-1]))
            n_cols = array_perm.shape[-1]
            reshaped = array_perm.reshape(n_rows, n_cols)
            
            ### Fused Softmax ###
            out_flat = FO.fused_softmax_forward(reshaped)

            ### Reshape Back ###
            out_perm = out_flat.reshape(array_perm.shape)
            
            if dim != ndim - 1:
                inv_permute = np.argsort(permute_axis)
                out_data = out_perm.transpose(inv_permute)
            else:
                out_data = out_perm
        
            def _softmax_backward(input_grad):

                # Extract raw array
                if hasattr(input_grad, "_array"):
                    grad_array = input_grad._array
                else:
                    grad_array = input_grad

                # Get shapes
                orig_shape = grad_array.shape
                ndim = len(orig_shape)
                dim_idx = dim % ndim  # support negative dims

                # Flatten our grad and out data ###
                if dim_idx != ndim - 1:
                    permute_axes = [i for i in range(ndim) if i != dim_idx] + [dim_idx]
                    grad_perm = grad_array.transpose(permute_axes)
                    out_perm = out_data.transpose(permute_axes)  # permute softmax output similarly
                else:
                    grad_perm = grad_array
                    out_perm = out_data

                n_rows = int(np.prod(grad_perm.shape[:-1]))
                n_cols = grad_perm.shape[-1]
                grad_flat = grad_perm.reshape(n_rows, n_cols)
                out_flat = out_perm.reshape(n_rows, n_cols)

                ### Fused Backward Op ###
                grad_input_flat = FO.fused_softmax_backward(grad_flat, out_flat)

                # Step 4: Reshape back
                grad_input_perm = grad_input_flat.reshape(grad_perm.shape)

                if dim_idx != ndim - 1:
                    inv_permute = np.argsort(permute_axes)
                    grad_input = grad_input_perm.transpose(inv_permute)
                else:
                    grad_input = grad_input_perm

                if x.grad is None:
                    x.grad = grad_input
                else:
                    x.grad += grad_input

        requires_grad = x.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_softmax_backward if requires_grad else None,
            grad_fn_name="<SoftmaxBackward>" if requires_grad else None
        )

        # Add child to autograd graph
        if requires_grad:
            out._add_parents(x)

        return out

def cross_entropy(logits, targets, ignore_index=-100, auto=False, fused=False):

    """
    Standard cross entropy loss between raw logits and targets

    logits: (* x num_classes)
    targets (*, )
    ignore_index: If a label is -100 it wont contribute to the loss

    For precision, we will compute the loss at fp32 and then cast back to fp16 if needed!
    """

    ### Flatten Logits to be (*, num_classes) ###
    *other_dims, num_classes = logits.shape

    ### Get total flattened dimension ###
    flattened_dim = np.prod(other_dims)

    ### Make sure targets are always int32 ###
    targets = targets.astype("int32")
    
    if auto:
        
        ### Flatten Logits ###
        logits = logits.reshape(flattened_dim, num_classes)

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
    
    else:

        if fused and not FUSED_AVAIL:
            if not FLAG_ONCE:
                warnings.warn("Fused Ops not available, defaulting to normal ops, install Triton for Fused Operations!")
                FLAG_ONCE = True
            fused = False

        if not fused:
            # Our Logits will be some (N x NUM_CLASSES)
            # Our Targets will be some (N, ) where each value in the target 
            # is between 0 and NUM_CLASSES-1

            # Cross Entropy Formula w. Softmax together was just:

            # CE = log(sum(e^x)) = x_{correct}

            # And we know the index of correct, its just our label the cooresponding labels. 
            # So lets just write a kernel that processes one row at a time. We will grab the 
            # NUM_CLASSES length vector of logits and the single label 
            
            logits_data = logits.data.reshape(flattened_dim, num_classes).astype("float32")
            targets_data = targets.data.reshape(flattened_dim)

            # logits_data = logits.xp.ascontiguousarray(logits_data, dtype=logits.dtype)
            mask = (targets_data != ignore_index)
            valid_counts = mask.sum()

            # Stable logsumexp per row
            logits_max = np.max(logits_data, axis=1, keepdims=True)
            exp_shifted = np.exp(logits_data - logits_max)  # shape (B, C)
            logsumexp = np.log(np.sum(exp_shifted, axis=1, keepdims=True)) + logits_max  # shape (B, 1)

            # Negative log-likelihood only for valid rows
            nll = (logsumexp.flatten() - logits_data[np.arange(flattened_dim), targets_data]) * mask
            loss_value = np.sum(nll) / valid_counts

            loss_value = loss_value.astype(logits.dtype)
            
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
                    grad_input = grad_input.reshape(logits.shape).astype(logits.dtype)

                    if logits.grad is None:
                        logits.grad = grad_input
                    else:
                        logits.grad += grad_input
        
        else:
            
            assert "cuda" in logits.device, "Fused Operations can only be performed on Cuda Tensors!"

            ### Fused Op only happens in float32 ###
            logits_data = logits.data.reshape(flattened_dim, num_classes).astype("float32")

            ### Triton kernel expects long tensors (int64) labels ###
            targets_data = targets.data.reshape(flattened_dim).astype("int64")

            targets_flat = targets_data
            mask = (targets_flat != ignore_index)
            valid_counts = mask.sum()
            
            # Triton kernel forward
            loss_cp, logsumexp_cp = FO.fused_cross_entropy_forward(logits_data, targets_data)

            loss_value = loss_cp.sum() / valid_counts

            loss_value = loss_value.astype(logits.dtype)

            def _cross_entropy_backward(grad_output):
      
                ### The loss is the last thing in our model ###
                ### so our upstream grad is just a bunch of ones so nothing ###
                ### to really use here! ###
                if logits.requires_grad:
                    grad_cp = FO.fused_cross_entropy_backward(
                        logits_data,
                        targets_data,
                        logsumexp_cp
                    ) 

                    grad_cp *= (grad_output / valid_counts) 

                    # Reshape back to original logits shape and dtype
                    grad_input = grad_cp.reshape(*logits.shape).astype(logits.dtype)

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

def mse_loss(pred, labels, auto=False):

    if auto:
        return ((pred - labels)**2).mean()

    else:

        diff = pred.data - labels.data
        out_data = (diff**2).mean()

        def _mse_backward(grad_output):

            N = diff.shape[0]
            grad_input = (2.0 / N) * diff * grad_output

            if pred.grad is None:
                pred.grad = grad_input
            else:
                pred.grad += grad_input
        
        requires_grad = pred.requires_grad and Tensor.build_graph_enabled()
        out = Tensor(
            out_data,
            requires_grad=requires_grad,
            grad_fn=_mse_backward if requires_grad else None,
            grad_fn_name="<MSEBackward>" if requires_grad else None
        )

        if requires_grad:
            out._add_parents(pred)
        
        return out

def scaled_dot_product_attention(Q, K, V, causal=False, softmax_scale=None):
    
    if not FUSED_AVAIL:
        raise Exception("Fused ops not available, install Triton!!!")
    
    Q_data = Q.data
    K_data = K.data
    V_data = V.data

    ### If we are an Array drill down to ndarray ###
    if hasattr(Q_data, "_array"):
        Q_data = Q_data._array
    if hasattr(K_data, "_array"):
        K_data = K_data._array
    if hasattr(V_data, "_array"):
        V_data = V_data._array
    
    assert (
        Q_data.shape == K_data.shape == V_data.shape
    ), f"Shapes mismatch: Q={Q_data.shape}, K={K_data.shape}, V={V_data.shape}"
    assert len(Q_data.shape) == 4, f"Expected 4D tensors, got {len(Q_data.shape)}D"

    Q_data, K_data, V_data, attn_out, M = FO.fused_sdpa_forward(
        Q_data, K_data, V_data, 
        causal=causal, softmax_scale=softmax_scale
    )

    def _sdpa_backward(grad_output):
  
        dQ, dK, dV = FO.fused_sdpa_backward(grad_output, 
                                            Q_data, K_data, V_data, 
                                            attn_out, M, 
                                            causal=causal,
                                            softmax_scale=softmax_scale)

        ### Cast grads back to original dtype ###
        dQ = dQ.astype(Q.dtype)
        dK = dK.astype(K.dtype)
        dV = dV.astype(V.dtype)

        if Q.grad is None:
            Q.grad = dQ
        else:
            Q.grad += dQ

        if K.grad is None:
            K.grad = dK
        else:
            K.grad += dK

        if V.grad is None:
            V.grad = dV
        else:
            V.grad += dV

    requires_grad = Q.requires_grad or K.requires_grad or V.requires_grad
    requires_grad = requires_grad and Tensor.build_graph_enabled()

    out = Tensor(
        attn_out,
        requires_grad=requires_grad,
        grad_fn=_sdpa_backward if requires_grad else None,
        grad_fn_name="<SDPABackward>" if requires_grad else None,
        dtype=Q.dtype
    )
    
    if requires_grad:
        out._add_parents(Q, K, V)

    return out
