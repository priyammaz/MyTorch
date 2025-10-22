"""
Convolutions can be written as MatMul with the Im2Col Algorithm

This implementation is greatly based on awesome implementation here!
https://github.com/l1351868270/ld_triton/blob/main/ld_triton/ops/convolution/triton_conv2d.py

We will be using Grouped MatMul just like we saw in our `fused_ops/matmul.py` so make sure you
understand that before continuing here!
"""
import triton
import triton.language as tl
import torch
import pytest

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8},
                      num_stages=5, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['B', 'C_in', 'H_in', 'W_in', 'C_out', 'H_out', 'W_out', 'K_h', 'K_w', 
         'str_h', 'str_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def _implicit_gemm_conv2d_fwd_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr, 
    B, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    
    """
    Convolutions are just MatMul if we use the Im2Col Algorithm. What we want to do is:

    1) We have a weight matrix (C_out, C_in, K_h, K_w)
    2) We have data matrix (B, C_in, H, W)

    And we want to matmul the different locations in our images with our filter. This means, 

    Grab a patch of our image (same shape as our Kernel)

    Image Patch: (B, C_in, K_h, K_w)

    Reshape it for Matmul (B, C_in*K_h*K_w)

    Reshape our Weights for Matmul: (C_in*K_h*K_w, C_out)

    Do the MatMul: (B, C_in*K_h*K_w) @ (C_in*K_h*K_w, C_out) = (B, C_out)

    What this has done is taken the entire patch of our original images, 
    flattend the patch to C_in*K_h*K_w, and then projected it to the number of output channels that
    we actually want! Thus a standard Matmul!

    Issue: We dont have one patch we want to process, we have a bunch of them! We want to slide our kernel
    across all the patches in our image and repeat this matmul. So either we can do a For Loop, OR we can use
    a GPU to parallelize this!


    ### Im2Col Algorithm ###

    The Im2Col algorithm is pretty simple. Identify the patches you want, grab all of them and place them in a
    matrix. For example if our images are in the standard shape of (B, C_in, H, W), the patches are all the smal
    chunks of the H,W dimension that we want. Well we know everything. We know our Kernel size, stride, etc...

    So we can do the following:

    Image (1x4x4)

    [i11, i12, i13, i14]
    [i21, i22, i23, i24]
    [i31, i32, i33, i34]
    [i41, i42, i43, i44]

    Kernel (3x3) with stride of 1

    [w11 w12 w13]
    [w21 w22 w23]
    [w31 w32 w33]

    So we want to apply our kernel to the image as the following (standard multiply accumulate op). 
    These should make you think of Dot Products!

    Top left patch:
       | [i11, i12, i13]   [w11 w12 w13] |
    sum| [i21, i22, i23] * [w21 w22 w23] |
       | [i31, i32, i33]   [w31 w32 w33] |

    Top right patch:
       | [i12, i13, i14]   [w11 w12 w13] |
    sum| [i22, i23, i24] * [w21 w22 w23] |
       | [i32, i33, i34]   [w31 w32 w33] |

    Bottom left patch:
       | [i21, i22, i23]   [w11 w12 w13] |
    sum| [i31, i32, i33] * [w21 w22 w23] |
       | [i41, i42, i43]   [w31 w32 w33] |

    Bottom right patch:
       | [i22, i23, i24]   [w11 w12 w13] |
    sum| [i32, i33, i34] * [w21 w22 w23] |
       | [i42, i43, i44]   [w31 w32 w33] |


    Now this method would require us to loop through the data. So can we just
    create a matrix and do a single Matmul? Yes! That is the Im2Col Algorithm. Lets
    just grab each patch and flatten them and store them in one giant matrix. In
    this case we just have 4 patches, but you can imagine that for large images
    we will have more! 

                    (4 x 9)                                     (9 x 1)
    
    [i11, i12, i13, i21, i22, i23, i31, i32, i33]
    [i12, i13, i14, i22, i23, i24, i32, i33, i34] @ [w11 w12 w13 w21 w22 w23 w31 w32 w33]
    [i21, i22, i23, i31, i32, i33, i41, i42, i43] 
    [i22, i23, i24, i32, i33, i34, i42, i43, i44]

    Issue: Although this will totally work, this requires us to create a massive copy of our 
    data in memory. We want to do this, but without creating an explicit Im2Col array. That is
    why this is an Imlicit Gemm, you can read more here: 
    https://docs.nvidia.com/cutlass/media/docs/cpp/implicit_gemm_convolution.html#implicit-gemm-algorithm


    ### SETUP ###

    We want to setup our problem as MatMul, BUT without creating a giant Im2Col matrix. This means that we need to do
    imagine our problem as matmul BUT dynamically load the correct indexes from our images and weights.  

    
    STEP 1:
    We know the output size of a convolution given the input:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        dil = dilation
        str = stride

        H_out = (H_in + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
        W_out = (W_in + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    This means our output image will have H_out * W_out total pixels that were reduced from
    H_out*W_out number of patches from our original image

    STEP 2: Setup the Matmul

    The matmul we will be doing (implicitly) is:

    (B*H_out*W_out x C_in*K_h*K_w) @ (C_in*K_h*K_w x C_out) = (B*H_out*W_out x C_out)

    But of course to do matmul we need to use our Grouped Matmul for maximum efficiency! This means 
    each thread will be responsible to computing a block of our output matrix!

    Specifically, [BLOCK_SIZE_M x BLOCK_SIZE_N] at a time!

    But in our matmul, O = A@B we knew based on the block of our output, which rows from A and which 
    columns from B we needed. Now we dont. We need to create the rows of our Im2Col algorithm 
    dynamically based on which block we are processing. And similarly, we need to grab the columns of our
    weight matrix dynamically! But once we have them we are back to normal Matmul! 

    Step 3: 

    Our output is in the shape of (B, C_out, H_out, W_out). We need to take our computed reduction for this
    block, (where the block refers to a chunk of our B*H_out*W_out x C_out matrix) and store it in the 
    cooresponding block in our B, C_out, H_out, W_out. This again requires some indexing to do correctly!

    """

    ### Grouping logic as described in fused_ops/matmul.py! ###
    ### We want threads to launch blocks close to each other at similar times ###
    ### so we can leverage already cached data! ###
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    ### Now get our offsets for the Block of the Matmul we want to compute ###
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) # which rows of our im2col mat do we want
    offs_channel = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # which cols of our weight matrix do we want (which channel)

    ### Remember that our matrix we are doing matmul between is ###
    ### (B*H_out*W_out x C_in*K_h*K_w) @ (C_in*K_h*K_w x C_out) = (B*H_out*W_out x C_out)
    
    ### But we are only processing a specific block of (B*H_out*W_out x C_out). 
    ### And we have the top left starting index of the block now from pid_m/pid_n
    ### and have computed the offsets for this block.

    ### The issue: Our input data is (B x C_in x H_in x W_in)
    ###            out input weight is (C_out, C_in, K_h, K_w)
    ### computing a block of (B*H_out*W_out x C_out) means 
    ### we need to find the corresponding:
    ### rows of (B*H_out*W_out x C_in*K_h*K_w)
    ### columns of (C_in*K_h*K_w x C_out)

    ### but because we dont have those explicitly, we need to convert it!
    ### Lets start with the first one: (B*H_out*W_out x C_out)
    ### We need to find which Bs we are on, and H_outs, W_outs we are on
    
    ### Which sample in the batch are we at? ###
    n = offs_m // (H_out*W_out)

    ### Advance to that batch so we can now get our H_out and W_out ###
    advance_to_batch = offs_m % (H_out*W_out)

    ### Within the remaining index we can figure out the output height and width index ###
    p = advance_to_batch // W_out # starting ptr output Height
    q = advance_to_batch % W_out # starting ptr to output Width

    ### Matmul is still the same, we are computing a BLOCK_SIZE_M x BLOCK_SIZE_N output ###
    ### because we are computing a block of our final matrix [NPQ x K]
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    ### Loop through the inner dimension in chunks of size BLOCK_SIZE_K ###
    ### This is over our C_in * K_h * K_w Dimensions that we are reducing over ###
    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):

        ### Get our offsets for this block of K (inner dim) ###
        ### Which chunk of our C_in * K_h * K_w  are we processing right now? ###
        offs_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))

        ### Just like earlier, we computed the indexes of n,p,q which is for our ###
        ### output location. Similarly, we need to know our our index in our weight ###
        ### that we are processing. Remember, our weights are in the shape ###
        ### [C_out x C_in x K_h x K_w] and we are treating it as [C_in*K_h*K_w x C_out]
        ### So what C, K_h, K_w are we processing in this specific iteration of the loop? ###
        c = offs_k // (K_h * K_w)
        advance_to_channel = offs_k % (K_h * K_w)
        r = advance_to_channel // K_w
        s = advance_to_channel % K_w

        ### Now we need to actual index our inputs spatial indexes (h,w) ###
        ### that coorespond to the output position (p,q) and our kernel position (r,s) ###
        ### Remember that:
        ### str_h/w is the stepsize in out input for each output pixel
        ### dil_h/w is the spacing between our kernel elements
        ### pad_h/w are just offsets to our input boundaries

        ### so if we know what p,q we are at (in the output spatial coordinate) we need to 
        ### compute what the input spatial coordinate was ###
        ### The formula for this is just h = p * stride_h + r * dilation_h - padding_h

        ### Lets do it piece by piece:

        ### STRIDING ###
        ### p * stride_h
        ### p is our position in the output, and we want its cooresponding h in the input. 
        ### lets look at it in one dimension:

        ### input: [0,1,2,3,4,5,6]
        ### kernel [w1 w2 w3]
        ### stride: 2

        ### Output will be:
        ### dot([0,1,2], [w1 w2 w3]) -> move over by 2
        ### dot([2,3,4], [w1 w2 w3]) -> move over by 2
        ### dot([4,5,6], [w1 w2 w3])

        ### And we can call this [o1 o2 o3] for the three outputs
        ### p is our index in our output matrix. 
        ### p = 0 refers to o1, which starts at [0] in our input and we just add a shift of [0,1,2] to cover all [0,1,2]
        ### p = 1 refers to o2, which starts at [2] in our input and we just add a shift of [0,1,2] to cover all [2,3,4]
        ### P = 2 refers to o3, which starts at [4] in our input and we just add a shift of [0,1,2] to cover all [4,5,6]

        ### You should see the pattern here, our to get the starting index
        ### of our input, it is proportional to the index of our output and the stride. 
        
        ### DILATION ###
        ### Dilation is the separation between the kernel. So consecutive values in our 
        ### kernel will represent spaced out values in our input (by a dilation factor)
        
        ### input: [0,1,2,3,4,5,6]
        ### kernel [w1 w2 w3]
        ### dilation: 2
        ### stride: 1

        ### Output will be:
        ### dot([0,2,4], [w1 w2 w3]) -> move over by 1
        ### dot([1,3,5], [w1 w2 w3]) -> move over by 1
        ### dot([2,4,6], [w1 w2 w3])

        ### And we can call this [o1 o2 o3] for the three outputs
        ### p is our index in our output matrix. 
        ### p = 0 refers to o1, which starts at [0] in our input and we just add a shift of [0,2,4] to cover all [0,2,4]
        ### p = 1 refers to o2, which starts at [1] in our input and we just add a shift of [0,2,4] to cover all [1,3,5]
        ### P = 2 refers to o3, which starts at [2] in our input and we just add a shift of [0,2,4] to cover all [2,4,6]

        ### You should see the pattern here! 
        ### Dilation effects how much we shift over by to access our values!

        ### PADDING ###
        ### Padding is handled a little differently

        ### input: [0,1,2,3,4,5,6]
        ### padded input (pad by 1 on each side):
        ### padded input: [PAD 0,1,2,3,4,5,6 PAD]
        ### kernel [w1 w2 w3]

        ### By default we would start at index [0,1,2], but we dont want that, we want to move
        ### back by 1 to include the pad  (so we will have w1 off the image and w2/w3 on the image)

        ### [0,1,2] - 1 = dot([-1,0,1], [w1 w2 w3])
        ### but what is -1? That is an invalid index! So what we will do is when we load our data 
        ### for indexes that are invalid, we can create a mask, and fill invalid accesses with 0. 
        ### filling with 0 is the equivalent of the standard 0 padding that we do!

        ### Now that we have all that, we have a grid of points we need to create. 

        ### p was created from offs_m thus is a vector of length BLOCK_SIZE_M. And P is a height
        ### so we can make P a column vector [BLOCK_SIZE_M x 1]

        ### r was created from our offs_k, thus is a [BLOCK_SIZE_K] vector. r is also a height for
        ### our kernel weights, so lets make this a row vector of shape [1 x BLOCK_SIZE_K]

        ### We scale our p by str_h to get our starting position in the input data, and we scale our 
        ### r by our dilation to get the correct spacing as we index over. And we subtract the padding
        ### which we will do a 0 fill for later. 

        ### this gives us a final [BLOCK_SIZE_M x BLOCK_SIZE_K] of indexes, identifying which input
        ### indexes we want from our images
        h = p[:, None] * str_h + r[None, :] * dil_h - pad_h

        ### Identical logic for the width dimension as well!
        w = q[:, None] * str_w + s[None, :] * dil_w - pad_w

        ### Ensure we only grab valid positions in our image (want to make sure padding is 0) ###
        mask_input = (h >= 0) & (h < H_in) & (w >= 0) & (w < W_in)

        ### Make sure we grab only valid indexes of our weights (for this specific iteration of BLOCK_SIZE_K)
        mask_weight = (r[:, None] < K_h) & (s[:, None] < K_w) & (c[:, None] < C_in)

        ### Now we advance to the actual data we need ###
        ### Remember our input data is actually still in our standard image shape [B, C_in, H_in, W_in] ###
        ### and we now know what B, C_in, H_in, W_in we want! So lets advance to that block ###
        ### This is just multiplying every dimension by its computed stride. 
        ### n * (C * H * W) + c * (H * W) + h * W + wn * (C * H * W) + c * (H * W) + h * W + w

        ### Remember, n was computed from offs_m so it has [BLOCK_SIZE_M] elements, we make this a column vector [BLOCK_SIZE_M, 1]
        ### c was computed from offs_k so it has [BLOCK_SIZE_K] elements we make this a row vector [1, BLOCK_SIZE_K]
        ### h,w are are already [BLOCK_SIZE_M, BLOCK_SIZE_K] matricies

        ### Our data is (B x C_in x H_in, W_in)
        
        ### STEP 1
        ### So to get to the right B that we are processing (our n), we need to advance it by all the other dimensions
        ### Thus we have  n[:, None] * C_in * H_in * W_in -> [BLOCK_SIZE_M, 1]

        ### STEP 2:
        ### This only gets us to the correct batch. Which channels are we processing? Well not all of them at once, 
        ### as we are computing our channels a chunk at a time (loop through C_in*K_h*K_w)
        ### This means we have to advance to the correct channels as well, so similarly
        ### (n[:, None] * C_in * H_in * W_in) + c[None, :] * H_in * W_in -> [BLOCK_SIZE_M, BLOCK_SIZE_K]
        
        ### STEP 3:
        ### So we are at the right batch and the right channel now. But what pixels do we want to process? 
        ### We dont want all of them, we already computed in h/w what pixels we actually need (pre-indexed for stride/dilation/padding)
        ### So lets just advance our pointers one more time to those specific height and width!
        ### + h * W_in + w <- first advance the height (moving over W_in amount for each) and then advance the width
        ### offs_input [BLOCK_SIZE_N, BLOCK_SIZE_K]
        offs_input = n[:, None] * C_in * H_in * W_in + c[None, :] * H_in * W_in + h * W_in + w

        ### In the same way we know what block of our weights we are also processing ###
        ### As we have computed for our [C_out, C_in, K_h, K_w]  what channel, and output K_h,K_w coordinate we want 
        ### offs_channel[None, :] * C_in * K_h * K_w <- Advance the right output channels in our kernel [1, BLOCK_SIZE_N]
        ### c[:, None] * K_h * K_w <- Advance to the right input channels in our kernel [BLOCK_SIZE_K, 1]
        ### r[:, None] * K_w + s[:, None] <- Advance to the right K_h and K_w [BLOCK_SIZE_K, 1]
        ### offs_weight [BLOCK_SIZE_K, BLOCK_SIZE_N]
        offs_weight = offs_channel[None, :] * C_in * K_h * K_w + c[:, None] * K_h * K_w + r[:, None] * K_w + s[:, None]

        ### Now we can actuall get our data ! ###
        input_ptrs = input_ptr + offs_input
        weight_ptrs = weight_ptr + offs_weight

        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0) # <- 0 padding
        weight_data = tl.load(weight_ptrs, mask=mask_weight, other=0.0) # <- 0 padding

        accumulator += tl.dot(input_data, weight_data)

    ### If we have a Bias we can just add it here! We have one bias value per output ###
    ### channel (k), so we can just index it and add it in here ###
    if bias_ptr is not None:
            offs_bias = offs_channel[None, :]
            bias_ptrs = bias_ptr + offs_bias
            bias_data = tl.load(bias_ptrs)
            accumulator = accumulator + bias_data

    accumulator = accumulator.to(tl.float16)

    ### And finally we can store this data ###
    ### this is the same formula from before, just now its on the output shapes! ###
    ### n[:, None] * C_out * H_out * W_out <- Advance to the right batch [BLOCK_SIZE_M, 1]
    ### offs_channel[None, :] * H_out * W_out  <- Advance to the right channel [1, BLOCK_SIZE_N]
    ### p[:, None] * W_out + q[:, None] <- Advance to the right output positions [BLOCK_SIZE_M, 1]
    ### offs_output: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_output = n[:, None] * C_out * H_out * W_out + offs_channel[None, :] * H_out * W_out + p[:, None] * W_out + q[:, None]

    ### And we need to make sure we dont store in invalid positions
    ### n[:, None] < B <- Make sure our batches are inside the total batch size [BLOCK_SIZE_M, 1]
    ### offs_channel[None, :] < C_out <- Make sure our channels are within our output channels [1, BLOCK_SIZE_N]
    ### p[:, None] < H_out and q[:, None] < W_out   <- Make sure our output pixels are within the max Height/Width p [BLOCK_SIZE_M, 1]
    ### mask_output [BLOCK_SIZE_M, BLOCK_SIZE_N]
    mask_output = (n[:, None] < B) & (offs_channel[None, :] < C_out) & (p[:, None] < H_out) & (q[:, None] < W_out)

    output_ptrs = output_ptr + offs_output
    tl.store(output_ptrs, accumulator, mask=mask_output)

@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'str_h', 'str_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def _implicit_gemm_conv2d_input_bwd_kernel(
    dinput_ptr, doutput_ptr, weight_ptr, 
    B, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):             
    
    """
    The backward pass will be very similar to our forward pass! Lets remind ourselves, what what the operation
    that we did in the forward pass

    (B*H_out*W_out x C_in*K_h*K_w) @ (C_in*K_h*K_w x C_out) = (B*H_out*W_out x C_out)

    But we did this without explicitly constructing that large first matrix. Still, matmul rules apply like normal
   
    We have our upstream grads:
        doutput: (B, C_out, H_out, W_out)
        
    We need to do :
        Grad w.r.t input (B*H_out*W_out x C_in*K_h*K_w)
        Grad w.r.t weight (C_in*K_h*K_w x C_out)

    We will do our grad w.r.t input first:

    Just like in a linear layer:

    output = input @ weight + bias
    dinput = doutput @ weight.T 


    """
    
    ### Identical Grouped Matmul Setup again ###
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    ### Now get our offsets for the Block of the Matmul we want to compute ###
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_channel = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    n = offs_m // (H_in * W_in)
    nhw_residual = offs_m % (H_in * W_in)
    h = nhw_residual // W_in
    w = nhw_residual % W_in

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):

        ### Our weight matrix is (C_in*K_h*K_w, C_out), but we need its transpose! ###
        ### We will do this later, but our weight mat will be actually (C_out, C_in*K_h*K_w,)
        ### and our output_grad will be (B*H_out*W_out x C_out). Therefore our reduction 
        ### Dimension (that we are chunking by BLOCK_SIZE_K) will be on our C_out dimension this time!
        offs_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        c = offs_k // (K_h * K_w)
        krs_residual = offs_k % (K_h * K_w)
        r = krs_residual // K_w
        s = krs_residual % K_w

        ### In our forward pass we were computing the input shape from the output. ###
        ### basically what (h,w) from our input corresponded to this (p,q) ###
        ### Now we are going backward. We have gradients in the shape of our output ###
        ### And we want to accumulate it into gradients the shape of our input ###
        ### So what we want to compute is what is the (p,q) for the (h,w) i want!

        ### if h = p * str_h + r * dil_h - pad_h
        ### then p = (h - r * dil_h + pad) // str_h
        h_tmp = (h[:, None] + pad_h - r[None, :] * dil_h) # <- what indexes of h coorespond to q (unstrided)
        p = h_tmp // str_h  # <- adjust for stride (check which h positions in our input is a part of our strided output)

        # Input indices:  [0, 1, 2, 3, 4, 5, 6]
        #                  ↓     ↓     ↓     ↓
        # Output indices: [0,    1,    2,    3]   (only even positions used)

        mask_p = (h_tmp % str_h == 0) & (p >= 0) & (p < H_out) # <- remember that we want the valid output height index p.
                                                           #    well its only valid IF 
                                                           # 1) it is within our bounds
                                                           # 2) it comes from an h (input height index) that was divisible by our stride
                                                           #    if it isnt divisible then it was skipped over when striding
                                                           #    and should have no contribution to the gradients

        ### Identical logic to get our width
        w_tmp = (w[:, None] + pad_w - s[None, :] * dil_w) 
        q = w_tmp // str_w
        mask_q = (w_tmp % str_w == 0) & (q >= 0) & (q < W_out)
        
        ### Ensure we only grab valid regions of our doutput ###
        mask_doutput = (c[None, :] < C_out) & (n[:, None] < B) & mask_p & mask_q

        ### Ensure we only grad valid regions of our weights ###
        ### Before our mask_weight was on C_in, but now its on C_out as that is what our c
        ### is representing! A chunk along our C_out dimension
        mask_weight = (c[:, None] < C_out) & (r[:, None] < K_h) & (s[:, None] < K_w)

        ### Grab our doutput like normal ###
        offs_doutput = n[:, None] * C_out * H_out * W_out + c[None, :] * H_out * W_out + p * W_out + q

        ### Grab the transpose of our weights! As we want doutput @ W^T !!!
        ### before we did 
        ### offs_weight = offs_channel[None, :] * C_in * K_h * K_w + c[:, None] * K_h * K_w + r[:, None] * K_w + s[:, None]
        ### Notice the flip in our strides, its an easy way to do a transpose without having to to a tl.trans again after
        ### transposing after the fact (saves an op)
        offs_weight = c[:, None] * C_in * K_h * K_w + offs_channel[None, :] * K_h * K_w + r[:, None] * K_w + s[:, None]

        ### Grab the data ###
        doutput_ptrs = doutput_ptr + offs_doutput
        weight_ptrs = weight_ptr + offs_weight

        doutput_data = tl.load(doutput_ptrs, mask=mask_doutput, other=0.0)
        weight_data_T = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        ### Do our Matmul! ###
        acc = tl.dot(doutput_data, weight_data_T, acc)

    acc = acc.to(tl.float16)

    ### Get the corresponding dinput values this block refers to ###
    offs_dinput = n[:, None] * C_in * H_in * W_in + offs_channel[None, :] * H_in * W_in + h[:, None] * W_in + w[:, None]
    mask_dinput = (n[:, None] < B) & (offs_channel[None, :] < C_in) & (h[:, None] < H_in) & (w[:, None] < W_in)
    dinput_ptrs = dinput_ptr + offs_dinput
    tl.store(dinput_ptrs, acc, mask=mask_dinput)

@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'str_h', 'str_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def _implicit_gemm_conv2d_weight_bwd_kernel(
    dweight_ptr, doutput_ptr, input_ptr, 
    B, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):  
    
    """
    The backward pass will be very similar to our forward pass! Lets remind ourselves, what what the operation
    that we did in the forward pass

    (B*H_out*W_out x C_in*K_h*K_w) @ (C_in*K_h*K_w x C_out) = (B*H_out*W_out x C_out)

    But we did this without explicitly constructing that large first matrix. Still, matmul rules apply like normal
   
    We have our upstream grads:
        doutput: (B, C_out, H_out, W_out)
        
    We need to do :
        Grad w.r.t input (B*H_out*W_out x C_in*K_h*K_w)
        Grad w.r.t weight (C_in*K_h*K_w x C_out)

    We will now do our grad w.r.t weight

    Just like in a linear layer:

    output = input @ weight + bias
    dweight = input.T @ doutput 

    This matmul will then do: (C_in * K_h * K_w, B * H_out * W_out) @ (B*H_out*W_out, C_out)
    which will produce a (C_in * K_h * K_w, C_out). 

    One caveat is its easier for us to compute our dweight^T as we will need to transpose it anyway
    for later!

    dweight^T = [(C_in * K_h * K_w, B * H_out * W_out) @ (B*H_out*W_out, C_out)]^T
               = (B*H_out*W_out, C_out)^T @ (C_in * K_h * K_w, B * H_out * W_out)^T
               = (C_out, B*H_out*W_out) @  (B * H_out * W_out, C_in * K_h * K_w)
               = (C_out, C_in*K_h*K_w)
    

    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_channel = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) 

    ### Notice that offs_channel now goes with C_out with index M and offs_n goes with C_in*K_h*K_w ###
    ### this is where we do our flip in indexes so evetrything going forward is now transposed! ###
    ### Previously, we had offs_m go with our C_out ###
                     
    c = offs_n // (K_h * K_w)                   # <- which input channel are we on?
    crs_residual = offs_n % (K_h * K_w)         # <- advance to the input channel
    r = crs_residual // K_w                     # <- which kernel height are we at?
    s = crs_residual % K_w                      # <- which kernel width are we at?
     

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):

        ### We will be computing our inner dimension in chunks. In this case our inner dimemsion ###
        ### will be (B * H_out * W_out) ###

        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))        
        n = gemm_k // (H_out * W_out)                                   # <- what batch are we processing?
        npq_residual = gemm_k % (H_out * W_out)                         # <- advance to the batch 
        p = npq_residual // W_out                                       # <- what output heights are we processing
        q = npq_residual % W_out                                        # <- what output widths are we processing

        ### Grab our doutput like normal by unrolling our indexing ###
        ### n[None, :] * C_out * H_out * W_out -> [1 x BLOCK_SIZE_K]
        ### offs_channel[:, None] * H_out * W_out -> [BLOCK_SIZE_M, 1]
        offs_doutput = n[None, :] * C_out * H_out * W_out + offs_channel[:, None] * H_out * W_out + p[None, :] * W_out + q[None, :]

        ### Now we need to compute the input that cooresponds to the output (just like we did in the forward pass)
        ### We have our doutput, we need the portion of the input that cooresponds to that to do our matmul
        # p: [BLOCK_SIZE_K, 1]
        # r: [1, BLOCK_SIZE_N]
        # h: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        h = p[:, None] * str_h + r[None, :] * dil_h - pad_h
        w = q[:, None] * str_w + s[None, :] * dil_w - pad_w
        
        ### Unroll our indexes to get what inputs we want ###
        offs_input = n[:, None] * C_in * H_in * W_in + c[None, :] * H_in * W_in + h * W_in + w

        ### Mask out invalid positions ###
        ### [BLOCK_SIZE_M, BLOCK_SIZE_K]
        mask_doutput = (n[None, :] < B) & (offs_channel[:, None] < C_out) & (p[None, :] < H_out) & (q[None, :] < W_out)
        ### [BLOCK_SIZE_K, BLOCK_SIZE_N]
        mask_input = (n[:, None] < B) & (c[None, :] < C_in) & (h < H_in) & (w < W_in) & (h >= 0) & (w >= 0)

        ### Load our data and matmul! ###
        doutput_ptrs = doutput_ptr + offs_doutput
        input_ptrs = input_ptr + offs_input

        doutput_data = tl.load(doutput_ptrs, mask=mask_doutput, other=0.0)
        weight_data = tl.load(input_ptrs, mask=mask_input, other=0.0)

        acc = tl.dot(doutput_data, weight_data, acc)

    acc = acc.to(tl.float16)
    
    ### We meed to get the cooresponding positions in our weights that we want to store in ###
    ### our weights are in the shape of (C_out, C_in, K_h, K_w)
    ### and offs_channel tells me which C_out i am in, and GEMM_N is  C_in*K_h*K_w, thus
    ### offs_channel[:, None] * GEMM_N will advance us to the correct channel, and 
    ### offs_n will advance us to the correct positions in our columns of the matrix 
    offs_weight = offs_channel[:, None] * GEMM_N + offs_n[None, :]

    ### Similarly we want to makes ure we dont save in any invalid positions ###
    mask_weight = (offs_channel[:, None] < GEMM_M) & (offs_n[None, :] < GEMM_N)

    ### Advance our pointer and save! ###
    dweight_ptrs = dweight_ptr + offs_weight
    tl.store(dweight_ptrs, acc, mask=mask_weight)

@triton.jit
def _implicit_gemm_conv2d_bias_bwd_kernel(dbias, doutput_ptr, N, K, P, Q, BLOCK_SIZE: tl.constexpr):
    k = tl.program_id(0)

    offs_pq = tl.arange(0, BLOCK_SIZE)
    mask_pq = offs_pq < P * Q

    offs_k = k + tl.arange(0, 1)

    acc = tl.zeros((1, ), dtype=tl.float32)

    for idx_n in range(0, N):
        offs_nkpq = idx_n * K * P * Q + k * P * Q + offs_pq
        doutput_ptrs = doutput_ptr + offs_nkpq
        doutput_data = tl.load(doutput_ptrs, mask=mask_pq, other=0.0)
        acc = acc + tl.sum(doutput_data)

    acc = acc.to(tl.float16)

    dbias_ptrs = dbias + offs_k
    tl.store(dbias_ptrs, acc)


def _implicit_gemm_conv2d_bias_bwd(doutput):
    N, K, P, Q = doutput.shape
    BLOCK_SIZE = triton.next_power_of_2(P * Q)
    dbias = torch.zeros((K), dtype=doutput.dtype, device=doutput.device)
    _implicit_gemm_conv2d_bias_bwd_kernel[K, ](dbias, doutput, N, K, P, Q, BLOCK_SIZE)
    return dbias


def _implicit_gemm_conv2d_bias_bwd(doutput):
    B, C_out, H_out, W_out = doutput.shape

    ### grad for bias is super simple! We just sum up all the gradient contributions of our 
    ### grad from all dimensions but C_out. Our bias vector has C_out elements, so this is 
    ### just a simple reduction op (probably dont even keen a kernel for it!)
    BLOCK_SIZE = triton.next_power_of_2(H_out * W_out)
    dbias = torch.zeros((C_out), dtype=doutput.dtype, device=doutput.device)
    _implicit_gemm_conv2d_bias_bwd_kernel[C_out, ](dbias, doutput, B, C_out, H_out, W_out, BLOCK_SIZE)

    return dbias

def fused_conv2d_forward(input, weight, bias, stride, padding, dilation):

    B, C_in, H_in, W_in = input.shape
    C_out, kernel_C_in, K_h, K_w = weight.shape
    
    assert C_in == kernel_C_in, f"Mismatch In Channels, Expected {kernel_C_in} channels, got {C_in}"

    if isinstance(stride, tuple):
        str_h, str_w = stride
    else:
        str_h = str_w = stride

    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding

    if isinstance(dilation, tuple):
        dil_h, dil_w = dilation
    else:
        dil_h = dil_w = dilation

    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // str_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // str_w + 1

    ### The matmul is between a (B * H_out * W_out, C_in * K_h * K_w) @ (C_in * K_h * K_w, C_out)
    ### We will not construct the Im2Col matrix explicitly, but we can compute the dims for the 
    ### matmul we do later! 
    
    ### Rewrite it as:
    ### (GEMM_M, GEMM_K) @ (GEMM_K, GEMM_N)
    GEMM_M = B * H_out * W_out
    GEMM_N = C_out
    GEMM_K = C_in * K_h * K_w

    ### We know what the output shape should be already as well!
    output = torch.zeros((B, C_out, H_out, W_out), dtype=input.dtype, device=input.device)

    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )

    _implicit_gemm_conv2d_fwd_kernel[grid](
        output, input, weight, bias, 
        B, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
    )

    return output

def _implicit_gemm_conv2d_input_bwd(doutput, weight, H_in, W_in, stride, padding, dilation):

    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    B, C_out, H_out, W_out = doutput.shape
    C_out, C_in, K_h, K_w = weight.shape

    ### To compute our doutput, we follow standard rules ###
    ### d_input = d_output @ W^T
    ### d_output comes in the shape (B x C_out x H_out x W_out)
    ### W is in the shape (C_out x C_in x K_h x K_w)

    ### But of course, the way we actually perform this op is by implicitly creating
    ### the im2col matrix. So actually we will have:

    ### d_output will be viewed as (B*H_out*W_out, C_out) <- grad at every output position for every channel
    ###                                                      these grads have to be distributed to all the 
    ###                                                      inputs that contributed to them!

    ### W will be viewed as (C_in*K_h*K_w, C_out), but once transposed it will be (C_out, C_in*K_h*K_w) ###
    ### Thus the actual matmul we will be doing is:
    ### (B*H_out*W_out, C_out) @ (C_out, C_in*K_h*K_w) -> (B*H_out*W_out, C_in*K_h*K_w)
    ### which will then use our internal indexing to map gradient contribution from all our output patches 
    ### (B*H_out*W_out) to our inputs!

    GEMM_M = B * H_in * W_in
    GEMM_N = C_in
    GEMM_K = C_out * K_h * K_w

    dinput = torch.zeros((B, C_in, H_in, W_in), dtype=doutput.dtype, device=doutput.device)
    
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )

    _implicit_gemm_conv2d_input_bwd_kernel[grid](
        dinput, doutput, weight, 
        B, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
    )

    return dinput

def _implicit_gemm_conv2d_weight_bwd(doutput, input, K_h, K_w, stride, padding, dilation):
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation
    B, C_out, H_out, W_out = doutput.shape
    B, C_in, H_in, W_in = input.shape

    ### To compute our dweight, we follow standard rules ###
    ### d_weight = input^T @ d_output
    ### d_output comes in the shape (B x C_out x H_out x W_out)
    ### input is in the shape (B, C_in, H_in, W_in)

    ### But of course, the way we actually perform this op is by implicitly creating
    ### the im2col matrix. So actually we will have:

    ### d_output will be viewed as (B*H_out*W_out, C_out) <- grad at every output position for every channel
    ###                                                      these grads have to be distributed to all the 
    ###                                                      inputs that contributed to them!
    
    ### W will be viewed as (B * H_out * W_out, C_in * K_h * K_w) which is our number of patches by the patch values
    ### and this will then be transposed so we will really have:
    ### input^T = (C_in * K_h * K_w, B * H_out * W_out)

    ### This matmul will then do: (C_in * K_h * K_w, B * H_out * W_out) @ (B*H_out*W_out, C_out)
    ### which will produce a (C_in * K_h * K_w, C_out). This is totally correct in terms of the
    ### actual operation we did in our Im2Col:

    ###               data                                 weights
    ### (B * H_out * W_out, C_in * K_h * K_w) @ (C_in * K_h * K_w, C_out)

    ### but remember our weights are actually in the shape: (C_out, C_in, K_h, K_w)
    ### but we are going to compute (C_in * K_h * K_w, C_out), which is transposed! 
    ### No worries we can just transpose everything. We now what to compute:

    ### dweight^T = [(C_in * K_h * K_w, B * H_out * W_out) @ (B*H_out*W_out, C_out)]^T
    ###           = (B*H_out*W_out, C_out)^T @ (C_in * K_h * K_w, B * H_out * W_out)^T
    ###           = (C_out, B*H_out*W_out) @  (B * H_out * W_out, C_in * K_h * K_w)

   
    GEMM_M = C_out
    GEMM_N = C_in * K_h * K_w
    GEMM_K = B * H_out * W_out

    dweight = torch.zeros((C_out, C_in, K_h, K_w), dtype=doutput.dtype, device=doutput.device)
    
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )

    _implicit_gemm_conv2d_weight_bwd_kernel[grid](
        dweight, doutput, input, 
        B, C_in, H_in, W_in, C_out, H_out, W_out, K_h, K_w, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K,
    )
    return dweight


@pytest.mark.parametrize("N, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding, dilation", [
    (1, 1, 3, 3, 1, 3, 3, 1, 0, 1),
    (1, 2, 3, 3, 2, 2, 2, 1, 0, 1),
    (2, 3, 8, 8, 4, 3, 3, 2, 1, 1),
    (1, 4, 9, 9, 8, 3, 3, 3, 0, 1),
    (2, 8, 16, 16, 16, 3, 3, 2, 1, 1),
    (2, 8, 8, 8, 16, 3, 3, 1, 1, 1),
    (1, 4, 8, 8, 4, 3, 3, 1, 3, 1),
    (2, 7, 8, 8, 5, 3, 3, 2, 2, 1),
    (1, 3, 16, 16, 8, 3, 3, 1, 2, 2),
    (1, 4, 20, 20, 8, 3, 3, 1, 3, 3),
    (2, 8, 16, 16, 16, 3, 3, 2, 1, 2),
    (2, 16, 32, 32, 128, 3, 3, 1, 1, 1),
    (1, 128, 16, 16, 32, 1, 1, 1, 0, 1),
    (1, 32, 16, 16, 128, 1, 1, 1, 0, 1),
    (2, 8, 4, 4, 16, 3, 3, 1, 1, 1),
    (4, 32, 64, 64, 64, 3, 3, 2, 1, 1),
    (2, 16, 128, 128, 32, 3, 3, 2, 1, 1),
    (1, 8, 256, 256, 16, 3, 3, 2, 1, 1),
    (2, 8, 16, 32, 16, 3, 3, 1, 1, 1),
    (1, 8, 7, 7, 16, 7, 7, 1, 0, 1),
])
def test_conv2d(N, C_in, H_in, W_in, C_out, K_h, K_w, stride, padding, dilation):

    factory_kwargs = {'device': 'cuda', 'dtype': torch.float16}
    input = torch.randn(N, C_in, H_in, W_in, requires_grad=True, **factory_kwargs)
    weight = torch.randn(C_out, C_in, K_h, K_w, requires_grad=True, **factory_kwargs)
    bias = torch.randn(C_out, requires_grad=True, **factory_kwargs)
    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)
    output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation)
    doutput = torch.randn_like(output)
    output.backward(doutput)

    dinput, input.grad = input.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None
    dbias, bias.grad = bias.grad.clone(), None

    triton_output = fused_conv2d_forward(input, weight, bias, stride, padding, dilation)
    triton_dinput = _implicit_gemm_conv2d_input_bwd(doutput, weight, H_in, W_in, stride, padding, dilation)
    triton_dweight = _implicit_gemm_conv2d_weight_bwd(doutput, input, K_h, K_w, stride, padding, dilation)
    triton_dbias = _implicit_gemm_conv2d_bias_bwd(doutput)

    assert torch.allclose(output, triton_output, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dinput, triton_dinput, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dweight, triton_dweight, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dbias, triton_dbias, atol=1e-1, rtol=1e-1)
