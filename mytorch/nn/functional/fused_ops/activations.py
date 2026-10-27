"""
Fused activation inspiration from Attorch!
https://github.com/BobMcDear/attorch/tree/main/attorch
"""
import cupy as cp
import torch
import triton
import triton.language as tl
from .utils import get_dtype_flag, to_torch_dtype
from .utils import TritonPtr

def element_wise_kernel_configs():
    return [
        triton.Config({"BLOCK_SIZE": 64},   num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128},  num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
    ]

###################################
### JUST A BUNCH OF ACTIVATIONS ###
###################################

_avail_activations = (
    "sigmoid", "tanh", "gelu", "gelu_tanh", "relu", "silu",
    "leaky_relu", "relu_squared"
)

@triton.jit
def sigmoid_forward_kernel(input):
    return (1 / (1 + tl.exp(-input)))

@triton.jit
def sigmoid_backward_kernel(input):
    output = sigmoid_forward_kernel(input)
    return output * (1 - output)

@triton.jit
def tanh_forward_kernel(input):
    return 2 * sigmoid_forward_kernel(2 * input) - 1

@triton.jit
def tanh_backward_kernel(input):
    output= tanh_forward_kernel(input)
    return 1 - output * output

@triton.jit
def gelu_forward_kernel(input):
    ### Exact GELU (same as torch.nn.functional.gelu(approximate="none")): x * Phi(x) ###
    return 0.5 * input * (1 + tl.math.erf(input * 0.7071067811865476))

@triton.jit
def gelu_backward_kernel(input):
    ### d/dx x * Phi(x) = Phi(x) + x * phi(x) ###
    cdf = 0.5 * (1 + tl.math.erf(input * 0.7071067811865476))
    pdf = tl.exp(-0.5 * input * input) * 0.3989422804014327
    return cdf + input * pdf

@triton.jit
def gelu_tanh_forward_kernel(input):
    ### tanh approximation (same as torch.nn.functional.gelu(approximate="tanh")) ###
    cdf = 0.5 * (1 + tanh_forward_kernel(0.7978845608 * input * (1 + 0.044715 * input * input)))
    return cdf * input

@triton.jit
def gelu_tanh_backward_kernel(input):
    tanh_res = tanh_forward_kernel(0.7978845608 * input * (1 + 0.044715 * input * input))
    sech_sq = 1 - tanh_res * tanh_res
    return 0.5 * (1 + tanh_res + input * sech_sq * 0.7978845608 * (1 + 3 * 0.044715 * input * input))

@triton.jit
def silu_forward_kernel(input):
    return (input * sigmoid_forward_kernel(input))

@triton.jit
def silu_backward_kernel(input):
    ### recompute sigmoid to save some memory ###
    output_sigmoid = sigmoid_forward_kernel(input)
    return (output_sigmoid * (input * (1 - output_sigmoid) + 1))

@triton.jit
def relu_forward_kernel(input):
    return tl.maximum(0, input)

@triton.jit
def relu_backward_kernel(input):
    return tl.where(input <= 0, 0, 1)

@triton.jit
def leaky_relu_forward_kernel(input, negative_slope):
    return relu_forward_kernel(input) + negative_slope * tl.minimum(0, input)

@triton.jit
def leaky_relu_backward_kernel(input, negative_slope):
    return tl.where(input <= 0, negative_slope, 1)

@triton.jit
def relu_squared_forward_kernel(input):
    output = relu_forward_kernel(input)
    return output * output

@triton.jit
def relu_squared_backward_kernel(input):
    output = relu_forward_kernel(input)
    return 2 * output * relu_backward_kernel(input)

#################################
### QUICK ACTIVATION SELECTOR ###
#################################

@triton.jit
def activation_switcher_forward(name, input):

    if name == "sigmoid":
        input = input.to(tl.float32)
        output = sigmoid_forward_kernel(input)
    if name == "tanh":
        input = input.to(tl.float32)
        output = tanh_forward_kernel(input)
    if name == "gelu":
        input = input.to(tl.float32)
        output = gelu_forward_kernel(input)
    if name == "gelu_tanh":
        input = input.to(tl.float32)
        output = gelu_tanh_forward_kernel(input)
    if name == "silu":
        input = input.to(tl.float32)
        output = silu_forward_kernel(input)
    if name == "relu":
        output = relu_forward_kernel(input)
    if name == "leaky_relu":
        output = leaky_relu_forward_kernel(input)
    if name == "relu_squared":
        output = relu_squared_forward_kernel(input)
    
    return output

@triton.jit
def activation_switcher_backward(name, out_grad, input):

    ### Activations with exp/erf/tanh need the extra precision, so we upcast each element to float32 ###
    ### inside the kernel (no full size float32 copies of the tensors) and store back in the original dtype ###
    if name == "sigmoid":
        input, out_grad = input.to(tl.float32), out_grad.to(tl.float32)
        output = sigmoid_backward_kernel(input)
    if name == "tanh":
        input, out_grad = input.to(tl.float32), out_grad.to(tl.float32)
        output = tanh_backward_kernel(input)
    if name == "gelu":
        input, out_grad = input.to(tl.float32), out_grad.to(tl.float32)
        output = gelu_backward_kernel(input)
    if name == "gelu_tanh":
        input, out_grad = input.to(tl.float32), out_grad.to(tl.float32)
        output = gelu_tanh_backward_kernel(input)
    if name == "silu":
        input, out_grad = input.to(tl.float32), out_grad.to(tl.float32)
        output = silu_backward_kernel(input)
    if name == "relu":
        output = relu_backward_kernel(input)
    if name == "leaky_relu":
        output = leaky_relu_backward_kernel(input)
    if name == "relu_squared":
        output = relu_squared_backward_kernel(input)

    return output * out_grad # <- multiply by upstream grads

#####################
### ACTUAL KERNEL ###
#####################

@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['n_elements'],
)
@triton.jit
def generic_activation_kernel_forward(
    name: tl.constexpr, 
    input_ptr, 
    output_ptr, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
    OUTPUT_DTYPE_FLAG: tl.constexpr
):
    
    pid = tl.program_id(0)
    input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))
    output_ptr = tl.cast(output_ptr, tl.pointer_type(tl.float32 if OUTPUT_DTYPE_FLAG == 0 else (tl.float16 if OUTPUT_DTYPE_FLAG == 1 else tl.bfloat16)))
    
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    input = tl.load(input_ptr + offset, mask=mask)
    tl.store(output_ptr + offset, 
             activation_switcher_forward(name, input), 
             mask=mask)

@triton.autotune(
    configs=element_wise_kernel_configs(),
    key=['n_elements'],
)
@triton.jit
def generic_activation_kernel_backward(
    name: tl.constexpr, 
    input_ptr, 
    input_grad_ptr,
    output_grad_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
    DTYPE_FLAG: tl.constexpr,
    INPUT_GRAD_DTYPE_FLAG: tl.constexpr,
    INPUT_DTYPE_FLAG: tl.constexpr,
):
    
    pid = tl.program_id(axis=0)

    ### DTYPE_FLAG is the upstream grad's dtype, the saved input can differ (e.g. fp16 input, fp32 grads) ###
    input_ptr = tl.cast(input_ptr, tl.pointer_type(tl.float32 if INPUT_DTYPE_FLAG == 0 else (tl.float16 if INPUT_DTYPE_FLAG == 1 else tl.bfloat16)))
    input_grad_ptr = tl.cast(input_grad_ptr, tl.pointer_type(tl.float32 if INPUT_GRAD_DTYPE_FLAG == 0 else (tl.float16 if INPUT_GRAD_DTYPE_FLAG == 1 else tl.bfloat16)))
    output_grad_ptr = tl.cast(output_grad_ptr, tl.pointer_type(tl.float32 if DTYPE_FLAG == 0 else (tl.float16 if DTYPE_FLAG == 1 else tl.bfloat16)))

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    output_grad = tl.load(output_grad_ptr + offset, mask=mask)
    input = tl.load(input_ptr + offset, mask=mask)

    tl.store(input_grad_ptr + offset,
             activation_switcher_backward(name, output_grad, input),
             mask=mask)
    
################################
### METHODS TO ACCESS KERNEL ###
################################
def fused_activation_forward(input, act_func, use_dlpack=True):

    assert act_func in _avail_activations, f"Select an activation from {_avail_activations}"

    ### Flatten (it is element wise so we will process a long vector) ###
    orig_shape = input.shape
    orig_dtype = input.dtype
    input = input.reshape(-1)
    n_elements = input.shape[0]
    
    if use_dlpack:

        ### Convert to Torch ###
        input = torch.utils.dlpack.from_dlpack(input)

        ### Make Sure Contiguous ###
        if not input.is_contiguous():
            input = input.contiguous()

        output = torch.empty_like(input, dtype=to_torch_dtype(orig_dtype))

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        generic_activation_kernel_forward[grid](
            name=act_func,
            input_ptr=input, 
            output_ptr=output, 
            n_elements=n_elements,
            DTYPE_FLAG=get_dtype_flag(input.dtype),
            OUTPUT_DTYPE_FLAG=get_dtype_flag(orig_dtype)
        )

        return cp.from_dlpack(output.reshape(orig_shape))

    else:

        ### Make Sure Contiguous ###
        if not input.flags.c_contiguous:
            input = cp.ascontiguousarray(input)

        output = cp.empty_like(input, dtype=orig_dtype)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        generic_activation_kernel_forward[grid](
            name=act_func,
            input_ptr=TritonPtr(input), 
            output_ptr=TritonPtr(output), 
            n_elements=n_elements,
            DTYPE_FLAG=get_dtype_flag(input.dtype),
            OUTPUT_DTYPE_FLAG=get_dtype_flag(orig_dtype)
        )

        return output.reshape(orig_shape)

def fused_activation_backward(input, output_grad, act_func, use_dlpack=True):
    
    assert act_func in _avail_activations, f"Select an activation from {_avail_activations}, got {act_func}"

    ### Flatten (it is element wise so we will process a long vector) ###
    orig_shape = input.shape
    orig_dtype = input.dtype
    input = input.reshape(-1)
    output_grad = output_grad.reshape(-1)
    n_elements = input.shape[0]
    
    if use_dlpack:

        ### Convert to Torch ###
        input = torch.utils.dlpack.from_dlpack(input)
        output_grad = torch.utils.dlpack.from_dlpack(output_grad)

        ### Make Sure Contiguous ###
        if not input.is_contiguous():
            input = input.contiguous()
        if not output_grad.is_contiguous():
            output_grad = output_grad.contiguous()

        input_grad = torch.empty_like(input, dtype=to_torch_dtype(orig_dtype))

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        generic_activation_kernel_backward[grid](
            name=act_func,
            input_ptr=input, 
            input_grad_ptr=input_grad,
            output_grad_ptr=output_grad, 
            n_elements=n_elements,
            DTYPE_FLAG= get_dtype_flag(output_grad.dtype),
            INPUT_GRAD_DTYPE_FLAG = get_dtype_flag(input_grad.dtype),
            INPUT_DTYPE_FLAG = get_dtype_flag(input.dtype),
        )
  
        return cp.from_dlpack(input_grad.reshape(orig_shape))

    else:

        ### Make Sure Contiguous ###
        if not input.flags.c_contiguous:
            input = cp.ascontiguousarray(input)
        if not output_grad.flags.c_contiguous:
            output_grad = cp.ascontiguousarray(output_grad)

        input_grad = cp.empty_like(input, dtype=orig_dtype)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        generic_activation_kernel_backward[grid](
            name=act_func,
            input_ptr=TritonPtr(input), 
            input_grad_ptr=TritonPtr(input_grad),
            output_grad_ptr=TritonPtr(output_grad), 
            n_elements=n_elements,
            DTYPE_FLAG= get_dtype_flag(output_grad.dtype),
            INPUT_GRAD_DTYPE_FLAG = get_dtype_flag(input_grad.dtype),
            INPUT_DTYPE_FLAG = get_dtype_flag(input.dtype),
        )

        return input_grad.reshape(orig_shape)

if __name__ == "__main__":


    test_input_gelu = torch.tensor([-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0], dtype=torch.float16).cuda()
    test_input_gelu_cp = cp.array(test_input_gelu.detach().cpu().numpy())
    print(test_input_gelu)

    # Forward pass with custom kernel
    output_custom_gelu = fused_activation_forward(test_input_gelu_cp, "gelu", use_dlpack=False)
    print("\nCustom GELU Forward:")
    print(output_custom_gelu)

    # Forward pass with PyTorch
    output_torch_gelu = torch.nn.functional.gelu(test_input_gelu, approximate="tanh")
    print("\nPyTorch GELU Forward:")
    print(output_torch_gelu)

    # Create upstream gradient
    upstream_grad_gelu = torch.ones_like(test_input_gelu)
    upstream_grad_gelu_cp = cp.array(upstream_grad_gelu.detach().cpu().numpy())

    print("\nUpstream Gradient:")
    print(upstream_grad_gelu)

    # Backward pass with custom kernel
    input_grad_custom_gelu = fused_activation_backward(test_input_gelu_cp, upstream_grad_gelu_cp, "gelu", use_dlpack=False)
    print("\nCustom GELU Backward:")
    print(input_grad_custom_gelu)

    # Backward pass with PyTorch
    test_input_torch_gelu = test_input_gelu.clone().requires_grad_(True)
    output_torch_grad_gelu = torch.nn.functional.gelu(test_input_torch_gelu, approximate="tanh")
    output_torch_grad_gelu.backward(upstream_grad_gelu)
    print("\nPyTorch GELU Backward:")
    print(test_input_torch_gelu.grad)

    print("Forward Match:", cp.allclose(output_custom_gelu, cp.array(output_torch_gelu.detach().cpu().numpy()), atol=1e-2))
    print("Backward Match:", cp.allclose(input_grad_custom_gelu, cp.array(test_input_torch_gelu.grad.detach().cpu().numpy()), atol=1e-2))