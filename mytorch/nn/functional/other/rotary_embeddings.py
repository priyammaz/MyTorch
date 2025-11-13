"""
Rotary embeddings as implemented in LLaMA
https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

LLaMA doesn't use the exact pairing from the original Rotary Embeddings paper,
but the result is mathematically equivalent.

Let's say we have an embedding: [x1, x2, x3, x4]

In the original RoPE paper, we apply rotations (with different frequencies) to
adjacent pairs:

First pair:
[x1] [cos(w1) -sin(w1)]
[x2] [sin(w1)  cos(w1)]

Second pair:
[x3] [cos(w2) -sin(w2)]
[x4] [sin(w2)  cos(w2)]

LLaMA instead groups dimensions differently:

First pair:
[x1] [cos(w1) -sin(w1)]
[x3] [sin(w1)  cos(w1)]

Second pair:
[x2] [cos(w2) -sin(w2)]
[x4] [sin(w2)  cos(w2)]

Notice the change? Originally we grouped [x1, x2] and [x3, x4],
but now we group [x1, x3] and [x2, x4].

Why? To make the implementation more efficient.
This layout allows us to apply rotations on contiguous halves of the tensor,
which is far simpler and faster to express in CUDA/Triton kernels.

Practically, it has no effect on the model:
the order of embedding dimensions doesn’t carry semantic meaning
as long as the same pairing is used consistently during training and inference.

NOTE: The ORIGINAL Llama implementation used normal rope. This is why in the Huggingface
implementation of llama, they permute the weights to match this pattern. You can see
more here:

https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2

"""

import mytorch
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional import _flags as FLAGS
from mytorch.nn.functional.utils import get_inner_inner_array
from ..fused_ops import fused_rope_forward, fused_rope_backward

def precompute_rotary_cos_sin(head_dim, 
                              max_position_embeddings=2048, 
                              base=10000,
                              device="cpu",
                              dtype=mytorch.float32,
                              scaling_factor=1.0):
    
    """
    compute all the rotary embeds we need for the max sequence length
    and we can index from here later.

    This returns cos/sin in the shape [B x L x D], so we have to 
    unsqueeze on the head dimension, which is to be provided
    in the apply_rotary_pos_embed method later!
    """

    # Compute inverse frequencies (every other dimension)
    every_other = mytorch.arange(0, head_dim, 2, dtype=mytorch.float32)
    inv_freq = 1.0 / (base ** (every_other / head_dim))

    # Position indices scaled by scaling_factor
    t = mytorch.arange(max_position_embeddings, dtype=mytorch.float32) / scaling_factor

    # Outer product to get frequency matrix: [max_pos, head_dim//2]
    freqs = t.unsqueeze(-1) @ inv_freq.unsqueeze(0)  # [max_pos, dim/2]

    # Duplicate freqs along last dimension to match full head_dim
    freqs = mytorch.concatenate([freqs, freqs], dim=-1)  # [max_pos, head_dim]

    # Compute cos/sin and reshape to [1 x Seq Len x Head Dim]
    cos = freqs.cos().to(device).astype(dtype).unsqueeze(0)
    sin = freqs.sin().to(device).astype(dtype).unsqueeze(0)
  
    return cos, sin

def rotate_half(x):
    """
    [a1, a2, a3, a4] -> [-a3, -a4, a1, a2]
    """
    x1, x2 = x.chunk(2, dim=-1)
    rotated = mytorch.concatenate([-x2,x1], dim=-1)
    return rotated

def auto_apply_rotary_pos_embed(q, k, cos, sin, unsqueeze_dim=2):
    """
    Data is by default (B x L x H x D) so we unsqueeze on the head dim
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def fused_rotary_embbedding(q, k, cos, sin):

    cos = get_inner_inner_array(cos)
    sin = get_inner_inner_array(sin)
    
    def make_backward_fn(tensor):
        """Factory function to properly capture the tensor"""
        def _rope_backward(output_grad):
            input_grad = fused_rope_backward(output_grad, cos, sin)
          
            if tensor.grad is None:
                tensor.grad = input_grad
            else:
                tensor.grad += input_grad
        return _rope_backward
    
    outputs = []

    for x in [q,k]:
        x_arr = get_inner_inner_array(x)
        x_arr,cos,sin = fused_rope_forward(x_arr,cos,sin)
          
        requires_grad = x.requires_grad and Tensor.build_graph_enabled()

        output = Tensor(
            x_arr,
            requires_grad=requires_grad,
            grad_fn=make_backward_fn(x) if requires_grad else None,
            grad_fn_name="<RopeBackward>" if requires_grad else None
        )

        if requires_grad:
            output._add_parents(x)

        outputs.append(output)
    
    q_out, k_out = outputs

    return q_out, k_out

def apply_rotary_pos_embed(q, k, cos, sin, unsqueeze_dim=2, auto=True, fused=False):
    if auto:
        return auto_apply_rotary_pos_embed(q, k, cos, sin, unsqueeze_dim)
    else:
        _use_fused = (fused and CHECKS.FUSED_AVAIL) or FLAGS.ALWAYS_USE_FUSED
        if not _use_fused:
            raise Exception("Fused Rotary Embeddings not Supported!! Install Triton, or use Auto method")
        return fused_rotary_embbedding(q, k, cos, sin)
        
if __name__ == "__main__":

    import cupy as cp
    q = cp.random.normal(size=(2,256,12,64)).astype(cp.float16)
    k = cp.random.normal(size=(2,256,12,64)).astype(cp.float16)
    
    q_clone = q.copy()
    k_clone = k.copy()

    q_orig = mytorch.Tensor(q, requires_grad=True)
    k_orig = mytorch.Tensor(k, requires_grad=True)

    q_clone = mytorch.Tensor(q_clone, requires_grad=True)
    k_clone = mytorch.Tensor(k_clone, requires_grad=True)

    cos, sin = precompute_rotary_cos_sin(64, max_position_embeddings=256, device="cuda", dtype=mytorch.float16)
    q_embed_orig, k_embed_orig = auto_apply_rotary_pos_embed(q_orig,k_orig,cos,sin)
    q_out_fused, k_out_fused = fused_rotary_embbedding(q_clone,k_clone,cos,sin)

    print("Q_diff:", mytorch.max(mytorch.abs(q_out_fused-q_embed_orig)).item())
    print("K_diff:", mytorch.max(mytorch.abs(k_out_fused-k_embed_orig)).item())
    
    q_embed_orig.backward()
    q_out_fused.backward()

    print("dQ_diff:", cp.max(cp.abs(q_clone.grad - q_orig.grad)).item())

    k_embed_orig.backward()
    k_out_fused.backward()

    print("dK_diff:", cp.max(cp.abs(k_clone.grad - k_orig.grad)).item())