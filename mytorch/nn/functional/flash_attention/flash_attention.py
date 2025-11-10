"""
This is just a wrapper on our Flash Attention Triton Implementation! This flash attention 
supports the following:

- Self Attention
- Causal Self Attention
- Cross Attention
- Custom Attention Masks
- Group Query Attention

Still missing: Dropout!!

"""
from mytorch import Tensor
from mytorch.nn.functional import _compat as CHECKS
from mytorch.nn.functional.utils import get_inner_inner_array
from ..fused_ops import fused_sdpa_forward, fused_sdpa_backward, \
    fused_cross_sdpa_forward, fused_cross_sdpa_backward

def scaled_dot_product_attention(Q, K, V, 
                                 attn_mask=None,
                                 is_causal=False, 
                                 softmax_scale=None, 
                                 enable_gqa=False):
    
    if not CHECKS.FUSED_AVAIL:
        raise Exception("Fused ops not available, install Triton!!!")
    
    Q_data = Q.data
    K_data = K.data
    V_data = V.data

    ### If we are an Array drill down to ndarray ###
    Q_data = get_inner_inner_array(Q_data)
    K_data = get_inner_inner_array(K_data)
    V_data = get_inner_inner_array(V_data)

    ### SDPA Sanity Checks ###
    batch_q, heads_q, len_q, embed_q = Q.shape
    batch_k, heads_k, len_k, embed_k = K.shape
    batch_v, heads_v, len_v, embed_v = V.shape

    if (not enable_gqa) and (heads_k != heads_q):
        raise Exception(f"Queries have {heads_q} heads and it doesn't match Keys heads {heads_k}!")
    elif enable_gqa:
        assert heads_q % heads_k == 0, "Heads of Queries must be divisible by Keys/Values for GQA!"

    assert len_k == len_v, "Keys and Values must have the same length"
    assert (embed_q == embed_k) and (embed_k == embed_v), "Q,K,V must all have the same embedding dimension"
    if is_causal:
        assert len_q == len_k, "Causal mode is only supported in Self-Attention, len of Q,K,V must all be the same!"
    self_attn = True
    if len_q != len_k:
        self_attn = False
    
    if attn_mask is not None:
        assert len(attn_mask.shape) == 4, "Expected Attention Mask in the shape of (B, ..., L, S)"
        assert (len_q == attn_mask.shape[2]) and (len_k == attn_mask.shape[3]), f"Expected Attention Mask is Shape ({batch_q}, ... {len_q}, {len_k}), got {attn_mask.shape}"
        if (attn_mask.shape[1] != heads_q) and (attn_mask.shape[1] != 1):
            raise Exception("Expected either {heads_q} head dimension or 1 in attention mask")

    if self_attn:
        Q_data, K_data, V_data, attn_out, M = fused_sdpa_forward(
            Q=Q_data, K=K_data, V=V_data, attn_mask=attn_mask, causal=is_causal, softmax_scale=softmax_scale
        )
    else:
        Q_data, K_data, V_data, attn_out, M = fused_cross_sdpa_forward(
            Q=Q_data, K=K_data, V=V_data, attn_mask=attn_mask, softmax_scale=softmax_scale
        )

    def _sdpa_backward(grad_output):
        
        if self_attn:
            dQ, dK, dV = fused_sdpa_backward(
                dO=grad_output, Q=Q_data, K=K_data, V=V_data, O=attn_out, M=M, 
                attn_mask=attn_mask, causal=is_causal, softmax_scale=softmax_scale
            )
        
        else:
            dQ, dK, dV = fused_cross_sdpa_backward(
                dO=grad_output, Q=Q_data, K=K_data, V=V_data, O=attn_out, M=M,
                attn_mask=attn_mask, softmax_scale=softmax_scale
            )



        dQ, dK, dV = fused_sdpa_backward(grad_output, 
                                         Q_data, K_data, V_data, 
                                         attn_out, M, 
                                         causal=is_causal,
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