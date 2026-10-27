"""
Sorted segment scatter-add: rows[index] += values, summing duplicate indices.

This is the backward of advanced indexing (Tensor.__getitem__ with integer arrays) and therefore of
embeddings (F.embedding is just weight[indices]). It follows PyTorch's embedding_dense_backward:

1. argsort the indices so duplicates become contiguous runs
2. one GPU program per sorted position, only the first position of each run does work. It sums the run
   sequentially (in float32) and adds that sum into rows[index] exactly once
"""

import cupy as cp
import triton
import triton.language as tl
from .utils import get_dtype_flag
from .utils import TritonPtr

MAX_BLOCK_SIZE = 4096

@triton.jit
def sorted_segment_add_kernel(
    dst_ptr,        # destination rows, addressed as row * stride_dst_row + col * stride_dst_col
    src_ptr,        # (n * row_size) values, contiguous
    order_ptr,      # (n,) int64, argsort of the flat indices
    sorted_ptr,     # (n,) int64, flat indices in sorted order
    n,              # number of indices
    row_size,       # elements per row
    stride_dst_row, stride_dst_col,  # destination strides in elements (a transposed grad is fine)
    BLOCK_SIZE: tl.constexpr,
    DST_DTYPE_FLAG: tl.constexpr,
    SRC_DTYPE_FLAG: tl.constexpr,
):
    pid = tl.program_id(0)       # position in sorted order
    pid_col = tl.program_id(1)   # which block of columns in the row

    dst_ptr = tl.cast(dst_ptr, tl.pointer_type(tl.float32 if DST_DTYPE_FLAG == 0 else (tl.float16 if DST_DTYPE_FLAG == 1 else tl.bfloat16)))
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.float32 if SRC_DTYPE_FLAG == 0 else (tl.float16 if SRC_DTYPE_FLAG == 1 else tl.bfloat16)))
    order_ptr = tl.cast(order_ptr, tl.pointer_type(tl.int64))
    sorted_ptr = tl.cast(sorted_ptr, tl.pointer_type(tl.int64))

    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < row_size

    ### Only the first position of each run of equal indices does any work ###
    row = tl.load(sorted_ptr + pid)
    prev = tl.load(sorted_ptr + tl.maximum(pid - 1, 0))
    if (pid == 0) or (row != prev):

        ### Sum the run sequentially in float32 ###
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        j = pid
        current = row
        while current == row:
            src_row = tl.load(order_ptr + j)
            acc += tl.load(src_ptr + src_row * row_size + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
            j += 1
            current = tl.load(sorted_ptr + tl.minimum(j, n - 1))
            if j >= n:
                current = row + 1

        ### Single writer for this destination row ###
        dst = dst_ptr + row * stride_dst_row + col_offsets * stride_dst_col
        tl.store(dst, tl.load(dst, mask=col_mask, other=0.0).to(tl.float32) + acc, mask=col_mask)


def sorted_segment_add(rows, index, values):
    """
    In place rows[index] += values, summing duplicates.

    rows:   cupy array (num_rows, ...) in float32/float16/bfloat16. Must be C contiguous, or 1D/2D with any
            non-negative strides (e.g. the transposed weight grad a Linear layer produces)
    index:  1D cupy int array of length n, already in [0, num_rows)
    values: cupy array reshapeable to (n, *rows.shape[1:]) in float32/float16/bfloat16
    """
    n = index.shape[0]
    if n == 0:
        return rows

    num_rows = rows.shape[0]
    row_size = rows.size // max(num_rows, 1)
    if rows.flags.c_contiguous:
        stride_dst_row, stride_dst_col = row_size, 1
    elif rows.ndim <= 2 and all(st >= 0 for st in rows.strides):
        stride_dst_row = rows.strides[0] // rows.itemsize
        stride_dst_col = rows.strides[1] // rows.itemsize if rows.ndim == 2 else 1
    else:
        raise ValueError("sorted_segment_add needs a C contiguous destination or a 1D/2D strided one")
    index = index.astype(cp.int64, copy=False)
    values = cp.ascontiguousarray(values)

    order = cp.argsort(index)
    sorted_index = index[order]

    BLOCK_SIZE = min(triton.next_power_of_2(max(row_size, 1)), MAX_BLOCK_SIZE)
    grid = (n, triton.cdiv(row_size, BLOCK_SIZE))

    with cp.cuda.Device(rows.device.id):
        sorted_segment_add_kernel[grid](
            TritonPtr(rows), TritonPtr(values), TritonPtr(order), TritonPtr(sorted_index),
            n, row_size, stride_dst_row, stride_dst_col,
            BLOCK_SIZE=BLOCK_SIZE,
            DST_DTYPE_FLAG=get_dtype_flag(rows.dtype),
            SRC_DTYPE_FLAG=get_dtype_flag(values.dtype),
        )
    return rows