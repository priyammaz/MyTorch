MAX_FUSED_SIZE = 2**16
def calc_num_warps(block_size):
    if block_size > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since block_size = {block_size} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   block_size >= 32768: num_warps = 32
    elif block_size >=  8192: num_warps = 16
    elif block_size >=  2048: num_warps = 8
    return num_warps