import cupy as cp
from cupyx.distributed import NCCLBackend
import os

def main(rank, world_size):
    cp.cuda.Device(rank).use()

    comm = NCCLBackend(n_devices=world_size, rank=rank, host="127.0.0.1", port=13333)

    # Each rank creates its own tensor
    x = cp.ones(4, dtype=cp.float32) * (rank + 1)
    y = cp.zeros_like(x)

    print(f"[Rank {rank}] Before allreduce: {x}")

    comm.all_reduce(x, y, op="sum")

    print(f"[Rank {rank}] After allreduce: {y}")

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main(rank, world_size)