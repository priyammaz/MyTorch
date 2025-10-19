import cupy as cp
from cupyx.distributed import NCCLBackend
import argparse

def main(rank, world_size):
    cp.cuda.Device(rank).use()

    comm = NCCLBackend(n_devices=world_size, rank=rank, host="127.0.0.1", port=13333)

    # Each rank creates its own tensor
    x = cp.ones(4, dtype=cp.float32) * (rank + 1)
    y = cp.zeros_like(x)

    print(f"[Rank {rank}] Before allreduce: {x}")

    # out_array must be passed in (y will store the result)
    comm.all_reduce(x, y, op="sum")

    print(f"[Rank {rank}] After allreduce: {y}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    args = parser.parse_args()

    main(args.rank, args.world_size)