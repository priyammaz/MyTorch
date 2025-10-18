import argparse
import os
import subprocess
import sys
import signal

def main():
    parser = argparse.ArgumentParser(description="mytorch distributed launch")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--training_script", type=str, required=True)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="13333")

    # Parse only launcher-specific args
    args, training_args = parser.parse_known_args()

    world_size = args.num_gpus
    procs = []

    try:
        for rank in range(world_size):
            env = os.environ.copy()
            env["RANK"] = str(rank)
            env["WORLD_SIZE"] = str(world_size)
            env["LOCAL_RANK"] = str(rank)
            env["CUPYX_DISTRIBUTED_HOST"] = os.environ.get("CUPYX_DISTRIBUTED_HOST", args.master_addr)
            env["CUPYX_DISTRIBUTED_PORT"] = os.environ.get("CUPYX_DISTRIBUTED_PORT", args.master_port)

            cmd = [sys.executable, args.training_script] + training_args

            ### Start new process group ###
            p = subprocess.Popen(
                cmd, 
                env=env, 
                preexec_fn=os.setsid # Ensures that every child gets its own process
            )

            procs.append(p)

        ### Wait for Processes ###
        for p in procs:
            p.wait()
    
    except KeyboardInterrupt:
        print("Killing Process Groups")
        for p in procs:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    ### Sanity check incase any worker refuses to close ###
    finally:
        for p in procs:
            if p.poll() is None: # none indicates still running
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)

if __name__ == "__main__":
    main()