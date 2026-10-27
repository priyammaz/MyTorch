import argparse
import os
import subprocess
import sys
import signal
import time

def main():
    parser = argparse.ArgumentParser(description="mytorch distributed launch")
    parser.add_argument("training_script", type=str) # <- Only positional argument (expected in first position)
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="13333")
    parser.add_argument("-m", "--module", action="store_true")

    # Parse only launcher-specific args
    args, training_args = parser.parse_known_args()
    world_size = args.num_gpus
    procs = []
    exit_code = 1

    try:
        for rank in range(world_size):
            env = os.environ.copy()
            env["RANK"] = str(rank)
            env["WORLD_SIZE"] = str(world_size)
            env["LOCAL_RANK"] = str(rank)
            env["CUPYX_DISTRIBUTED_HOST"] = os.environ.get("CUPYX_DISTRIBUTED_HOST", args.master_addr)
            env["CUPYX_DISTRIBUTED_PORT"] = os.environ.get("CUPYX_DISTRIBUTED_PORT", args.master_port)
            
            if args.module:
                ### run in module mode like python -m package.run
                cmd = [sys.executable, "-m", args.training_script] + training_args
            else:
                ### otherwise run like normal 
                cmd = [sys.executable, args.training_script] + training_args
                
            ### Start new process group ###
            p = subprocess.Popen(
                cmd, 
                env=env, 
                preexec_fn=os.setsid # Ensures that every child gets its own process
            )

            procs.append(p)

        ### Wait for Processes ###
        while True:
            codes = [p.poll() for p in procs]
            failed = [(rank, code) for rank, code in enumerate(codes) if code not in (None, 0)]
            if failed:
                rank, code = failed[0]
                print(f"Rank {rank} exited with code {code}, killing all ranks")
                for p in procs:
                    if p.poll() is None:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                time.sleep(5)
                exit_code = code
                break
            if all(code == 0 for code in codes):
                exit_code = 0
                break
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Killing Process Groups")
        for p in procs:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    ### Sanity check incase any worker refuses to close ###
    finally:
        for p in procs:
            try:
                ### kill the whole group, dataloader workers can outlive a crashed rank and hold GPU memory ###
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    return exit_code

if __name__ == "__main__":
    sys.exit(main())