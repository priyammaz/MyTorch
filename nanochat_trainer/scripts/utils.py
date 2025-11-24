import os

def get_last_checkpoint(path_to_checkpoint_dir, default_final_checkpoint="final_checkpoint"):

    """
    Quick helper method that looks at checkpoints that are in the format:

    checkpoint_100
    checkpoint_200
    ...

    where 100, 200, ... are the number of training steps done to get to that checkpoint. Each run will 
    end with a final_checkpoint folder. If this folder exists then training is done and we dont need to 
    do anything!

    """

    checkpoints = os.listdir(path_to_checkpoint_dir)

    if len(checkpoints) == 0:
        return None
    if "final_checkpoint" in checkpoints:
        return -1 # placeholder, this means this training run is done and we can stop
    else:
        latest_checkpoint = sorted(checkpoints, key = lambda x: int(x.split("_")[-1]))[-1]
        return latest_checkpoint

