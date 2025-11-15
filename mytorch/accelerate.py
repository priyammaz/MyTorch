"""
DDP trainer inspired by Huggingface Accelerate!

This should work regardless of if we are training on DDP/Single GPU or
if we are doing Mixed/Full precision training! If everything is disabled
its just a fancy way to log and checkpoint essentially!
"""
import os
import shutil
import cupy as cp
import numpy as np
from cupyx.distributed import NCCLBackend
import pickle
from safetensors.numpy import save_file, load_file
import warnings

from .nn import *
from .optim import *
from .utils import data
from .tensor import Tensor, zeros_like
from .dtypes import *

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False

terminal_width = shutil.get_terminal_size().columns

class GradScaler:
    def __init__(self, 
                 init_scale=2.0**16,
                 growth_factor=2.0, 
                 backoff_factor=0.5, 
                 growth_interval=2000):
        
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.unskipped = 0
        self._grads_unscaled = False

    def scale_loss(self, loss):
        """
        The key part of mixes precision training is scaling our loss. 
        The problem is fp16 has lower numerical precision, so small 
        loss values will lead to grad underflows. 

        So what we do is scale our gradients up by a factor, and the 
        scale is initialized as approximately the largest representable
        value of fp16 (65536.0) 

        This means if you had a loss like 0.01, it gets scaled up to 
        65536.0 * 0.01 = 655.36

        But what if our loss is 2? Then our scaled value is 65536 * 2
        which exceeds the maximum representable numbers in fp16. Thus
        we need some rules about how we dynamically update the scaling
        factor!
        """

        loss = loss.astype("float32") * self.scale
        return loss
    
    def unscale_loss(self, loss):
        loss.data = loss.data / self.scale
        return loss
    
    def unscale_grads(self, params):

        """
        The scaling we do in scale_loss() is just for keeping our
        precision when computing gradients. Of course, our gradients
        need to be scaled down again to we have the correct magnitudes!
        """

        ### If grads are alread unscaled, nothing to do! ###
        if self._grads_unscaled:
            return True
        
        inv_scale = 1.0 / self.scale
        for param in params:
            if hasattr(param, "grad") and param.grad is not None:
                param.grad *= inv_scale

        ### Flag that gradients have been unscaled ###
        self._grads_unscaled = True

    def update(self, found_inf):

        """
        To update our scaling factor:

        If we found INF/NAN, that means we are scaling
        our values too high (self.scale is too large) 
        so lets reduce our scaling factor. This is important
        at the start of training as our loss will be high

        If we dont find any INF/NAN, that means we are good 
        to go, and if we consistently find this to be true
        for atleast self.growth_interval steps, we can go 
        ahead and increase our scale. This is important as our
        loss gets smaller and smaller

        A small scale reduces the risk of overflows, but may not 
        sufficiently amplify tiny grads leading to a loss of precision

        Large scale better preserves the small gradients, but 
        increases risk of overflow.

        """
        if found_inf:
            self.scale *= self.backoff_factor
            self.unskipped = 0
        else:
            self.unskipped += 1
            if self.unskipped == self.growth_interval:
                self.scale *= self.growth_factor
                self.unskipped = 0
    
    def reset_unscaled_flag(self):
        self._grads_unscaled = False

class Accelerator:
    def __init__(self,
                 num_gpus=None, 
                 rank=None, 
                 gradient_accumulation_steps=1,
                 mixed_precision=False,
                 log_wandb=False,
                 master_addr=None, 
                 master_port=None):
        
        ### Set Number of GPUs if not provided from environment ###
        self.rank = rank if rank is not None else int(os.environ.get("RANK", 0))
        self.world_size = num_gpus if num_gpus is not None else int(os.environ.get("WORLD_SIZE", 1))

        ### Set Address and Port ####
        self.master_addr = master_addr if master_addr is not None else os.environ.get("CUPYX_DISTRIBUTED_HOST", "127.0.0.1")
        self.master_port = master_port if master_port is not None else os.environ.get("CUPYX_DISTRIBUTED_PORT", "13333")

        ### Set Device for this rank ###
        cp.cuda.Device(self.rank).use()
        
        ### Accumulation ###
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.step_counter = 0

        ### Initialize NCCL ###
        self.comm = None
        if self.world_size > 1:
            self.comm = NCCLBackend(
                n_devices=self.world_size, 
                rank=self.rank, 
                host=self.master_addr, 
                port=int(self.master_port)
            )

        ### Random Seed per Rank ###
        cp.random.seed(seed=42 + self.rank)

        ### Mixed Precision ### 
        env_val = os.environ.get("MYTORCHRUN_MIXED_PRECISION", False)
        if env_val is False:
            self.mixed_precision = mixed_precision
        else:
            self.mixed_precision = False if env_val == "No" else True
        self.skip_optimizer_step = False
        if self.mixed_precision:
            self.scaler = GradScaler()

        ### Store Grad Norm for every grad sync ###
        ### this lets us keep an eye on model training health ###
        self._grad_norm = None

        ### Import wandb if logging with it ###
        self.wandb_enabled = False
        if log_wandb:
            if not WANDB_AVAILABLE:
                warnings.warn("Wandb is not installed! Run `pip install wandb`")
            else:
                self.wandb_enabled = True

    def is_main_process(self):
        return self.rank == 0
    
    def env(self):

        gpu_indices = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        if gpu_indices == "all":
            gpu_indices = ",".join(str(x) for x in list(range(0, self.world_size)))
     
        config = {
            "distributed": "Multi-GPU Training" if self.comm is not None else False, 
            "num_gpus": self.world_size, 
            "mixed_precision": self.mixed_precision,
            "gpu_indices": gpu_indices,
            "master_addr": self.master_addr, 
            "master_port": self.master_port
        }

        if self.is_main_process():
            print("\n" + "-" * terminal_width)
            print("Launch Configuration:")
            print("-" * terminal_width)
            for key, value in config.items():
                print(f"  {key}: {value}")
            print("-" * terminal_width)

    @property
    def device(self):
        return f"cuda:{self.rank}"
    
    @property
    def num_processes(self):
        return self.world_size
    
    @property
    def sync_grad(self):
        return self.step_counter % self.gradient_accumulation_steps == 0
    
    @property 
    def grad_norm(self):
        return self._grad_norm
    
    def prepare(self, *args):

        prepared = []

        for obj in args:
            if isinstance(obj, Module):
                prepared.append(self.prepare_model(obj))
            elif isinstance(obj, Optimizer):
                prepared.append(self.prepare_optimizer(obj))
            elif isinstance(obj, data.DataLoader):
                prepared.append(self.prepare_dataloaders(obj))
        
        return prepared
    
    def prepare_model(self, model):

        ### Store Access to Model ###
        self.model = model.to(f"cuda:{self.rank}")
        
        ### Broadcast Weights from Model into Other GPUs ###
        if self.comm is not None:
            for param in self.model.parameters():
                self.comm.broadcast(param.data._array, root=0)

        ### Set Up Mixed-Precision Training ###
        if self.mixed_precision:
    
            ### We will keep an internal copy of fp32 weights 
            ### the model may be trained in float16, but we will keep our  
            ### internal buffer in float32 and copy it back after every iteration
            ### thus it is MIXED PRECISION
            self.fp32_params = []

            ### We may have some shared tensors (in gpt2 we have weight tying)
            ### we can track which unique cupy pointers we have saved and which is a copy 
            seen = {}

            ### Loop through all parameters (including duplicates)
            for param in self.model._parameters_no_dedup():
                
                ### Get pointer to parameter 
                data_ptr = int(param.data._array.data.ptr)

                ### if we have seen this pointer before, reuse the same data
                if data_ptr in seen:
                    # reuse the same fp32 param for tied weights
                    fp32_param = seen[data_ptr]

                ### otherwise load like normal 
                else:
                    fp32_param = Tensor(param.data._array.copy(), dtype=float32)
                    fp32_param.requires_grad = param.requires_grad
                    seen[data_ptr] = fp32_param

                self.fp32_params.append(fp32_param)

            ### Now cast all our parameters to float16 ###
            for param in self.model._parameters_no_dedup():
                param.data._array = param.data._array.astype("float16")
       
        return self.model
    
    def prepare_optimizer(self, optimizer):

        accelerator = self 

        if not hasattr(self, "model"):
            raise Exception("No model found!! Make sure to run .prepare() on model as well!")

        if self.mixed_precision:
            """
            Mixed precision is pretty simple for the optimizer. What we want to do is train our model
            in fp16 but use fp32 weights. So how we do this is we do all of our computatiosn in fp16, 
            but we keep aside a copy of the weights in fp32 (this is what our optimizer sees) so when 
            we do grad updates, we arent going to have floating point errors (to a certain limit, our
            grads could have NANs but this is what the dynamic grad scaling handles!)
            """

            fp16_to_fp32 = {}
            model_parameters = list(self.model._parameters_no_dedup())

            ### Create a dictionary of ids to map our fp32 and fp16 params together ###
            ### We only do this because our parameters can be in different groups in our ###
            ### optimizer, so we need to make sure we are mapping the correct fp16 param in our ###
            ### model (which is just a list of all the parameters in our model) to its corresponding ###
            ### fp32 copy we kept aside in our `prepare_model()` method! ###
            for fp16_param, fp32_param in zip(model_parameters, self.fp32_params):
                fp16_to_fp32[int(fp16_param.data._array.data.ptr)] = fp32_param
            
            ### Now loop through optimizer groups ###
            for group in optimizer.param_groups:

                new_params = []
                for param in group["params"]:
                     
                    ### look up the cooresponding fp32 params ###
                    ptr = int(param.data._array.data.ptr)
                    if ptr in fp16_to_fp32:
                        new_params.append(fp16_to_fp32[ptr])
                    else:
                        raise ValueError("Parameter not found in fp32 mapping")
                    
                ### Our optimizer now is looking at the fp32 params only! ###
                group["params"] = new_params

        ### Reinit the Optimizer on the correct device (just incase) ###
        optimizer._init_optimizer_state(device=self.device)
      
        class OptimizerWrapper:
            def __init__(self, base_optimizer):
                self.base_optimizer = base_optimizer
            
            def step(self, *args, **kwargs):

                ### Only Step After Grad Accumulations are Done ###
                if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
                    ### In mixed precision we may have to skip a step ###
                    if accelerator.mixed_precision and accelerator.skip_optimizer_step:
                        ### Reset the flag and do nothing ###
                        accelerator.skip_optimizer_step = False
                        return 
                    
                    ### update our parameters ###
                    self.base_optimizer.step(*args, **kwargs)

                    ### If we just updated and were in mixed precision mode, we only updated ###
                    ### our fp32 copy of the weights. We need to copy those back into our model ###
                    ### now for the next iteration! It is important to use the copy() operation ###
                    ### as we dont want to cast our fp32_copy to fp16! ###
                    if accelerator.mixed_precision:
                        for fp32_param, param in zip(accelerator.fp32_params, accelerator.model._parameters_no_dedup()):
                            param.data[:] = fp32_param.data.astype(cp.float16, copy=True)

            def zero_grad(self, *args, **kwargs):   

                if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
                    self.base_optimizer.zero_grad(*args, **kwargs)

                    ### If in Mixed Precision ###
                    ### Remember, our optimizer looks at the copy of full precision weights ###
                    ### our model still looks at our half precision weights, so we need to manually ###
                    ### zero them out here ###
                    if accelerator.mixed_precision:
                        for param in accelerator.model.parameters():
                            if hasattr(param, "grad") and param.grad is not None:
                                param.grad[:] = 0.0

                        ### We have updated our model, reset flag for the next scaling ###
                        accelerator.scaler.reset_unscaled_flag()

            def __getattr__(self, name):
                return getattr(self.base_optimizer, name)

        ### Store access to optimizer object internally 
        self.optimizer = OptimizerWrapper(optimizer)  

        return self.optimizer
    
    def prepare_dataloaders(self, dataloader):

        if self.world_size <= 1:
            return dataloader
        
        accelerator = self

        class ShardDataset:
            def __init__(self, base_dataset, rank, world_size, shuffle=False):
                self.base = base_dataset
                self.rank = rank
                self.world_size = world_size
                self.shuffle = shuffle
                self.epoch = 0

                ### Number of Samples per Rank ###
                self.num_samples_per_rank = (len(self.base) + self.world_size - 1) // self.world_size
                self.total_size = self.num_samples_per_rank * self.world_size
             
                ### Initialize Indices ###
                ### This includes indexes for ALL samples ###
                self.indices = np.arange(len(self.base))

                ### Set Epoch ###
                self.set_epoch(self.epoch)

            def set_epoch(self, epoch):

                ### Per Epoch Reshuffle of Data Before Resharding ###
                self.epoch = epoch
                rand_gen = np.random.default_rng(seed=self.epoch)
                indices = np.arange(len(self.base))

                ### Random Shuffle Indices ###
                if self.shuffle:
                    indices = rand_gen.permutation(indices)
                    print(indices)
                ### Pad to make Divisible by World Size * Samples Per Rank ###
                ### This makes sure we have even number of batches every time ###
                if len(indices) < self.total_size:
                    padding = rand_gen.choice(indices, self.total_size - len(indices), replace=True)
                    indices = np.concatenate([indices, padding])

                self.indices = indices

            def __len__(self):
                """
                Remember this dataset is sharded, so each will only see a portion of self.indices
                """
                return (len(self.base) + self.world_size - 1) // self.world_size
        
            def __getitem__(self, idx):

                """
                Interleaved sampling:
                Dataset indices: 0 1 2 3 4 5 6 7 8 9 10 11, ...

                Rank 0 gets: 0, 4, 8, ...
                Rank 1 gets: 1, 5, 9, ...
                Rank 2 gets: 2, 6, 10, ...
                Rank 3 gets: 3, 7, 11, ...

                our idx is determined by __len__ and we have made sure our __len__ is limited
                to number of samples PER rank. So we need to just convert that to the actual
                index in the dataset
                """

                real_idx = self.indices[idx*self.world_size + self.rank]
                return self.base[real_idx]

        ### Grab Old Dataset ###
        shuffle_flag = getattr(dataloader, "shuffle", True) # Check if we were shuffling in the dataloader
        base_dataset = dataloader.dataset # grab that dataset

        ### Wrap the dataset with our sharded dataset ###
        ### The sharded dataset will both handle the separation of samples
        ### between ranks and also random shuffling (through set_epoch)
        sharded_dataset = ShardDataset(base_dataset, 
                                       world_size=self.world_size, 
                                       rank=self.rank, 
                                       shuffle=shuffle_flag)
        
        ### Replace our dataloaders dataset with this new sharded dataset 
        dataloader.dataset = sharded_dataset

        ### Dataset is already shuffling internally, we can disable shuffle on the dataloader then
        dataloader.shuffle = False

        ### Wrap Dataloader for Epoch Based Shuffling ###
        class EpochShuffledDataLoader:

            """
            This is a wrapper on our actual Dataloader that you can find in mytorch.utils.data::DataLoader

            There are a few things of note to keep in mind here:

            1: Dataset Indexes

            Lets say our dataset has 10 samples, then we have indexes 0-9 to represent them. But we have already
            wrapped our dataset above in a sharded dataset which splits the data by rank

            So the len(dataset) = 10 but len(shard_dataset) = 5 assuming we have 2 GPUs. Now the len(shard_dataset)
            may be 5, but we index back to the original 10 in our __getitem__. We store ahead of time some 
            self.indices that have all indexes for each item of the dataset (this can be preshuffled)

            ```
            real_idx = self.indices[idx*self.world_size + self.rank]
            return self.base[real_idx]
            ```

            Now this sharded_dataset is placed into our Dataloader

            ```
            dataloader.dataset = sharded_dataset
            ```

            But the dataloader uses len(self.dataset) to identify how many samples there are. So as far
            as the Dataloader is concerned it only has 5 samples in it (with again the dataset __getitem__
            containing the logic to return back to the original dataset index. 
            
            2. Two Sets of Indexes

            So now we have two sets of indexes. Our Dataloader thinks there are only 5 samples so it has the 
            indexes:

            [0, 1, 2, 3, 4]

            But our actual indexes (stored in self.sharded_dataset.indices) are

            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            So when the dataloader says, grab sample index 0 from [0, 1, 2, 3, 4], we do:

            real_idx = self.indices[idx*self.world_size + self.rank]

            so for GPU0: self.indices[0 * 2 + 0] = self.indices[0] = 0
            and for GPU1: self.indices[0*2 + 1] = self.indices[1] = 1

            So look! We have grabbed two samples (one per gpu) for every index in the DataLoader!

            3. Shuffling

            Shuffling is handled by the sharded dataset. So our self.sharded_dataset.indices could be:

            [2, 9, 5, 6, 1, 7, 4, 0, 3, 8]

            Now when we grab sample index 0 from [0, 1, 2, 3, 4], we do:

            real_idx = self.indices[idx*self.world_size + self.rank]

            so for GPU0: self.indices[0 * 2 + 0] = self.indices[0] = 1
            and for GPU1: self.indices[0*2 + 1] = self.indices[1] = 9

            For this to work two things needed to happen:

            - Random shuffle is seeded, all ranks must have the same shuffle. This is done in set_epoch!
            - our dataloader indexes [0, 1, 2, 3, 4] CANNOT be shuffled. These are basically an indicator
              that tells us, at index 0, grab the first two samples of our data, at index 1, grab the next
              two samples and so on. Shuffling is already done on the sharded_dataset indexes, no need to do it 
              again!

            4. Saving

            We want to be able to resume training. This should give a perfect resume if we use 0 dataloader workers
            but if we have multiple workers, then due to race conditions the order in which we load data into the 
            queue may be slightly different. So then it will be an approximate, where we may redo or skip a few samples

            This could probably be solved with a Priority queue but its probably not worth the effort!

            To save we create a STATE DICT

            return {
                    'epoch': self.epoch,
                    'batches_consumed': self.batches_consumed,
                    'dataloader_state': self.dataloader.state_dict() if hasattr(self.dataloader, 'state_dict') else None,
                    'sharded_dataset_epoch': self.sharded_dataset.epoch,
                    'sharded_dataset_indices': self.sharded_dataset.indices.tolist(),
                }

            This stores:
            - epoch: to reshuffle our indexes in set_epoch to exactly what it was when we stopped in this epoch
            - batches_consumed: how many batches have we already done in this dataset? So we can start from the new batch after that
            - dataloader_state: the internal dataloader state that has its own tracker. The main thing we care about here 
              is that it stores the indices [0, 1, 2, 3, 4]. This will repeat some of the metadatas we have (epoch, batches_consumed)
              but its fine, we can just leave it as is for simplicity as this doesnt take much space at all
            - sharded_dataset_indices: 


            """
            def __init__(self, dataloader):
                self.dataloader = dataloader # store dataloader inside 
                self.sharded_dataset = self.dataloader.dataset # grab internal sharded dataset
                self.epoch = 0 # how many epochs have we trained (for set_epoch)
                self.batches_consumed = 0 # How many batches have we consumed inside this epoch (for resuming)

            def __iter__(self):
                """
                Set epoch for consistent shuffling across ranks and iterate.
                If resuming mid-epoch, indices are already set from load_state_dict.
                """

                # Only regenerate indices if we're not resuming (indices already set)
                # if batches consumed is 0, we are at the start of an epoch so set the 
                # epoch for training. 
                if self.batches_consumed == 0:
                    self.sharded_dataset.set_epoch(self.epoch)
                
                # The underlying DataLoader's __iter__/__next__ will handle the skipping
                # based on its batches_consumed value
                return self._tracking_iterator(iter(self.dataloader))
            
            def _tracking_iterator(self, iterator):
                """
                Just a helper method that will catch StopIteration and 
                reset the epoch here
                """
                try:
                    while True:
                        batch = next(iterator)
                        self.batches_consumed += 1
                        ### Key code: We will not return batch but yield them. Ths way 
                        ### this loop will PAUSE until the next batch is requested
                        ### as for loops will internally keep calling next()
                        ### on this generator we just made
                        yield batch

                ### StopIteration is caught from the dataloader which means we finished, start a new epoch ###       
                except StopIteration:
                    self.reset_epoch()
                    return

            def __len__(self):
                """
                This is how many samples in our dataloader (which wraps a dataset that only has a partial portion of the data)
                thus this dataloader len will equivalently only see a part of the data
                """
                return len(self.dataloader)
            
            def state_dict(self):
                # return {
                #     'epoch': self.epoch,
                #     'batches_consumed': self.batches_consumed,
                #     'dataloader_state': self.dataloader.state_dict(),
                #     'sharded_dataset_epoch': self.sharded_dataset.epoch,
                #     'sharded_dataset_indices': self.sharded_dataset.indices.tolist(),
                # }
                return self.dataloader.state_dict()
            
            def load_state_dict(self, state_dict):
                """
                Resume from checkpoint.
                
                Args:
                    state_dict (dict): State from state_dict()
                """

                ### Set variables from state_dict
                self.epoch = state_dict['epoch']
                self.batches_consumed = state_dict["batches_consumed"]
                
                # Restore the sharded dataset state FIRST before loading dataloader state
                # self.sharded_dataset.epoch = state_dict.get('sharded_dataset_epoch', self.epoch)
                # if 'sharded_dataset_indices' in state_dict:
                #     self.sharded_dataset.indices = np.array(state_dict['sharded_dataset_indices'])

                ### We can just use set_epoch and this will set the self.shared_dataset.indices ###
                ### to exactly what it was for this epoch (seeded for random)
                self.sharded_dataset.set_epoch(self.epoch)
                
                # if state_dict['dataloader_state'] is not None and hasattr(self.dataloader, 'load_state_dict'):
                self.dataloader.load_state_dict(state_dict)

            def reset_epoch(self):
                self.epoch += 1
                self.batches_consumed = 0
                self.dataloader.reset_epoch()
                    
        new_loader = EpochShuffledDataLoader(dataloader)
        
        ### Create a list of all dataloaders to store in save_state
        if not hasattr(self, "dataloaders"):
            self.dataloaders = [new_loader]
        else:
            self.dataloaders.append(new_loader)

        return new_loader

    def backward(self, loss):

        ### Sanity check to skip updated if we have a NAN Loss ###
        if cp.isnan(loss.data._array).any() or cp.isinf(loss.data._array).any():
            if self.is_main_process:
                warnings.warn(f"Warning: NaN/Inf loss detected at step {self.step_counter}. Skipping backward.")
            self.step_counter += 1
            if self.step_counter % self.gradient_accumulation_steps == 0:
                self.skip_optimizer_step = True
            return
        
        ### If we are not in mixed precision mode, go ahead and scale. If we are in fp16 here ###
        ### and we divide by too many grad steps, we can underflow. We will do it for mixed precision ###
        ## later on! ###
        if not self.mixed_precision:
            ### Scale Loss By Gradient Accumulation Steps ###
            loss = loss/self.gradient_accumulation_steps

        ### If Mixed Precision We Scale our Loss ###
        if self.mixed_precision:

            ### We scale our loss up here! ###
            ### this way when we do .backward() the magnitude is large enough ###
            ### to keep us from underflowing ###
            loss = self.scaler.scale_loss(loss)

            ### Set flag that our grads here are scaled (False to unscaled) ###
            self.scaler.reset_unscaled_flag()

        ### Normal backward ###
        loss.backward()
        self.step_counter += 1

        ### We only update every self.gradient_accumulation_steps ###
        if self.step_counter % self.gradient_accumulation_steps == 0:
            
            ### Flag for if we want to skip updating incase of nan grads ###
            skip_step = False
            if self.mixed_precision:

                ### Scale down our accumulated gradients ###
                for param in self.model.parameters():
                    if hasattr(param, "grad") and param.grad is not None:
                        param.grad /= self.gradient_accumulation_steps

                ### Now we can unscale our gradients ###
                self.scaler.unscale_grads(self.model.parameters())

                ### Check for NAN/Inf on our unscaled grads ###
                found_inf = 0.0
                for param in self.model.parameters():
                    if hasattr(param, "grad") and param.grad is not None:
                        if cp.any(cp.isinf(param.grad)) or cp.any(cp.isnan(param.grad)):
                            found_inf = 1.0
                            break

                ### Gather from all devices ###
                found_inf_arr = cp.array([found_inf])
                if self.comm is not None:
                    out = cp.empty_like(found_inf_arr)
                    self.comm.all_reduce(found_inf_arr, out, op="max")
                    found_inf = out[0] > 0.0
                
                if found_inf:
                    
                    ### If we found inf, just go ahead and zero out ###
                    ### all the grad as we wont be updating in this step ###
                    for param in self.model.parameters():
                        if hasattr(param, "grad") and param.grad is not None:
                            param.grad[:] = 0.0
                    
                    ### Let out scaler know to adjust the scaling accordingly ###
                    self.scaler.update(True)
                    skip_step = True

                    if self.is_main_process:
                        warnings.warn(f"Warning: NaN/Inf Grad detected at step {self.step_counter}. Reducing Scale to {self.scaler.scale} and Skipping backward.")

                else:
                    ### If no issues, just update our scaler accordingly ###
                    self.scaler.update(False)

            ### Allreduce Gradients ###
            if self.comm is not None:
                for param in self.model.parameters():
                    if hasattr(param, "grad") and param.grad is not None:

                        ### Quick check. In our backward pass there are two options:
                        ### - Auto backward which will use our Array type
                        ### - Manual backward which will use either cp.ndarray or np.ndarray. But 
                        ###   we are in distributed training here so we only care about cp.ndarray

                        ### This means if we have an Array type we need to get the "_array" that hold the 
                        ### actual underlying data in Cupy for NCCL all_reduce. But if its already a 
                        ### cp.ndarray theres nothing to get, so we just have a quick sanity check here
                        self.comm.all_reduce(param.grad._array/self.world_size if hasattr(param.grad, "_array") else param.grad/self.world_size, # <- in_array
                                             param.grad._array/self.world_size if hasattr(param.grad, "_array") else param.grad/self.world_size, # <- out_array
                                             op="sum")
                        
            ### Cast Grads back to FP32 to Update our FP32 Copy of Weights ###
            if self.mixed_precision and not skip_step:
        
                ### Dictionary of parameter pointers and their grads ###
                seen_params = {}
                
                for fp32_param, param in zip(self.fp32_params, self.model._parameters_no_dedup()):
                    if hasattr(param, "grad") and param.grad is not None:
                        
                        ### Get pointer to parameter ###
                        param_ptr = int(param.data._array.data.ptr)
                        
                        ### If we've seen this gradient before, reuse the same fp32 grad ###
                        if param_ptr in seen_params:
                            fp32_param.grad = seen_params[param_ptr].astype(cp.float32)
                        else:
                            ### Create new fp32 gradient and store it ###
                            fp32_grad = param.grad.astype(cp.float32)
                            fp32_param.grad = fp32_grad
                            seen_params[param_ptr] = fp32_grad
                    else:
                        fp32_param.grad = None
  
            self.skip_optimizer_step = skip_step

    def clip_grad_norm_(self, max_norm=1.0):
        
        ### Only Clip Norm when Accumulation Complete ###
        if self.step_counter % self.gradient_accumulation_steps != 0:
            return None
        
        ### If we skipped our optimizer step earlier in backward() we will also skip our clipping ###
        if self.skip_optimizer_step:
            return None
        
        ### If we are in Mixed Precision Mode ###
        ### This wont actually do anything as we already unscaled our grads ###
        ### in the backward pass. Assuming we are just called clipping after backward ###
        ### this is basically a no-op
        if self.mixed_precision:
            self.scaler.unscale_grads(self.model.parameters())
        
        ### Compute local norm squared ###
        # local_norm_sq = 0.0
        # for param in self.model.parameters():
        #     if hasattr(param, "grad") and param.grad is not None:
        #         norm = float(cp.sum(param.grad ** 2))
        #         local_norm_sq += norm

        ### Same as above but avoids multiple kernel launches ###
        local_norm_sq = 0.0
        grads = [param.grad for param in self.model.parameters() if hasattr(param, "grad") and param.grad is not None]
        if grads:
            flat_grads = cp.concatenate([g.reshape(-1) for g in grads])
            local_norm_sq = float(cp.sum(flat_grads ** 2))
        else:
            local_norm_sq = 0.0
 
        ### AllReduce to get global norm squared ###
        if self.comm is not None:
            local_norm_sq_arr = cp.array([local_norm_sq])
            global_norm_sq_arr = cp.empty_like(local_norm_sq_arr)
            self.comm.all_reduce(local_norm_sq_arr, global_norm_sq_arr, op="sum")
            total_norm = float(cp.sqrt(global_norm_sq_arr[0]))
        else:
            total_norm = float(cp.sqrt(local_norm_sq))

        ### Clip based on GLOBAL norm ###
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in self.model.parameters():
                if hasattr(param, "grad") and param.grad is not None:
                    param.grad *= clip_coef

        ### Copy to fp32_params ###
        if self.mixed_precision:
            for fp32_param, param in zip(self.fp32_params, self.model.parameters()):
                if hasattr(param, "grad") and param.grad is not None:
                    fp32_param.grad = param.grad.astype(cp.float32)
                else:
                    fp32_param.grad = None
        
        self._grad_norm = total_norm

    def gather_for_metrics(self, value):
        assert isinstance(value, Tensor), "Value must be a Tensor"
        assert value.shape == (), "Value must be a Scalar"
        
        if self.world_size == 1 or self.comm is None:
            return float(value.data._array)
        
        ### Cast to float32 for stability in allreduce ###
        data = value.data._array if hasattr(value.data, "_array") else value.data
        data_f32 = data.astype(cp.float32) if data.dtype == cp.float16 else data
        
        out = cp.zeros_like(data_f32)
        self.comm.all_reduce(data_f32, out, op="sum")
        return float(out / self.world_size)
    
    def wait_for_everyone(self):
        if self.comm is not None:
            self.comm.barrier()
    
    def end_training(self):
        if self.comm is not None:
            self.comm.barrier()
            self.comm.stop()

    def print(self, *args, **kwargs):
        if self.is_main_process():
            print(*args, **kwargs)

    def init_tracker(self,
                     project_name, 
                     run_name, 
                     config=None):

        if config is not None:
            assert isinstance(config, dict), "Config must be a dictionary!"
        
        if not self.wandb_enabled:
            warnings.warn("log_wandb not enabled at init of Accelerator, doing it now!")
            self.wandb_enabled = True

        if self.rank == 0:
            wandb.init(
                project=project_name,
                name=run_name,
                config=config
            )

    def log(self, log_dict, step):

        assert isinstance(log_dict, dict), "log_dict must be dictionary!"
        assert isinstance(step, int), "step must be integer!"

        if not self.wandb_enabled:
            raise Exception("Wandb not enabled! Currently only Wandb is supported for automatic tracking")
        
        if self.rank == 0:
            wandb.log(log_dict, step=step)

    def save_state(self, path_to_checkpoint, save_model_only=False):

        """
        Checkpoint model weights and optimizer
        """

        if self.is_main_process():

            print(f"Saving Checkpoint to {path_to_checkpoint}")    

            if hasattr(self, "model"):

                ### Create checkpoint directory
                os.makedirs(path_to_checkpoint, exist_ok=True)

                ### Get Model Weights ###
                model_state = {}
    
                for i, (name, param) in enumerate(self.model._named_parameters_no_dedup()):
                    
                    ### if in mixed precision we can grab our full precision parameters ###
                    ### from our stored internal buffer ###
                    if self.mixed_precision:
                        fp32_param = self.fp32_params[i]
                        
                        model_state[name] = cp.asnumpy(fp32_param.data._array)
                    else:
                        model_state[name] = cp.asnumpy(param.data._array)

                save_file(model_state, os.path.join(path_to_checkpoint, "model.safetensors"))

            if not save_model_only:
                
                ### Get Optimizer states ###
                if hasattr(self, "optimizer") and self.optimizer is not None:
    
                    opt_state = self.optimizer.state_dict()
                    if opt_state is not None:
                        opt_path = os.path.join(path_to_checkpoint, "optimizer.bin")
                        with open(opt_path, "wb") as f:
                            pickle.dump(opt_state, f)

                if self.mixed_precision:

                    mixed_precision_config = {"scale": self.scaler.scale,
                                              "growth_factor": self.scaler.growth_factor,
                                              "backoff_factor": self.scaler.backoff_factor, 
                                              "growth_interval": self.scaler.growth_interval, 
                                              "unskipped": self.scaler.unskipped}

                    mp_config_path = os.path.join(path_to_checkpoint, "mp_config.bin")
                    with open(mp_config_path, "wb") as f:
                        pickle.dump(mixed_precision_config, f)

                if hasattr(self, "dataloaders"):
                    assert isinstance(self.dataloaders, list)
                    ### Loop through all prepped dataloaders and save state ###
                    for idx, loader in enumerate(self.dataloaders):
                        path_to_loader_save = os.path.join(path_to_checkpoint, f"dataloader_{idx}.bin")
                        with open(path_to_loader_save, "wb") as f:
                            pickle.dump(loader.state_dict(), f)

        self.wait_for_everyone()
    
    def load_state(self, path_to_checkpoint):
        
        self.print(f"Loading Checkpoint From {path_to_checkpoint}")

        ### Load model ###
        path_to_model = os.path.join(path_to_checkpoint, "model.safetensors")
        if os.path.exists(path_to_model):
            model_state = load_file(path_to_model)

            ### Copy Weights in ###
            for i, (name, param) in enumerate(self.model._named_parameters_no_dedup()):

                weights = model_state[name]

                ### If we are in mixed precision we need to update both our model weights 
                ### in fp16 and our fp32 buffer
                if self.mixed_precision:
                    self.fp32_params[i].data._array[:] = cp.asarray(weights, dtype=cp.float32)
                    param.data._array[:] = self.fp32_params[i].data._array.astype(cp.float16)
                
                else:
                    param.data._array[:] = cp.asarray(weights, dtype=param.data._array.dtype)   

            ### Ensure all GPUs have same starting point ###
            if self.comm is not None:
                for param in self.model.parameters():
                    self.comm.broadcast(param.data._array, root=0)
                if self.mixed_precision:
                    for fp32_param in self.fp32_params:
                        self.comm.broadcast(fp32_param.data._array, root=0)

        ### Load Optimizer ###
        path_to_optimizer = os.path.join(path_to_checkpoint, "optimizer.bin")
        if os.path.exists(path_to_optimizer):
            with open(path_to_optimizer, "rb") as f:
                opt_state = pickle.load(f)
            
            if self.optimizer is not None:
                self.optimizer.load_state_dict(opt_state)

        if self.mixed_precision:
            path_to_mp_config = os.path.join(path_to_checkpoint, "mp_config.bin")
            if os.path.exists(path_to_mp_config):
                with open(path_to_mp_config, "rb") as f:
                    mp_config = pickle.load(f)

                self.scaler.scale = mp_config["scale"]
                self.scaler.growth_factor = mp_config["growth_factor"]    
                self.scaler.backoff_factor = mp_config["backoff_factor"]    
                self.scaler.growth_interval = mp_config["growth_interval"]    
                self.scaler.unskipped = mp_config["unskipped"]    

                self.print(f"Loading from Mixed Precision Training. Setting starting scale as {self.scaler.scale}")

            else:
                warnings.warn("If you are resuming mixed precision training without its checkpointed config, you may have instability in training")

        ### Load Dataloaders ###
        if hasattr(self, "dataloaders"):
            assert isinstance(self.dataloaders, list)

            for idx, loader in enumerate(self.dataloaders):
                path_to_loader_save = os.path.join(path_to_checkpoint, f"dataloader_{idx}.bin")
                if os.path.exists(path_to_loader_save):
                    with open(path_to_loader_save, "rb") as f:
                        data_state_dict = pickle.load(f)
                    loader.load_state_dict(data_state_dict)
                else:
                    warnings.warn(f"Couldn't find {path_to_loader_save}, cannot resume dataloader, starting from scratch!!")

    def __del__(self):
        if self.comm is not None:
            self.comm.stop()