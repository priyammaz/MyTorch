"""
DDP trainer inspired by Huggingface Accelerate!

This should work regardless of if we are training on DDP/Single GPU or
if we are doing Mixed/Full precision training! If everything is disabled
its just a fancy way to log and checkpoint essentially!
"""
import os
import cupy as cp
import numpy as np
from cupyx.distributed import NCCLBackend
import mytorch
import pickle
from safetensors.numpy import save_file, load_file
import warnings

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
                 master_addr="127.0.0.1", 
                 master_port="13333",
                 gradient_accumulation_steps=1,
                 mixed_precision=False,
                 log_wand=False):
        
        ### Set Number of GPUs if not provided from environment ###
        self.rank = rank if rank is not None else int(os.environ.get("RANK", 0))
        self.world_size = num_gpus if num_gpus is not None else int(os.environ.get("WORLD_SIZE", 1))

        ### Set Address and Port ####
        self.master_addr = master_addr
        self.master_port = master_port

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
        self.mixed_precision = mixed_precision
        self.skip_optimizer_step = False
        if self.mixed_precision:
            self.scaler = GradScaler()

        ### Store Grad Norm for every grad sync ###
        ### this lets us keep an eye on model training health ###
        self._grad_norm = None

        ### Import wandb if logging with it ###
        self.wandb_enabled = False
        if log_wand:
            self.wandb_enabled = True
            import wandb

    def is_main_process(self):
        return self.rank == 0
    
    @property
    def device(self):
        return f"cuda:{self.rank}"
    
    @property
    def sync_grad(self):
        return self.step_counter % self.gradient_accumulation_steps == 0
    
    @property 
    def grad_norm(self):
        return self._grad_norm
    
    def prepare(self, *args):

        prepared = []

        for obj in args:
            if isinstance(obj, mytorch.nn.Module):
                prepared.append(self.prepare_model(obj))
            elif isinstance(obj, mytorch.optim.Optimizer):
                prepared.append(self.prepare_optimizer(obj))
            elif isinstance(obj, mytorch.data.DataLoader):
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
                    fp32_param = mytorch.Tensor(param.data._array.copy(), dtype=mytorch.float32)
                    fp32_param.requires_grad = param.requires_grad
                    seen[data_ptr] = fp32_param

                self.fp32_params.append(fp32_param)

            ### Now cast all our parameters to float16 ###
            for param in self.model._parameters_no_dedup():
                param.data._array = param.data._array.astype("float16")
        
        return self.model
    
    def prepare_optimizer(self, optimizer):

        accelerator = self 

        ### We will update Full Precision Params but train w/ Half Precision ###
        if self.mixed_precision:
            optimizer.params = self.fp32_params
            
        ### Adam has momentum params that have already been initialized ###
        ### we need to reinit them on the correct device ###
        if hasattr(optimizer, "m"):
            optimizer.m = [mytorch.zeros_like(p).data for p in optimizer.params]
        if hasattr(optimizer, "v"):
            optimizer.v = [mytorch.zeros_like(p).data for p in optimizer.params]

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
                    ### now for the next iteration! ###
                    if accelerator.mixed_precision:
                        for fp32_param, param in zip(accelerator.fp32_params, accelerator.model._parameters_no_dedup()):
                            param.data = fp32_param.data.astype(cp.float16)

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

        class ShardDataset:
            def __init__(self, base_dataset, rank, world_size, shuffle=True):
                self.base = base_dataset
                self.rank = rank
                self.world_size = world_size
                self.shuffle = shuffle
                self.epoch = 0

                ### Number of Samples per Rank ###
                self.num_samples_per_rank = (len(self.base) + self.world_size - 1) // self.world_size
                self.total_size = self.num_samples_per_rank * self.world_size

                ### Initialize Indices ###
                self.indices = np.arange(len(self.base))

            def set_epoch(self, epoch):

                ### Per Epoch Reshuffle of Data Before Resharding ###
                self.epoch = epoch
                rand_gen = np.random.default_rng()
                indices = np.arange(len(self.base))

                ### Random Shuffle Indices ###
                if self.shuffle:
                    indices = rand_gen.permutation(indices)

                ### Pad to make Divisible by World Size * Samples Per Rank ###
                ### This makes sure we have even number of batches every time ###
                if len(indices) < self.total_size:
                    padding = rand_gen.choice(indices, self.total_size - len(indices), replace=True)
                    indices = np.concatenate([indices, padding])

                self.indices = indices

            def __len__(self):
                return (len(self.base) + self.world_size - 1) // self.world_size
        
            def __getitem__(self, idx):

                """
                Interleaved sampling:
                Dataset indices: 0 1 2 3 4 5 6 7 8 9 10 11, ...

                Rank 0 gets: 0, 4, 8, ...
                Rank 1 gets: 1, 5, 9, ...
                Rank 2 gets: 2, 6, 10, ...
                Rank 3 gets: 3, 7, 11, ...
                """
                real_idx = self.indices[idx*self.world_size + self.rank]
                return self.base[real_idx]
        
        ### Grab Old Dataset ###
        shuffle_flag = getattr(dataloader, "shuffle", True)
        base_dataset = dataloader.dataset
        sharded_dataset = ShardDataset(base_dataset, world_size=self.world_size, rank=self.rank, shuffle=shuffle_flag)
        dataloader.dataset = sharded_dataset

        ### Wrap Dataloader for Epoch Based Shuffling ###
        class EpochShuffledDataLoader:
            def __init__(self, dataloader, sharded_dataset):
                self.dataloader = dataloader
                self.sharded_dataset = sharded_dataset
                self.epoch = 0

            def __iter__(self):
                self.sharded_dataset.set_epoch(self.epoch)
                self.epoch += 1
                return iter(self.dataloader)

            def __len__(self):
                return len(self.dataloader)
            
        return EpochShuffledDataLoader(dataloader, sharded_dataset)
    
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
                        out = cp.empty_like(param.grad)

                        ### Quick check. In our backward pass there are two options:
                        ### - Auto backward which will use our Array type
                        ### - Manual backward which will use either cp.ndarray or np.ndarray. But 
                        ###   we are in distributed training here so we only care about cp.ndarray

                        ### This means if we have an Array type we need to get the "_array" that hold the 
                        ### actual underlying data in Cupy for NCCL all_reduce. But if its already a 
                        ### cp.ndarray theres nothing to get, so we just have a quick sanity check here
                        self.comm.all_reduce(param.grad._array if hasattr(param.grad, "_array") else param.grad, out, op="sum")
                        param.grad[:] = out / self.world_size

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
                            fp32_param.grad = seen_params[param_ptr]
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
        local_norm_sq = 0.0
        for param in self.model.parameters():
            if hasattr(param, "grad") and param.grad is not None:
                norm = float(cp.sum(param.grad ** 2))
                local_norm_sq += norm
 
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
        assert isinstance(value, mytorch.Tensor), "Value must be a Tensor"
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

    def save_state(self, path_to_checkpoint):

        """
        Checkpoint model weights and optimizer
        """

        if self.is_main_process():

            print(f"Saving Checkpoint to {path_to_checkpoint}")    

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

            ### Get Optimizer states ###
            if hasattr(self, "optimizer") and self.optimizer is not None:
 
                opt = self.optimizer
                opt_state = {}

                if hasattr(opt, "m"):
                    opt_state["m"] = [cp.asnumpy(m) for m in opt.m]
                if hasattr(opt, "v"):
                    opt_state["v"] = [cp.asnumpy(v) for v in opt.v]
                if hasattr(opt, "t"):
                    opt_state["t"] = opt.t
                if hasattr(opt, "beta1_pow"):
                    opt_state["beta1_pow"] = opt.beta1_pow
                if hasattr(opt, "beta2_pow"):
                    opt_state["beta2_pow"] = opt.beta2_pow

                if len(opt_state) > 0:
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

        self.wait_for_everyone()
    
    def load_state(self, path_to_checkpoint):
        
        self.print(f"Loading Checkpoint From {path_to_checkpoint}")

        ### Load model ###
        path_to_model = os.path.join(path_to_checkpoint, "model.safetensors")
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

                if hasattr(self.optimizer, "m") and "m" in opt_state:
                    self.optimizer.m = [cp.asarray(m) for m in opt_state["m"]]

                    if self.comm is not None:
                        for m in self.optimizer.m:
                            self.comm.broadcast(m)

                if hasattr(self.optimizer, "v") and "v" in opt_state:
                    self.optimizer.v = [cp.asarray(v) for v in opt_state["v"]]

                    if self.comm is not None:
                        for v in self.optimizer.v:
                            self.comm.broadcast(v)

                if hasattr(self.optimizer, "t") and "t" in opt_state:
                    self.optimizer.t = opt_state["t"]
                if hasattr(self.optimizer, "beta1_pow") and "beta1_pow" in opt_state:
                    self.optimizer.beta1_pow = opt_state["beta1_pow"]
                if hasattr(self.optimizer, "beta2_pow") and "beta2_pow" in opt_state:
                    self.optimizer.beta2_pow = opt_state["beta2_pow"]

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

    def __del__(self):
        if self.comm is not None:
            self.comm.stop()

# class Accelerator:
#     def __init__(self, 
#                  num_gpus=None, 
#                  rank=None, 
#                  master_addr="127.0.0.1", 
#                  master_port="13333",
#                  gradient_accumulation_steps=1):
        
#         ### Set Number of GPUs if not provided from environment ###
#         self.rank = rank if rank is not None else int(os.environ.get("RANK", 0))
#         self.world_size = num_gpus if num_gpus is not None else int(os.environ.get("WORLD_SIZE", 1))

#         ### Set Address and Port ####
#         self.master_addr = master_addr
#         self.master_port = master_port

#         ### Set Device for this rank ###
#         cp.cuda.Device(self.rank).use()
        
#         ### Accumulation ###
#         self.gradient_accumulation_steps = gradient_accumulation_steps
#         self.step_counter = 0

#         ### Initialize NCCL ###
#         self.comm = None
#         if self.world_size > 1:
#             self.comm = NCCLBackend(
#                 n_devices=self.world_size, 
#                 rank=self.rank, 
#                 host=self.master_addr, 
#                 port=int(self.master_port)
#             )

#         ### Random Seed per Rank ###
#         cp.random.seed(seed=42 + self.rank)

#     def is_main_process(self):
#         return self.rank == 0
    
#     def prepare(self, *args, **kwargs):

#         prepared = []

#         for obj in args:
#             if isinstance(obj, mytorch.nn.Module):
#                 prepared.append(self.prepare_model(obj))
#             elif isinstance(obj, mytorch.optim.Optimizer):
#                 prepared.append(self.prepare_optimizer(obj))
#             elif isinstance(obj, mytorch.data.DataLoader):
#                 prepared.append(self.prepare_dataloaders(obj))
        
#         return prepared
    
#     def prepare_model(self, model):

#         ### Store Access to Model ###
#         self.model = model

#         ### Broadcast Weights from Model into Other GPUs ###
#         if self.comm is not None:
#             for param in self.model.parameters():
#                 self.comm.broadcast(param.data, root=0)
        
#         return self.model
    
#     def prepare_optimizer(self, optimizer):

#         accelerator = self 

#         class OptimizerWrapper:
#             def __init__(self, base_optimizer):
#                 self.base_optimizer = base_optimizer
            
#             def step(self, *args, **kwargs):
#                 if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
#                     return self.base_optimizer.step(*args, **kwargs)
            
#             def zero_grad(self, *args, **kwargs):
#                 if accelerator.step_counter % accelerator.gradient_accumulation_steps == 0:
#                     return self.base_optimizer.zero_grad(*args, **kwargs)
            
#             def __getattr__(self, name):
#                 return getattr(self.base_optimizer, name)
            
#         return OptimizerWrapper(optimizer)
    
#     def prepare_dataloaders(self, dataloader):

#         if self.world_size <= 1:
#             return dataloader

#         class ShardDataset:
#             def __init__(self, base_dataset, rank, world_size, shuffle=True):
#                 self.base = base_dataset
#                 self.rank = rank
#                 self.world_size = world_size
#                 self.shuffle = shuffle
#                 self.epoch = 0

#                 ### Number of Samples per Rank ###
#                 self.num_samples_per_rank = (len(self.base) + self.world_size - 1) // self.world_size
#                 self.total_size = self.num_samples_per_rank * self.world_size

#                 ### Initialize Indices ###
#                 self.indices = np.arange(len(self.base))

#             def set_epoch(self, epoch):

#                 ### Per Epoch Reshuffle of Data Before Resharding ###
#                 self.epoch = epoch
#                 rand_gen = np.random.default_rng()
#                 indices = np.arange(len(self.base))

#                 ### Random Shuffle Indices ###
#                 if self.shuffle:
#                     indices = rand_gen.permutation(indices)

#                 ### Pad to make Divisible by World Size * Samples Per Rank ###
#                 ### This makes sure we have even number of batches every time ###
#                 if len(indices) < self.total_size:
#                     padding = rand_gen.choice(indices, self.total_size - len(indices), replace=True)
#                     indices = np.concatenate([indices, padding])

#                 self.indices = indices

#             def __len__(self):
#                 return (len(self.base) + self.world_size - 1) // self.world_size
        
#             def __getitem__(self, idx):

#                 """
#                 Interleaved sampling:
#                 Dataset indices: 0 1 2 3 4 5 6 7 8 9 10 11, ...

#                 Rank 0 gets: 0, 4, 8, ...
#                 Rank 1 gets: 1, 5, 9, ...
#                 Rank 2 gets: 2, 6, 10, ...
#                 Rank 3 gets: 3, 7, 11, ...
#                 """
#                 real_idx = self.indices[idx*self.world_size + self.rank]
#                 return self.base[real_idx]
        
#         ### Grab Old Dataset ###
#         shuffle_flag = getattr(dataloader, "shuffle", True)
#         base_dataset = dataloader.dataset
#         sharded_dataset = ShardDataset(base_dataset, world_size=self.world_size, rank=self.rank, shuffle=shuffle_flag)
#         dataloader.dataset = sharded_dataset

#         ### Wrap Dataloader for Epoch Based Shuffling ###
#         class EpochShuffledDataLoader:
#             def __init__(self, dataloader, sharded_dataset):
#                 self.dataloader = dataloader
#                 self.sharded_dataset = sharded_dataset
#                 self.epoch = 0

#             def __iter__(self):
#                 self.sharded_dataset.set_epoch(self.epoch)
#                 self.epoch += 1
#                 return iter(self.dataloader)

#             def __len__(self):
#                 return len(self.dataloader)
            
#         return EpochShuffledDataLoader(dataloader, sharded_dataset)
    
#     def backward(self, loss):

#         ### Scale Loss By Gradient Accumulation Steps ###
#         loss = loss/self.gradient_accumulation_steps

#         ### Normal backward ###
#         loss.backward()
#         self.step_counter += 1

#         if self.step_counter % self.gradient_accumulation_steps == 0:
#             ### Allreduce Gradients ###
#             if self.comm is not None:
#                 for param in self.model.parameters():
#                     if hasattr(param, "grad") and param.grad is not None:
#                         out = cp.empty_like(param.grad)
#                         self.comm.all_reduce(param.grad, out, op="sum")
#                         param.grad[:] = out / self.world_size

#     def clip_grad_norm_(self, max_norm=1.0):
        
#         ### Only Clip Norm when Accumulation Complete ###
#         if self.step_counter % self.gradient_accumulation_steps != 0:
#             return None
        
#         ### Compute Total Norm across
#         total_norm = 0.0
#         for param in self.model.parameters():
#             if hasattr(param, "grad") and param.grad is not None:
#                 total_norm += float(cp.linalg.norm(param.grad.reshape(-1), ord=2.0)) ** 2
#         total_norm = total_norm ** 0.5

#         clip_coef = max_norm / (total_norm + 1e-6)
#         if clip_coef < 1.0:
#             for param in self.model.parameters():
#                 if hasattr(param, "grad") and param.grad is not None:
#                     param.grad *= clip_coef
        
#     def gather_for_metrics(self, value):
 
#         assert isinstance(value, mytorch.Tensor), "Value must be a Tensor"
#         assert value.shape == (), "Value must be a Scalar"

#         if self.world_size <= 1 or self.comm is None:
#             return float(value.data[0])

#         out = cp.zeros_like(value.data)
#         self.comm.all_reduce(value.data, out, op="sum")
#         return out.item() / self.world_size
    
#     def wait_for_everyone(self):
#         self.comm.barrier()
    
#     def end_training(self):
#         self.comm.barrier()
#         self.comm.stop()

#     def print(self, *args, **kwargs):
#         if self.is_main_process():
#             print(*args, **kwargs)

#     def init_tracker(self,
#                      project_name, 
#                      run_name, 
#                      config=None):

#         if config is not None:
#             assert isinstance(config, dict), "Config must be a dictionary!"
            
#         if self.rank == 0:
#             wandb.init(
#                 project=project_name,
#                 name=run_name,
#                 config=config
#             )

#     def log(self, log_dict, step):

#         assert isinstance(log_dict, dict), "log_dict must be dictionary!"
#         assert isinstance(step, int), "step must be integer!"

#         if self.rank == 0:
#             wandb.log(log_dict, step=step)

#     def __del__(self):
#         if self.comm is not None:
#             self.comm.stop()


        # if self.step_counter % self.gradient_accumulation_steps == 0:

        #     ### Check for any NAN Gradients ###

        #     ### Flag if we want to skip grad update incase we find nan ###
        #     skip_step = False

        #     ### If in mixed precision we do our check 3##
        #     if self.mixed_precision:

        #         ### 0.0 acts as False for any inf grads ###
        #         found_inf = 0.0

        #         ### Loop through our parameters ###
        #         for param in self.model.parameters():

        #             ### Check for 
        #             if hasattr(param, "grad") and param.grad is not None:
        #                 ### Detect INF and NAN (If we do set as 1.0) ###
        #                 if cp.any(cp.isinf(param.grad)) or cp.any(cp.isnan(param.grad)):
        #                     found_inf = 1.0
        #                     break
                
        #         ### Gather from all devices to see if ANY device had an NAN/INF ###
        #         found_inf_arr = cp.array([found_inf])
        #         if self.comm is not None:
        #             out = cp.empty_like(found_inf_arr)
        #             self.comm.all_reduce(found_inf_arr, out, op="max")
        #         found_inf = found_inf_arr > 0.0

        #         if found_inf:
        #             ### Zero Scaled Grads as its is not a goo dupdate ###
        #             for param in self.model.parameters():
        #                 if hasattr(param, "grad") and param.grad is not None:
        #                     param.grad[:] = 0.0
        #             self.scaler.update(True)
        #             skip_step = True
                
        #         else:
        #             self.scaler.unscale_grads(self.model.parameters())
        #             self.scaler.update(False)
