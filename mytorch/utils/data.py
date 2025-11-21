"""
A whatever Dataloader. I did not have fun making this, im pretty 
sure it works good enough, probably could be better, but I also cant really bother
to make it better. 
"""
from mytorch.tensor import arange
from mytorch.ops import stack
import numpy as np
import threading
import queue

class Dataset:
    def __init__(self):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, idx):
        raise NotImplementedError
    
class DataLoader:
    """
    Minimal multithreaded DataLoader with optional collate function.

    Features:
    - Supports single-threaded and multi-threaded batch loading
    - Optional shuffling at the start of each epoch
    - Prefetch queue for asynchronous loading
    - Custom collate function or a default stack-based collator
    - Clean worker shutdown
    - __len__ gives number of batches per epoch
    """

    def __init__(self, 
                 dataset, 
                 batch_size=1, 
                 shuffle=False, 
                 num_workers=0, 
                 prefetch_factor=2, 
                 collate_fn=None,
                 timeout=30):

        """
        Args:
            dataset: Any object implementing __len__ and __getitem__.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle dataset indices at start of each epoch.
            num_workers (int): Number of background worker threads. 0 = no multithreading.
            prefetch (int): Number of batches to prefetch in the queue.
            collate_fn (callable): Function to merge list of samples into a batch.
                                   Defaults to stacking along axis 0.
            timeout (float): Amount of seconds you will wait for batches to be grabbed before exiting
        """

        self.dataset = dataset ### Iterable Dataset
        self.batch_size = batch_size ### Batch Size
        self.shuffle = shuffle ### Do we want to shuffle?
        self.num_workers = num_workers ### Number of workers 
        self.prefetch = prefetch_factor  # number of batches to prefetch
        self.collate_fn = collate_fn if collate_fn is not None else self.default_collate
        self.timeout = timeout ### How long to wait for timeout error
        self.indices = arange(len(dataset)) ### indexes from 0 -> dataset length

        ### if we have multiple workers, we can prefetch data in the background and store in a queue
        ### we should have technically done a priority queue here so we can fetch in an exact order (if not shuffled)
        ### due to race conditions but this is close enough!
        if self.num_workers > 0:
            self.queue = queue.Queue(maxsize=prefetch_factor * num_workers)
        else:
            self.queue = None

        self.workers = []
        self.lock = threading.Lock() # Can use to makes sure before a thread can access a shared resource it must aquire a lock
        self.stop_signal = threading.Event() # enables inter-thread comm. Initialized as true, can be toggled
        self.current_idx = 0 # Current index of dataset that we are on, that will be advanced by batch_size every iteration
        self.worker_finished_count = 0 # If we spawn workers, how many are running? Important when we want to kill workers

        ### State tracking ###
        self.epoch = 0 # What epoch are we on? If we do multiple full passes through the dataloader we have multiple epoch
        self.batches_consumed = 0 # Inside an epoch how many batches have we consumed?
        self._shuffled_indices = None # If we shuffle our indexes we will store them here
        self._resume_from_state = False # Flag for if we are resuming from a state

    def default_collate(self, batch):
        """
        Default collate function: stacks each element along axis 0.
        Handles datasets returning tuples/lists per sample.
        """

        if isinstance(batch[0], (tuple, list)):
            return tuple(self.default_collate([b[i] for b in batch]) for i in range(len(batch[0])))
        else:
            return stack(batch)
    
    def _get_next_batch_indices(self):

        """
        Grab next slice of indexes for sampling
        """
        
        ### Thread safe lock to make sure multiple workers dont grab the same indices ###
        ### at the same time!!! They have to be given permission (the lock) to ###
        ### even enter this chunk of code. Otherwise they wait for their turn ###
        with self.lock:

            ### If our current index goes past how many samples we have in our data we are done
            if self.current_idx >= len(self.dataset):
                
                ### If we have no workers then we are only running on the main thread. Lets just
                ### return None so we can catch it in our __next__ and stop iteration
                if self.num_workers == 0:
                    return None
                
                ### If we have multiple workers, one of them will catch this. We can raise the 
                ### Stopiteration so the _worker_loop catches it and can send it to the main thread
                else:
                    raise StopIteration
            
            ### end_idx makes sure we dont go over the end of our dataset
            end_idx = min(self.current_idx + self.batch_size, len(self.dataset))

            ### Grab the batch of indexes ###
            batch_indices = self.indices[self.current_idx:end_idx].tolist()
                        
            ### Set the current index ###
            self.current_idx = end_idx
            
        return batch_indices

    def _worker_loop(self):
        
        ### This worker_loop only applies if self.workers > 0
        ### While there is data keep storing in the queue ###
        try:

            ### as long as our self.stop_signal is False keep going ###
            while not self.stop_signal.is_set():
                
                ### Catch the stop iteration from _next_get_batch_indices
                try:
                    batch_indices = self._get_next_batch_indices()
                    batch = [self.dataset[i] for i in batch_indices]
                    batch_array = self.collate_fn(batch)

                    ### Check stop signal before putting ###
                    if not self.stop_signal.is_set():
                        self.queue.put(batch_array)

                except StopIteration:
                    break
        
        ### Put exception in queue so the main thread can get it 
        ### this is important because atleast one of the threads hit the
        ### end of the dataset, but this exception has to be seen by the main 
        ### thread for us to do anything about it 
        except Exception as e:
            self.queue.put(e)

        ### If we are done, just add a None to our Queue so we can ###
        ### identify later that this is over and we should exit ###
        finally:

            ### Track worker completion ###
            with self.lock:

                ### The workers will start to end so we tracj then as they end
                self.worker_finished_count += 1

                ### Once all workers are done 
                if self.worker_finished_count == self.num_workers:
                    self.queue.put(None)

    def __iter__(self):
        """
        The iter is how our iterable is initialied. Every time you take
        and iterable and for loop through it, it triggers __iter__ which 
        sets up and returns the thing we want to iterate over
        """
        ### __iter__ starts the iteration of our object. Go ahead and ###
        ### restart the index and stop_signal flag should be False ###
        self.worker_finished_count = 0
        self.stop_signal.clear()

        ### A freshly created dataloader will have _shuffled_indices as None, 
        ### but if we load a state_dict, we will want to resume
        if hasattr(self, '_resume_from_state') and self._resume_from_state:

            ### RESUMING FROM CHECKPOINT ###
            self.indices = self._shuffled_indices
            self.current_idx = self.batches_consumed * self.batch_size # This is a slight approximate
                                                                       # incase some batches had different number of samples
            self._resume_from_state = False  # Clear the flag

        else:
            ### FRESH START (either new epoch or first time) ###
            self.batches_consumed = 0
            self.current_idx = 0
            
            ### Random shuffle ###
            if self.shuffle:
                self.indices = np.random.permutation(len(self.dataset))
                self._shuffled_indices = self.indices.copy()
            else:
                self.indices = np.arange(len(self.dataset))
                self._shuffled_indices = self.indices.copy()

        ### Clear queue from previous epoch (its full of Nones) ###
        if self.queue is not None:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

        ### Spawn workers to start filling queue asynchronously ###
        if self.num_workers > 0:
            self.workers = []
            for _ in range(self.num_workers):
                t = threading.Thread(target=self._worker_loop, daemon=True)
                t.start()
                self.workers.append(t)
        
        return self

    def __next__(self):
        
        """
        __iter__ sets up an iterable, __next__ is the logic of what returns
        from each iteration through the iterable
        """
        ### When not using queue ###
        if self.num_workers == 0:
            batch_indices = self._get_next_batch_indices()
            
            ### Catch None means we have run out of data as indicated by _get_next_batch_indices
            if batch_indices is None:
                self.reset_epoch()
                raise StopIteration
            
            ### Otherwise we have data so go ahead and collate it
            else:
                batch = [self.dataset[i] for i in batch_indices]
                self.batches_consumed += 1
                return self.collate_fn(batch)

        ### For multi workers we can just grab data from our queue as its 
        ### Already prefetched batches
        try:
            batch = self.queue.get(timeout=self.timeout)
      
            ### We threw exceptions into the queue, this is how our main thread catches them
            if isinstance(batch, Exception):
                raise batch
            
            if batch is None:
                ### This means we are out of data and have completed one iteration of our dataloader ###
                self.reset_epoch()
                raise StopIteration
            
            self.batches_consumed += 1
            return batch
        
        except queue.Empty:

            # If queue is empty after timeout, check if workers are still alive
            alive_workers = [w for w in self.workers if w.is_alive()]
            if not alive_workers:
                self.reset_epoch()
                raise StopIteration
            else:
                raise RuntimeError(
                    f"DataLoader timeout after {self.timeout}s. "
                    f"Workers may be blocked or dataset loading is too slow. "
                    f"Try increasing timeout or reducing num_workers."
                )
    
    def shutdown(self):
        
        ### Sets stop_signal to True telling all workers to stop collecting ###

        if self.num_workers > 0:
            self.stop_signal.set()

            if self.queue is not None:
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
        
        
            for t in self.workers:
                t.join(timeout=5)
                
            self.workers.clear()
    
    def __del__(self):
        """
        What to do when called by Garbage Collector
        """
        self.shutdown()

    def __len__(self):
        ### Ensures always rounds up for num batches ###
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def reset_epoch(self):

        ### Gaurd against multiple updates (in DDP) ###
        ### If we already have batch_consumed at 0 and our _shuffled_indices is None
        ### then we are already at the start, no need to reset
        if self.batches_consumed != 0 and self._shuffled_indices is not None:
            self.epoch += 1
            self.batches_consumed = 0
            self._shuffled_indices = None

    def state_dict(self):
        """
        store where we are currently at in the batch so we can resume from this 
        exact state
        """
        return {
            "epoch": self.epoch, 
            "batches_consumed": self.batches_consumed, 
            'shuffled_indices': self._shuffled_indices.tolist() if self._shuffled_indices is not None else None,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.batches_consumed = state_dict["batches_consumed"]
        if state_dict["shuffled_indices"] is not None:
            self._shuffled_indices = np.array(state_dict["shuffled_indices"])
        else:
            self._shuffled_indices = np.arange(len(self.dataset))
        self._resume_from_state = True 


if __name__ == "__main__":

    from tqdm import tqdm

    train_dataset = list(arange(100))

    ### Create dataloader ###
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0
    )
    
    counter = 0
    max_counter = 3
    train = True

    while train:
        for idx in tqdm(train_loader):
            print(idx)
            counter += 1

            if counter == max_counter:
                train = False            

                ### create state dict ###
                state_dict = train_loader.state_dict()
                
                ### Dont continue
                train = False

                ### print next 4 samples to make sure it matches
                ### Where we pick up from 

                print("-" * 50)
                idx1 = next(train_loader)
                idx2 = next(train_loader)
                idx3 = next(train_loader)
                idx4 = next(train_loader)
                
                print(idx1)
                print(idx2)
                print(idx3)
                print(idx4)

                break
    

    new_train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0, 
    )
    
    new_train_loader.load_state_dict(state_dict)
    print(state_dict)

    print("-" * 50)
    counter = 0
    for idx in tqdm(new_train_loader):
        counter += 1
        print(idx)
        if counter == 10:
            break
