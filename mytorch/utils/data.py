from ..tensor import arange
from ..ops import stack
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

    def __init__(self, dataset, batch_size=4, shuffle=True, 
                 num_workers=0, prefetch_factor=2, collate_fn=None,
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

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch = prefetch_factor  # number of batches to prefetch
        self.collate_fn = collate_fn if collate_fn is not None else self.default_collate
        self.timeout = timeout
        self.indices = arange(len(dataset))
        
        if self.num_workers > 0:
            self.queue = queue.Queue(maxsize=prefetch_factor * num_workers)
        else:
            self.queue = None

        self.workers = []
        self.lock = threading.Lock() # Can use to makes sure before a thread can access a shared resource it must aquire a lock
        self.stop_signal = threading.Event() # enables inter-thread comm. Initialized as true, can be toggled
        self.current_idx = 0
        self.worker_finished_count = 0

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
        
        ### Thread safe lock to make sure multiple workers dont grab for indices ###
        ### at the same time!!! They have to be given permission (the lock) to ###
        ### even enter this chunk of code. Otherwise they wait for their turn ###
        with self.lock:
            if self.current_idx >= len(self.dataset):
                raise StopIteration
            
            end_idx = min(self.current_idx + self.batch_size, len(self.dataset))
            batch_indices = self.indices[self.current_idx:end_idx]
            self.current_idx = end_idx
            
        return batch_indices

    def _worker_loop(self):

        ### While there is data keep storing in the queue ###
        try:
            while not self.stop_signal.is_set():

                try:
                    batch_indices = self._get_next_batch_indices()
                    batch = [self.dataset[i] for i in batch_indices]
                    batch_array = self.collate_fn(batch)

                    ### Check stop signal before putting ###
                    if not self.stop_signal.is_set():
                        self.queue.put(batch_array)
                except StopIteration:
                    break
        
        ### Put exception in queue so the main thread can get it ###
        except Exception as e:
            self.queue.put(e)

        ### If we are done, just add a None to our Queue so we can ###
        ### identify later that this is over and we should exit ###
        finally:

            ### Track worker completion ###
            with self.lock:
                self.worker_finished_count += 1
                if self.worker_finished_count == self.num_workers:
                    self.queue.put(None)

    def __iter__(self):

        ### __iter__ starts the iteration of our object. Go ahead and ###
        ### restart the index and stop_signal flag should be False ###
        self.current_idx = 0
        self.worker_finished_count = 0
        self.stop_signal.clear()
        
        ### Random shuffle ###
        if self.shuffle:
            self.indices = np.random.permutation(len(self.dataset))
        
        ### Clear queue from previous epoch (its full of Nones) ###
        if self.queue is not None:
            while not self.queue.empty(): # While not empty
                try: # try to 
                    self.queue.get_nowait() # remove an item without waiting (no block)
                
                ### Even though we checked for not self.queue.empty() another thread may
                ### have just removed the last item before this call 
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

        ### When not using queue ###
        if self.num_workers == 0:
            batch_indices = self._get_next_batch_indices()
            batch = [self.dataset[i] for i in batch_indices]
            return self.collate_fn(batch)

        try:

            batch = self.queue.get(timeout=self.timeout)

            if isinstance(batch, Exception):
                raise batch
            
            if batch is None:
                raise StopIteration
            
            return batch
        
        except queue.Empty:
            # If queue is empty after timeout, check if workers are still alive
            alive_workers = [w for w in self.workers if w.is_alive()]
            if not alive_workers:
                raise StopIteration
            else:
                raise RuntimeError(
                    f"DataLoader timeout after {self.timeout}s. "
                    f"Workers may be blocked or dataset loading is too slow. "
                    f"Try increasing timeout or reducing num_workers."
                )

        # ### Otherwise keep grabbing batches until StopIteration ###
        # while True:
        #     batch = self.queue.get(timeout=self.timeout)
        #     if batch is None: # this is the None we inserted earlier
        #         # Re-insert the None again so the other workers can see this None 
        #         # as well. Otherwise we would just remove the None and the other 
        #         # workers could just keep going
        #         self.queue.put(None)
        #         raise StopIteration
        #     return batch
    
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


if __name__ == "__main__":

    from torchvision.datasets import MNIST
    from tqdm import tqdm

    def default_collate(batch):
        # stack features and labels
        x = np.stack([np.array(b[0]).reshape(-1) for b in batch])
        y = np.array([b[1] for b in batch])
        return x, y
    
    ### Load Dataset ###
    train_dataset = MNIST("../../data", train=True, download=True)

    ### Create dataloader ###
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=6, 
        collate_fn=default_collate
    )

    for images, labels in tqdm(train_loader):
        print(labels)
        pass