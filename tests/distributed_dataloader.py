"""
This is a quick script that verifies that we can save our current iter
in a dataset and resume from that position as well
"""
import mytorch
from mytorch.accelerate import Accelerator
from mytorch.utils.data import Dataset, DataLoader

accelerator = Accelerator()

arr = mytorch.arange(30)

class dummydataset(Dataset):
    def __init__(self):
        self.data = arr
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return arr[idx]
    
dataset = dummydataset()
loader = DataLoader(dataset, batch_size=2, shuffle=False)

loader = accelerator.prepare(loader)[0]
iter_loader = iter(loader)

a = next(iter_loader)
print(accelerator.device, a)
b = next(iter_loader)
print(accelerator.device, b)
c = next(iter_loader)
print(accelerator.device, c)

store_state_dict = loader.state_dict()
print(store_state_dict)

accelerator.wait_for_everyone()
print("-----")
new_loader = DataLoader(dataset, batch_size=2, shuffle=False)
new_loader = accelerator.prepare(new_loader)[0]
new_loader.load_state_dict(store_state_dict)
    
new_iter_loader = iter(new_loader)

a = next(new_iter_loader)
print(accelerator.device, a)
b = next(new_iter_loader)
print(accelerator.device, b)
c = next(new_iter_loader)
print(accelerator.device, c)


accelerator.wait_for_everyone()
print(new_loader.state_dict())

