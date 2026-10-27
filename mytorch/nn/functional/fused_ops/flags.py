import os

### Default to not PyTorch, this flag will be removed
### all-together later
DLPACK_DISABLE = True

### FLAG FOR USING TRITON AUTOTUNE OR DEFAULT ###
AUTOTUNE_MODE = os.getenv("TRITON_AUTOTUNE_MODE", "none").lower()
if AUTOTUNE_MODE=="max":
    print("Triton Autotuning Starting... This can take a little time!!!")