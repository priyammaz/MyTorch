import os

### FLAG FOR DLPACK ###
### Triton is highly optimzied for Torch Tensors compared to Cupy Arrays. We can leverage DLPACK
### to temporarily make a torch Tensor representation of our data to accelerate our kernel ###
### and then return it back as a cupy array as it exists the method! 
### The best part of DLPACK is it is a ZERO COPY method, so there is very little overhead!
### Additioanlly we add a use_dlpack flag within the function so we can turn it off without 
### messing with environment variables!
DLPACK_DISABLE = False if os.environ.get("DLPACK_DISABLE", "false") == "false" else True
if DLPACK_DISABLE:
    print("DLPACK Conversion Has Been Disabled on Fused Operations!")

### FLAG FOR USING TRITON AUTOTUNE OR DEFAULT ###
AUTOTUNE_MODE = os.getenv("TRITON_AUTOTUNE_MODE", "none").lower()
if AUTOTUNE_MODE=="max":
    print("Triton Autotuning Starting... This can take a little time!!!")