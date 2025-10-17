import os
import cupy as cp

##### QUICK ENABLE FOR TENSOR CORE OPS ###
device = cp.cuda.Device()
# string containing the major index and the minor index. 
# For example, compute capability 3.5 is represented by the string ‘35’.
cc_major, cc_minor = device.compute_capability 
if int(cc_major) >= 8:
    os.environ["CUPY_TF32"] = "1"
##########################################

from .tensor import Tensor, no_grad, zeros, ones, empty, full, \
    arange, linspace, eye, tril, randn, rand, randint, zeros_like, \
        ones_like, empty_like, randn_like, rand_like, full_like

from .ops import *
from .dtypes import *
from .nn import *
from .optim import *
from .utils import * 
from .save_load import save, load
from .sampling import *