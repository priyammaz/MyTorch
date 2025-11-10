import os
from ._compat import warn_triton_missing, FUSED_AVAIL

### If an ENV Flag is passed to always use Fused ops we capture it here! ###
ALWAYS_USE_FUSED = os.getenv("USE_FUSED_OPS", "False").lower() == "true"
if ALWAYS_USE_FUSED and not FUSED_AVAIL:
    warn_triton_missing()