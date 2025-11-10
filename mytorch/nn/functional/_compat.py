import os
from contextlib import suppress
import warnings

### Check for Triton Install ###
FUSED_AVAIL = False
_ALWAYS_WARNED = False

with suppress(ImportError):
    import triton
    FUSED_AVAIL = True

def warn_triton_missing():
    """Warn once that Triton is missing and fused kernels are disabled."""
    global _ALWAYS_WARNED
    if not FUSED_AVAIL and not _ALWAYS_WARNED:
        warnings.warn(
            "Triton not installed – fused kernels disabled, defauling to non-fused ops "
            "Install with: pip install .[triton]",
            UserWarning,
        )
        _ALWAYS_WARNED = True