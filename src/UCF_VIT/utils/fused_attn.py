from enum import Enum

class FusedAttn(Enum):
    """Enumeration of the fused attention implementations that can be selected for a model.

    Attributes:
        FLASH: Use flash attention (e.g. via xformers/PyTorch SDPA).
        CK: Use ROCm Composable Kernels fused attention, for AMD GPUs.
        DEFAULT: Use PyTorch/Triton's built-in fused attention.
        NONE: Do not use a fused attention implementation; fall back to a basic
            Python implementation.
    """
    FLASH = "FLASH"
    CK = "CK"
    DEFAULT = "DEFAULT"
    NONE = "NONE"

