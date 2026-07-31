"""Native Kronecker-factored approximate curvature for MLX."""

from .capture import instrument_model
from .linear import KFACLinear
from .memory import estimate_kfac_state_bytes
from .optimizer import KFAC, precondition_linear_gradient
from .runtime import CompiledKFACStep, compile_kfac_step

__all__ = [
    "CompiledKFACStep",
    "KFAC",
    "KFACLinear",
    "compile_kfac_step",
    "estimate_kfac_state_bytes",
    "instrument_model",
    "precondition_linear_gradient",
]
