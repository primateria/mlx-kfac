"""Native Kronecker-factored approximate curvature for MLX."""

from .capture import instrument_model
from .linear import KFACLinear
from .optimizer import KFAC, precondition_linear_gradient

__all__ = [
    "KFAC",
    "KFACLinear",
    "instrument_model",
    "precondition_linear_gradient",
]
