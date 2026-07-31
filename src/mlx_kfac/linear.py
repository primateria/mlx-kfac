"""Backward-compatible pre-instrumented Linear layer."""

import mlx.nn as nn

from .capture import _LinearCaptureMixin


class KFACLinear(_LinearCaptureMixin, nn.Linear):
    """A drop-in :class:`mlx.nn.Linear` that exposes K-FAC observations.

    MLX does not provide module backward hooks.  This subclass therefore puts
    an identity with a custom VJP after the affine operation.  The identity
    records the output cotangent without changing either the forward result or
    the backward result.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__(input_dims, output_dims, bias=bias)
        self._kfac_install_capture()
