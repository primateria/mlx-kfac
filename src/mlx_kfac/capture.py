"""Autodiff observation capture and in-place MLX module instrumentation."""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn


class MissingObservationError(RuntimeError):
    """Raised when a supported module has no complete capture for this step."""


class _CaptureMixin:
    """Record call inputs and output cotangents without changing computation."""

    def _kfac_install_capture(self):
        if not hasattr(self, "_kfac_slots"):
            object.__setattr__(self, "_kfac_slots", [self._kfac_make_slot()])
            object.__setattr__(self, "_kfac_cursor", 0)
            object.__setattr__(self, "_kfac_active_count", 0)

    @staticmethod
    def _kfac_make_slot():
        record = [None, None, None]

        @mx.custom_function
        def capture(y):
            return y

        @capture.vjp
        def capture_vjp(y, dy, result):
            record[1] = mx.stop_gradient(dy)
            return dy

        record[2] = capture
        return record

    def _kfac_capture(self, inputs, output):
        self._kfac_install_capture()
        index = self._kfac_cursor
        slots = self._kfac_slots
        if index == len(slots):
            slots.append(self._kfac_make_slot())
        record = slots[index]
        record[0] = mx.stop_gradient(inputs)
        record[1] = None
        object.__setattr__(self, "_kfac_cursor", index + 1)
        object.__setattr__(self, "_kfac_active_count", index + 1)
        return record[2](output)

    @property
    def kfac_observations(self):
        records = getattr(self, "_kfac_slots", ())
        records = [
            record
            for record in records[: getattr(self, "_kfac_active_count", 0)]
            if record[0] is not None and record[1] is not None
        ]
        if not records:
            raise MissingObservationError(
                "No complete K-FAC observation is available. Instrument the "
                "model before tracing, then run forward and backward before "
                "calling KFAC.update()."
            )
        return tuple((record[0], record[1]) for record in records)

    @property
    def kfac_observation(self):
        """Backward-compatible access to the latest observation."""
        return self.kfac_observations[-1]

    def clear_kfac_observations(self, *, hard=False):
        # Preserve custom-function slots: transformed/compiled graphs retain
        # references to them. Only the per-forward cursor is reset.
        if hard:
            object.__setattr__(self, "_kfac_slots", [self._kfac_make_slot()])
        object.__setattr__(self, "_kfac_cursor", 0)
        object.__setattr__(self, "_kfac_active_count", 0)


class _LinearCaptureMixin(_CaptureMixin):
    def __call__(self, x):
        return self._kfac_capture(x, super().__call__(x))


class _Conv2dCaptureMixin(_CaptureMixin):
    def __call__(self, x):
        return self._kfac_capture(x, super().__call__(x))


class _Conv1dCaptureMixin(_CaptureMixin):
    def __call__(self, x):
        return self._kfac_capture(x, super().__call__(x))


class _EmbeddingCaptureMixin(_CaptureMixin):
    def __call__(self, x):
        return self._kfac_capture(x, super().__call__(x))

    def as_linear(self, x):
        object.__setattr__(self, "_kfac_as_linear_used", True)
        return super().as_linear(x)

    def clear_kfac_observations(self, *, hard=False):
        super().clear_kfac_observations(hard=hard)
        object.__setattr__(self, "_kfac_as_linear_used", False)


@lru_cache(maxsize=None)
def _instrumented_class(original_class, mixin):
    return type(
        f"KFACInstrumented{original_class.__name__}",
        (mixin, original_class),
        {"__module__": original_class.__module__},
    )


def _instrument_instance(module, mixin):
    if isinstance(module, _CaptureMixin):
        module._kfac_install_capture()
        return
    original_class = type(module)
    module.__class__ = _instrumented_class(original_class, mixin)
    object.__setattr__(module, "_kfac_original_class", original_class)
    module._kfac_install_capture()


def instrument_model(
    model: nn.Module,
    *,
    linear: bool = True,
    conv1d: bool = True,
    conv2d: bool = True,
    embedding: bool = False,
):
    """Instrument supported modules in place and return registered paths.

    Parameters and module paths are preserved. Grouped convolutions are left
    untouched because their curvature requires a separate factor per group.
    """
    registered = {}
    for path, module in list(model.named_modules()):
        if linear and isinstance(module, nn.Linear):
            _instrument_instance(module, _LinearCaptureMixin)
            registered[path] = module
        elif (
            conv1d
            and isinstance(module, nn.Conv1d)
            and getattr(module, "groups", 1) == 1
        ):
            _instrument_instance(module, _Conv1dCaptureMixin)
            registered[path] = module
        elif (
            conv2d
            and isinstance(module, nn.Conv2d)
            and getattr(module, "groups", 1) == 1
        ):
            _instrument_instance(module, _Conv2dCaptureMixin)
            registered[path] = module
        elif embedding and isinstance(module, nn.Embedding):
            _instrument_instance(module, _EmbeddingCaptureMixin)
            registered[path] = module
    return registered


def is_instrumented(module):
    return isinstance(module, _CaptureMixin)


def clear_observations(model: nn.Module, *, hard=False):
    for _, module in model.named_modules():
        if isinstance(module, _CaptureMixin):
            module.clear_kfac_observations(hard=hard)
