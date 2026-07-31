"""Production-oriented runtime helpers for MLX K-FAC training."""

from __future__ import annotations

import inspect
from typing import Callable, Optional

import mlx.core as mx
from mlx.utils import tree_flatten


def _materialize_finite_arrays(**trees):
    """Materialize array trees and reject non-finite values before commit."""
    finite_checks = []
    for label, tree in trees.items():
        if tree is None:
            continue
        for path, value in tree_flatten(tree):
            if not hasattr(value, "dtype") or not hasattr(value, "shape"):
                continue
            rendered = f"{label}.{path}" if path else label
            finite_checks.append(
                (rendered, mx.all(mx.isfinite(value)))
            )
    mx.eval(trees, [check for _, check in finite_checks])
    nonfinite = [
        path for path, check in finite_checks if not bool(check)
    ]
    if nonfinite:
        raise FloatingPointError(
            "Non-finite values would be committed at "
            + ", ".join(nonfinite)
        )


class CompiledKFACStep:
    """Host dispatcher over separately compiled refresh and cached steps.

    MLX does not currently expose a lazy conditional primitive. A single
    compiled graph with a dynamic refresh predicate must therefore construct
    decomposition candidates on every invocation. This dispatcher traces two
    graphs with static ``refresh`` values and selects between them on the host.

    ``step`` must accept a keyword-only (or regular keyword) argument named
    ``refresh`` and forward it to :meth:`mlx_kfac.KFAC.update`.
    """

    def __init__(
        self,
        optimizer,
        step: Callable,
        *,
        inputs=None,
        outputs=None,
        shapeless: bool = False,
    ):
        if not getattr(optimizer, "_initialized", False):
            raise ValueError(
                "Initialize KFAC state before compiling a dispatched step"
            )
        try:
            signature = inspect.signature(step)
        except (TypeError, ValueError) as error:
            raise TypeError("step must expose a Python signature") from error
        accepts_refresh = "refresh" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_refresh:
            raise TypeError(
                "step must accept a 'refresh' keyword and pass it to KFAC.update"
            )

        self.optimizer = optimizer
        self.step_function = step
        self.refresh_calls = 0
        self.cached_calls = 0
        self.last_refresh = None
        self._materialize = _materialize_finite_arrays

        def refresh_step(*args, **kwargs):
            return step(*args, refresh=True, **kwargs)

        def cached_step(*args, **kwargs):
            return step(*args, refresh=False, **kwargs)

        self.refresh_step = mx.compile(
            refresh_step,
            inputs=inputs,
            outputs=outputs,
            shapeless=shapeless,
        )
        self.cached_step = mx.compile(
            cached_step,
            inputs=inputs,
            outputs=outputs,
            shapeless=shapeless,
        )

    def refresh_due(self) -> bool:
        """Return the host-side inverse refresh decision for the next step."""
        step = int(self.optimizer.step)
        force = bool(self.optimizer.state["force_refresh"])
        return step % self.optimizer.inverse_update_interval == 0 or force

    def __call__(self, *args, **kwargs):
        if "refresh" in kwargs:
            raise TypeError(
                "CompiledKFACStep owns the refresh schedule; do not pass refresh"
            )
        refresh = self.refresh_due()
        function = self.refresh_step if refresh else self.cached_step
        model = getattr(self.optimizer, "_model", None)
        if model is None:
            raise ValueError(
                "Compiled K-FAC execution requires a registered model"
            )

        def operation():
            result = function(*args, **kwargs)
            # Host-side dispatch already reads the completed step before each
            # invocation. Materialize this invocation inside the transaction
            # as well so asynchronous decomposition or collective failures
            # cannot commit partial captured state.
            self._materialize(
                model_state=model.state,
                optimizer_state=self.optimizer.state,
            )
            return result

        result = self.optimizer._run_update_transaction(model, operation)
        self.last_refresh = refresh
        if refresh:
            self.refresh_calls += 1
        else:
            self.cached_calls += 1
        return result


def compile_kfac_step(
    optimizer,
    step: Callable,
    *,
    inputs=None,
    outputs=None,
    shapeless: bool = False,
) -> CompiledKFACStep:
    """Compile static refresh/no-refresh graphs and return their dispatcher."""
    return CompiledKFACStep(
        optimizer,
        step,
        inputs=inputs,
        outputs=outputs,
        shapeless=shapeless,
    )


def format_bytes(size: int) -> str:
    """Format a non-negative byte count for actionable memory errors."""
    value = float(max(0, size))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def validate_state_memory_limit(
    estimated_bytes: int,
    limit_bytes: Optional[int],
) -> None:
    """Raise before allocation when projected optimizer state exceeds a limit."""
    if limit_bytes is None:
        return
    if estimated_bytes > limit_bytes:
        raise MemoryError(
            "Projected K-FAC optimizer state "
            f"({format_bytes(estimated_bytes)}) exceeds max_state_size_bytes "
            f"({format_bytes(limit_bytes)}). Reduce max_factor_size, freeze "
            "layers, disable embeddings, or explicitly raise the limit."
        )
