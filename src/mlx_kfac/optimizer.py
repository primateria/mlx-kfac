"""Native K-FAC optimizer for MLX affine modules."""

from __future__ import annotations

import math
import base64
import inspect
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .capture import (
    MissingObservationError,
    clear_observations,
    instrument_model,
    is_instrumented,
)
from .memory import estimate_kfac_state_bytes
from .runtime import (
    _materialize_finite_arrays,
    compile_kfac_step,
    validate_state_memory_limit,
)

_FLOAT32_INFO = mx.finfo(mx.float32)
_FLOAT32_MAX = float(_FLOAT32_INFO.max)
_FLOAT32_MIN_SUBNORMAL = 2.0**-149
# ``smallest_normal`` was added to MLX's ``finfo`` after the supported 0.25
# floor. IEEE-754 float32 has a fixed minimum normal value, so keep the
# constant as a compatibility fallback instead of raising the MLX floor.
_FLOAT32_SMALLEST_NORMAL = float(
    getattr(_FLOAT32_INFO, "smallest_normal", 2.0**-126)
)
_DEFAULT_RELATIVE_EIGENVALUE_FLOOR = 1e-6


def _validate_float32_scalar(name, value):
    """Return a finite scalar whose value is representable in float32."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a real scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real scalar") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    magnitude = abs(value)
    if magnitude > _FLOAT32_MAX or (
        magnitude != 0.0 and magnitude < _FLOAT32_MIN_SUBNORMAL
    ):
        raise ValueError(f"{name} must be representable in float32")
    return value


def _safe_eigenvalues(
    values: mx.array,
    relative_floor: Optional[float] = None,
):
    """Make eigenvalues finite and positive, with an optional relative floor."""
    values = values.astype(mx.float32)
    finite_values = mx.where(mx.isfinite(values), values, 0.0)
    floor = mx.array(_FLOAT32_SMALLEST_NORMAL, mx.float32)
    if relative_floor is not None:
        scale = mx.maximum(mx.max(mx.abs(finite_values)), floor)
        floor = mx.maximum(
            scale * mx.array(relative_floor, mx.float32), floor
        )
    # Leave headroom for reconstruction and matrix-vector products.
    ceiling = mx.array(_FLOAT32_MAX / 4.0, mx.float32)
    return mx.minimum(mx.maximum(finite_values, floor), ceiling).astype(
        mx.float32
    )


def _safe_psd_eigh(
    matrix: mx.array,
    relative_floor: float = _DEFAULT_RELATIVE_EIGENVALUE_FLOOR,
):
    """Compute a finite, scale-normalized eigendecomposition of a PSD matrix."""
    matrix = matrix.astype(mx.float32)
    matrix = mx.where(mx.isfinite(matrix), matrix, 0.0)
    matrix = 0.5 * matrix + 0.5 * matrix.T
    entry_scale = mx.maximum(
        mx.max(mx.abs(matrix)),
        mx.array(_FLOAT32_SMALLEST_NORMAL, mx.float32),
    )
    values, vectors = mx.linalg.eigh(matrix / entry_scale, stream=mx.cpu)
    values = _safe_eigenvalues(values, relative_floor) * entry_scale
    values = mx.minimum(
        mx.where(mx.isfinite(values), values, _FLOAT32_MAX / 4.0),
        mx.array(_FLOAT32_MAX / 4.0, mx.float32),
    )
    values = mx.maximum(
        values, mx.array(_FLOAT32_SMALLEST_NORMAL, mx.float32)
    )
    return values.astype(mx.float32), vectors.astype(mx.float32)


def _safe_cholesky_factor(cholesky: mx.array):
    """Make a cached triangular factor finite with representable pivots."""
    cholesky = cholesky.astype(mx.float32)
    cholesky = mx.where(mx.isfinite(cholesky), cholesky, 0.0)
    diagonal_floor = mx.array(
        math.sqrt(_FLOAT32_SMALLEST_NORMAL), mx.float32
    )
    diagonal = mx.diag(cholesky)
    safe_diagonal = mx.maximum(diagonal, diagonal_floor)
    return cholesky + mx.diag(safe_diagonal - diagonal)


class _Float32AdamW(optim.AdamW):
    """AdamW fallback with float32 moments and dtype-preserving parameters."""

    def init_single(self, parameter: mx.array, state: dict):
        state["m"] = mx.zeros(parameter.shape, dtype=mx.float32)
        state["v"] = mx.zeros(parameter.shape, dtype=mx.float32)

    def apply_single(self, gradient, parameter, state):
        updated = super().apply_single(
            gradient.astype(mx.float32), parameter.astype(mx.float32), state
        )
        return updated.astype(parameter.dtype)


def _cholesky(
    matrix: mx.array,
    relative_floor: float = _DEFAULT_RELATIVE_EIGENVALUE_FLOOR,
) -> mx.array:
    """Build a Cholesky factor after robust PSD projection and flooring."""
    values, vectors = _safe_psd_eigh(matrix, relative_floor)
    stabilized = (vectors * values[None, :]) @ vectors.T
    stabilized = 0.5 * stabilized + 0.5 * stabilized.T
    stabilized = mx.where(mx.isfinite(stabilized), stabilized, 0.0)
    # MLX 0.32 currently implements these decompositions on its CPU stream.
    factor = mx.linalg.cholesky(stabilized, stream=mx.cpu).astype(mx.float32)
    return _safe_cholesky_factor(factor)


def _eigh(
    matrix: mx.array,
    relative_floor: float = _DEFAULT_RELATIVE_EIGENVALUE_FLOOR,
):
    return _safe_psd_eigh(matrix, relative_floor)


def _solve_cholesky(cholesky: mx.array, rhs: mx.array) -> mx.array:
    """Solve ``(L L.T) x = rhs`` from a cached lower Cholesky factor."""
    cholesky = _safe_cholesky_factor(cholesky)
    y = mx.linalg.solve_triangular(cholesky, rhs, stream=mx.cpu)
    return mx.linalg.solve_triangular(
        cholesky.T, y, upper=True, stream=mx.cpu
    )


def precondition_linear_gradient(
    gradient: mx.array,
    g_cholesky: mx.array,
    a_cholesky: mx.array,
) -> mx.array:
    """Apply ``G^-1 gradient A^-1`` for an MLX ``[out, in]`` weight.

    The right solve is ``solve(A, left.T).T``. No explicit matrix inverse is
    formed.
    """
    left = _solve_cholesky(g_cholesky, gradient)
    return _solve_cholesky(a_cholesky, left.T).T


def _solve_eigen(values: mx.array, vectors: mx.array, rhs: mx.array) -> mx.array:
    values = _safe_eigenvalues(values)
    return vectors @ ((vectors.T @ rhs) / values[:, None])


def _multiply_cholesky(cholesky: mx.array, rhs: mx.array) -> mx.array:
    """Multiply ``rhs`` by the matrix represented by a cached Cholesky."""
    cholesky = _safe_cholesky_factor(cholesky)
    return cholesky @ (cholesky.T @ rhs)


def _multiply_eigen(
    values: mx.array, vectors: mx.array, rhs: mx.array
) -> mx.array:
    """Multiply ``rhs`` by the matrix represented by a cached eigendecomposition."""
    values = _safe_eigenvalues(values)
    return vectors @ (values[:, None] * (vectors.T @ rhs))


def _matrix_stats(samples: mx.array, weights: Optional[mx.array] = None):
    samples = samples.astype(mx.float32).reshape(-1, samples.shape[-1])
    if weights is None:
        return samples.T @ samples, mx.array(samples.shape[0], mx.float32)
    weights = weights.astype(mx.float32).reshape(-1)
    return samples.T @ (samples * weights[:, None]), mx.sum(weights)


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def _conv2d_patches(x: mx.array, layer: nn.Conv2d):
    """Extract NHWC convolution patches in MLX weight-flattening order."""
    kh, kw = layer.weight.shape[1:3]
    sh, sw = _pair(layer.stride)
    ph, pw = _pair(layer.padding)
    dh, dw = _pair(layer.dilation)
    x = mx.pad(x, [(0, 0), (ph, ph), (pw, pw), (0, 0)])
    out_h = (x.shape[1] - dh * (kh - 1) - 1) // sh + 1
    out_w = (x.shape[2] - dw * (kw - 1) - 1) // sw + 1
    patches = []
    for row in range(kh):
        row_start = row * dh
        for col in range(kw):
            col_start = col * dw
            patches.append(
                x[
                    :,
                    row_start : row_start + out_h * sh : sh,
                    col_start : col_start + out_w * sw : sw,
                    :,
                ]
            )
    return mx.concatenate(patches, axis=-1)


def _conv1d_patches(x: mx.array, layer: nn.Conv1d):
    """Extract NLC convolution patches in MLX weight-flattening order."""
    kernel = layer.weight.shape[1]
    stride = layer.stride
    padding = layer.padding
    dilation = layer.dilation
    x = mx.pad(x, [(0, 0), (padding, padding), (0, 0)])
    out_length = (x.shape[1] - dilation * (kernel - 1) - 1) // stride + 1
    patches = []
    for position in range(kernel):
        start = position * dilation
        patches.append(
            x[:, start : start + out_length * stride : stride, :]
        )
    return mx.concatenate(patches, axis=-1)


def _layer_kind(layer):
    if isinstance(layer, nn.Linear):
        return "linear"
    if isinstance(layer, nn.Conv1d):
        return "conv1d"
    if isinstance(layer, nn.Conv2d):
        return "conv2d"
    if isinstance(layer, nn.Embedding):
        return "embedding"
    return None


def _state_key(path):
    if not path:
        return "__root__"
    # Dots delimit paths in MLX tree utilities, so never put a raw module path
    # in a state dictionary key.
    encoded = base64.urlsafe_b64encode(path.encode()).decode().rstrip("=")
    return f"path_{encoded}"


def _block_sizes(total, limit):
    if limit is None or limit >= total:
        return [total]
    return [min(limit, total - start) for start in range(0, total, limit)]


def _select_refresh(refresh, new_value, old_value):
    return (
        new_value.astype(mx.float32)
        if refresh is True
        else mx.where(refresh, new_value, old_value).astype(mx.float32)
    )


class KFAC(optim.Optimizer):
    """Kronecker-factored optimizer for supported MLX affine modules.

    Pass ``model=`` at construction time (or call :meth:`register`) before the
    first forward pass to automatically instrument ordinary MLX modules.
    Unsupported and unregistered parameters use AdamW.

    Trace-based :math:`\\pi` damping is available with
    ``damping_strategy="pi"``:

    ``A_damped = A + sqrt(damping) * pi * I``
    ``G_damped = G + sqrt(damping) / pi * I``.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 1e-2,
        *,
        model: Optional[nn.Module] = None,
        damping: float = 1e-3,
        factor_decay: float = 0.95,
        inverse_update_interval: int = 10,
        momentum: float = 0.0,
        kl_clip: Optional[float] = None,
        kl_clip_mode: str = "natural",
        damping_strategy: str = "uniform",
        decomposition: str = "cholesky",
        weight_decay: float = 0.0,
        fallback_betas=(0.9, 0.999),
        fallback_eps: float = 1e-8,
        factor_aggregator: Optional[Callable] = None,
        distributed_group=None,
        aggregate_distributed_gradients: bool = True,
        output_gradient_scale: float = 1.0,
        loss_reduction: str = "sum",
        adaptive_damping: bool = False,
        damping_adaptation_interval: int = 5,
        damping_adaptation_decay: float = 0.9,
        min_damping: float = 1e-8,
        max_damping: float = 1e8,
        decomposition_relative_eigenvalue_floor: float = 1e-6,
        include_embeddings: bool = True,
        max_factor_size: Optional[int] = 1024,
        max_state_size_bytes: Optional[int] = 2 * 1024**3,
        attention_head_blocks: bool = False,
    ):
        if not callable(learning_rate):
            learning_rate = _validate_float32_scalar(
                "learning_rate", learning_rate
            )
        damping = _validate_float32_scalar("damping", damping)
        factor_decay = _validate_float32_scalar(
            "factor_decay", factor_decay
        )
        momentum = _validate_float32_scalar("momentum", momentum)
        if kl_clip is not None:
            kl_clip = _validate_float32_scalar("kl_clip", kl_clip)
        weight_decay = _validate_float32_scalar("weight_decay", weight_decay)
        fallback_betas = tuple(
            _validate_float32_scalar(f"fallback_betas[{index}]", beta)
            for index, beta in enumerate(fallback_betas)
        )
        fallback_eps = _validate_float32_scalar(
            "fallback_eps", fallback_eps
        )
        output_gradient_scale = _validate_float32_scalar(
            "output_gradient_scale", output_gradient_scale
        )
        damping_adaptation_decay = _validate_float32_scalar(
            "damping_adaptation_decay", damping_adaptation_decay
        )
        min_damping = _validate_float32_scalar(
            "min_damping", min_damping
        )
        max_damping = _validate_float32_scalar(
            "max_damping", max_damping
        )
        decomposition_relative_eigenvalue_floor = _validate_float32_scalar(
            "decomposition_relative_eigenvalue_floor",
            decomposition_relative_eigenvalue_floor,
        )
        if damping <= 0:
            raise ValueError("damping must be positive")
        if not 0 <= factor_decay < 1:
            raise ValueError("factor_decay must be in [0, 1)")
        if (
            isinstance(inverse_update_interval, bool)
            or not isinstance(inverse_update_interval, int)
            or inverse_update_interval < 1
        ):
            raise ValueError("inverse_update_interval must be at least 1")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if kl_clip is not None and kl_clip <= 0:
            raise ValueError("kl_clip must be positive")
        if kl_clip_mode not in {"natural", "update"}:
            raise ValueError("kl_clip_mode must be 'natural' or 'update'")
        if damping_strategy not in {"pi", "uniform"}:
            raise ValueError("damping_strategy must be 'pi' or 'uniform'")
        if decomposition not in {"cholesky", "eigh"}:
            raise ValueError("decomposition must be 'cholesky' or 'eigh'")
        if output_gradient_scale <= 0:
            raise ValueError("output_gradient_scale must be positive")
        if fallback_eps <= 0:
            raise ValueError("fallback_eps must be positive")
        if any(not 0 <= beta < 1 for beta in fallback_betas):
            raise ValueError("fallback_betas must be in [0, 1)")
        if len(fallback_betas) != 2:
            raise ValueError("fallback_betas must contain two values")
        if loss_reduction not in {"sum", "mean"}:
            raise ValueError("loss_reduction must be 'sum' or 'mean'")
        if max_factor_size is not None and (
            isinstance(max_factor_size, bool)
            or not isinstance(max_factor_size, int)
            or max_factor_size < 1
        ):
            raise ValueError("max_factor_size must be a positive integer")
        if max_state_size_bytes is not None and (
            isinstance(max_state_size_bytes, bool)
            or not isinstance(max_state_size_bytes, int)
            or max_state_size_bytes < 1
        ):
            raise ValueError(
                "max_state_size_bytes must be a positive integer or None"
            )
        if (
            isinstance(damping_adaptation_interval, bool)
            or not isinstance(damping_adaptation_interval, int)
            or damping_adaptation_interval < 1
        ):
            raise ValueError("damping_adaptation_interval must be positive")
        if not 0 < damping_adaptation_decay < 1:
            raise ValueError("damping_adaptation_decay must be in (0, 1)")
        if not 0 < min_damping <= damping <= max_damping:
            raise ValueError(
                "damping bounds must satisfy "
                "0 < min_damping <= damping <= max_damping"
            )
        if not 0 < decomposition_relative_eigenvalue_floor <= 1:
            raise ValueError(
                "decomposition_relative_eigenvalue_floor must be in (0, 1]"
            )

        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.damping = float(damping)
        self.factor_decay = factor_decay
        self.inverse_update_interval = inverse_update_interval
        self.momentum = momentum
        self.kl_clip = kl_clip
        self.kl_clip_mode = kl_clip_mode
        self.damping_strategy = damping_strategy
        self.decomposition = decomposition
        self.weight_decay = weight_decay
        self.factor_aggregator = factor_aggregator
        try:
            self._factor_aggregator_uses_stats = (
                factor_aggregator is not None
                and len(inspect.signature(factor_aggregator).parameters) >= 4
            )
        except (TypeError, ValueError):
            # Some extension/builtin callables do not expose a Python
            # signature. Keep the backwards-compatible factor callback form.
            self._factor_aggregator_uses_stats = False
        self.distributed_group = distributed_group
        self.aggregate_distributed_gradients = aggregate_distributed_gradients
        self.output_gradient_scale = output_gradient_scale
        self.loss_reduction = loss_reduction
        self.adaptive_damping = adaptive_damping
        self.damping_adaptation_interval = damping_adaptation_interval
        self.damping_adaptation_decay = damping_adaptation_decay
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.decomposition_relative_eigenvalue_floor = (
            decomposition_relative_eigenvalue_floor
        )
        self.include_embeddings = include_embeddings
        self.max_factor_size = max_factor_size
        self.max_state_size_bytes = max_state_size_bytes
        self._estimated_state_size_bytes = 0
        self.attention_head_blocks = attention_head_blocks
        self._partition_overrides = {}
        self._fallback = _Float32AdamW(
            learning_rate,
            betas=list(fallback_betas),
            eps=fallback_eps,
            weight_decay=weight_decay,
        )
        self._state["damping"] = mx.array(damping, mx.float32)
        self._state["damping_adaptation_step"] = mx.array(0, mx.uint64)
        self._state["force_refresh"] = mx.array(False)
        self._state["last_kl_scale"] = mx.array(1.0, mx.float32)
        self._state["layers"] = {}
        self._state["fallback"] = self._fallback.state
        # Keep checkpoint validation independent of ``self.state``: direct
        # assignment deliberately replaces that tree before validation occurs
        # on the next update.
        self._checkpoint_root_keys = frozenset(self._state)
        self._checkpoint_root_leaf_schema = {
            path: (leaf.shape, leaf.dtype)
            for path, leaf in tree_flatten(
                {
                    key: value
                    for key, value in self._state.items()
                    if key not in {"layers", "fallback"}
                }
            )
        }
        self._checkpoint_fallback_root_schema = {
            path: (leaf.shape, leaf.dtype)
            for path, leaf in tree_flatten(self._state["fallback"])
            if path in {"step", "learning_rate"}
        }
        self._model = None
        self._initializing_during_update = False
        self._state_assigned = False
        if model is not None:
            self.register(model)

    def _factor_cholesky(self, matrix):
        return _cholesky(
            matrix, self.decomposition_relative_eigenvalue_floor
        )

    def _factor_eigh(self, matrix):
        return _eigh(matrix, self.decomposition_relative_eigenvalue_floor)

    def register(self, model: nn.Module):
        """Instrument supported modules before forward/backward."""
        instrument_model(
            model,
            linear=True,
            conv1d=True,
            conv2d=True,
            embedding=self.include_embeddings,
        )
        self._partition_overrides = {}
        if self.attention_head_blocks:
            for path, module in model.named_modules():
                if not isinstance(module, nn.MultiHeadAttention):
                    continue
                prefix = f"{path}." if path else ""
                heads = module.num_heads
                for name in ("query_proj", "key_proj", "value_proj"):
                    projection = getattr(module, name)
                    self._partition_overrides[prefix + name] = (
                        [projection.weight.shape[0] // heads] * heads,
                        None,
                    )
                projection = module.out_proj
                self._partition_overrides[prefix + "out_proj"] = (
                    None,
                    [projection.weight.shape[1] // heads] * heads,
                )
        aliases = {}
        for path, module in model.named_modules():
            if is_instrumented(module):
                aliases.setdefault(id(module), []).append(path)
        duplicates = [paths for paths in aliases.values() if len(paths) > 1]
        if duplicates:
            rendered = ", ".join("/".join(paths) for paths in duplicates)
            raise ValueError(
                "Aliased supported modules are not yet representable by one "
                f"MLX parameter-tree owner: {rendered}"
            )
        parameter_owners = {}
        for path, parameter in tree_flatten(model.parameters()):
            parameter_owners.setdefault(id(parameter), []).append(path)
        tied = [paths for paths in parameter_owners.values() if len(paths) > 1]
        if tied:
            rendered = ", ".join("/".join(paths) for paths in tied)
            raise ValueError(
                "Tied or aliased parameters with incompatible curvature roles are not "
                f"supported: {rendered}"
            )
        self._model = model
        return model

    def estimate_state_size_bytes(self, model=None, parameters=None) -> int:
        """Estimate persistent float32 optimizer state for a model."""
        model = model or self._model
        if model is None:
            raise ValueError("A model is required for K-FAC memory estimation")
        return estimate_kfac_state_bytes(self, model, parameters)

    @property
    def estimated_state_size_bytes(self) -> int:
        """Most recent pre-allocation optimizer-state estimate."""
        return self._estimated_state_size_bytes

    def compile_step(
        self,
        step,
        *,
        inputs=None,
        outputs=None,
        shapeless=False,
    ):
        """Compile static refresh/cached graphs and return a host dispatcher."""
        return compile_kfac_step(
            self,
            step,
            inputs=inputs,
            outputs=outputs,
            shapeless=shapeless,
        )

    def init_single(self, parameter: mx.array, state: dict):
        # Initialization is model-aware and implemented below.
        pass

    def apply_single(self, gradient, parameter, state):
        raise RuntimeError("KFAC updates coupled parameters as a block")

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state
        self._initialized = False
        self._state_assigned = True
        if (
            hasattr(self, "_fallback")
            and isinstance(state, dict)
            and "fallback" in state
        ):
            self._fallback.state = state["fallback"]

    def _find_layers(self, model: nn.Module):
        layers = {}
        seen = set()
        for path, module in model.named_modules():
            if id(module) in seen:
                continue
            kind = _layer_kind(module)
            if kind == "embedding" and not self.include_embeddings:
                continue
            if is_instrumented(module) and kind is not None:
                layers[path] = module
                seen.add(id(module))
        return layers

    def _damping_values(self, a, g, curvature_scale=1.0):
        damping = self.state["damping"].astype(mx.float32) / mx.maximum(
            mx.array(curvature_scale, mx.float32),
            mx.array(1e-12, mx.float32),
        )
        if self.damping_strategy == "uniform":
            return damping, damping
        mean_a = mx.trace(a) / a.shape[0]
        mean_g = mx.trace(g) / g.shape[0]
        pi_candidate = mx.sqrt(
            mx.maximum(mean_a, mx.array(1e-12, mx.float32))
            / mx.maximum(mean_g, mx.array(1e-12, mx.float32))
        )
        zero_factor = mx.logical_or(mean_a <= 1e-12, mean_g <= 1e-12)
        pi = mx.where(zero_factor, mx.array(1.0, mx.float32), pi_candidate)
        root = mx.sqrt(damping)
        return root * pi, root / pi

    def _new_layer_state(self, out_dims, in_dims):
        a = mx.eye(in_dims, dtype=mx.float32)
        g = mx.eye(out_dims, dtype=mx.float32)
        damping_a, damping_g = self._damping_values(a, g)
        a_damped = a + damping_a * mx.eye(in_dims, dtype=mx.float32)
        g_damped = g + damping_g * mx.eye(out_dims, dtype=mx.float32)
        state = {
            "A": a,
            "G": g,
            "damping_A": damping_a.astype(mx.float32),
            "damping_G": damping_g.astype(mx.float32),
            "velocity": mx.zeros((out_dims, in_dims), dtype=mx.float32),
            "curvature_scale": mx.array(1.0, dtype=mx.float32),
            "cached_curvature_scale": mx.array(1.0, dtype=mx.float32),
        }
        if self.decomposition == "cholesky":
            state["A_cholesky"] = self._factor_cholesky(a_damped)
            state["G_cholesky"] = self._factor_cholesky(g_damped)
        else:
            state["A_eigenvalues"], state["A_eigenvectors"] = (
                self._factor_eigh(a_damped)
            )
            state["G_eigenvalues"], state["G_eigenvectors"] = (
                self._factor_eigh(g_damped)
            )
        if self.damping_strategy == "pi" and (
            in_dims == 1 or out_dims == 1
        ):
            if in_dims == 1:
                combined = a[0, 0] * g + self.state["damping"] * mx.eye(
                    out_dims, dtype=mx.float32
                )
            else:
                combined = g[0, 0] * a + self.state["damping"] * mx.eye(
                    in_dims, dtype=mx.float32
                )
            if self.decomposition == "cholesky":
                state["combined_cholesky"] = self._factor_cholesky(combined)
            else:
                (
                    state["combined_eigenvalues"],
                    state["combined_eigenvectors"],
                ) = self._factor_eigh(combined)
        return state

    def _partition_sizes(self, path, out_dims, in_dims):
        out_groups, in_groups = self._partition_overrides.get(
            path, (None, None)
        )

        def subdivide(total, groups):
            if groups is None:
                return _block_sizes(total, self.max_factor_size)
            sizes = []
            for group_size in groups:
                sizes.extend(
                    _block_sizes(group_size, self.max_factor_size)
                )
            remainder = total - sum(groups)
            if remainder > 0:
                # A homogeneous bias coordinate is outside attention heads.
                sizes.extend(_block_sizes(remainder, self.max_factor_size))
            return sizes

        return subdivide(out_dims, out_groups), subdivide(in_dims, in_groups)

    def _new_partitioned_state(self, out_dims, in_dims, out_sizes, in_sizes):
        a_blocks = [mx.eye(size, dtype=mx.float32) for size in in_sizes]
        g_blocks = [mx.eye(size, dtype=mx.float32) for size in out_sizes]
        state = {
            "A_blocks": a_blocks,
            "G_blocks": g_blocks,
            "velocity": mx.zeros((out_dims, in_dims), dtype=mx.float32),
            "curvature_scale": mx.array(1.0, mx.float32),
            "cached_curvature_scale": mx.array(1.0, mx.float32),
            "damping_A": mx.array(self.damping, mx.float32),
            "damping_G": mx.array(self.damping, mx.float32),
        }
        damping = (
            mx.sqrt(self.state["damping"])
            if self.damping_strategy == "pi"
            else self.state["damping"]
        )
        state["damping_A"] = damping.astype(mx.float32)
        state["damping_G"] = damping.astype(mx.float32)
        if self.decomposition == "cholesky":
            state["A_cholesky_blocks"] = [
                self._factor_cholesky(
                    block + damping * mx.eye(block.shape[0])
                )
                for block in a_blocks
            ]
            state["G_cholesky_blocks"] = [
                self._factor_cholesky(
                    block + damping * mx.eye(block.shape[0])
                )
                for block in g_blocks
            ]
        else:
            a_decompositions = [
                self._factor_eigh(
                    block + damping * mx.eye(block.shape[0])
                )
                for block in a_blocks
            ]
            g_decompositions = [
                self._factor_eigh(
                    block + damping * mx.eye(block.shape[0])
                )
                for block in g_blocks
            ]
            state["A_eigenvalue_blocks"] = [item[0] for item in a_decompositions]
            state["A_eigenvector_blocks"] = [item[1] for item in a_decompositions]
            state["G_eigenvalue_blocks"] = [item[0] for item in g_decompositions]
            state["G_eigenvector_blocks"] = [item[1] for item in g_decompositions]
        if self.damping_strategy == "pi" and (
            in_dims == 1 or out_dims == 1
        ):
            if in_dims == 1:
                scalar = a_blocks[0][0, 0]
                combined = [
                    scalar * block
                    + self.state["damping"]
                    * mx.eye(block.shape[0], dtype=mx.float32)
                    for block in g_blocks
                ]
            else:
                scalar = g_blocks[0][0, 0]
                combined = [
                    scalar * block
                    + self.state["damping"]
                    * mx.eye(block.shape[0], dtype=mx.float32)
                    for block in a_blocks
                ]
            if self.decomposition == "cholesky":
                state["combined_cholesky_blocks"] = [
                    self._factor_cholesky(block) for block in combined
                ]
            else:
                decompositions = [
                    self._factor_eigh(block) for block in combined
                ]
                state["combined_eigenvalue_blocks"] = [
                    item[0] for item in decompositions
                ]
                state["combined_eigenvector_blocks"] = [
                    item[1] for item in decompositions
                ]
        return state

    def _new_embedding_state(self, vocab_size, dims):
        a_diag = mx.ones((vocab_size,), dtype=mx.float32)
        feature_sizes = _block_sizes(dims, self.max_factor_size)
        g_blocks = [mx.eye(size, dtype=mx.float32) for size in feature_sizes]
        g = mx.eye(dims, dtype=mx.float32)
        damping_a, damping_g = self._embedding_damping_values(a_diag, g)
        state = {
            "A_diag": a_diag,
            "cached_A_diag": a_diag,
            "damping_A": damping_a.astype(mx.float32),
            "damping_G": damping_g.astype(mx.float32),
            "velocity": mx.zeros((vocab_size, dims), dtype=mx.float32),
            "curvature_scale": mx.array(1.0, mx.float32),
            "cached_curvature_scale": mx.array(1.0, mx.float32),
        }
        if len(g_blocks) == 1:
            state["G"] = g
            g_damped = g + damping_g * mx.eye(dims, dtype=mx.float32)
            if self.decomposition == "cholesky":
                state["G_cholesky"] = self._factor_cholesky(g_damped)
            else:
                state["G_eigenvalues"], state["G_eigenvectors"] = (
                    self._factor_eigh(g_damped)
                )
        else:
            state["G_blocks"] = g_blocks
            damped = [
                block + damping_g * mx.eye(block.shape[0], dtype=mx.float32)
                for block in g_blocks
            ]
            if self.decomposition == "cholesky":
                state["G_cholesky_blocks"] = [
                    self._factor_cholesky(block) for block in damped
                ]
            else:
                decompositions = [
                    self._factor_eigh(block) for block in damped
                ]
                state["G_eigenvalue_blocks"] = [
                    item[0] for item in decompositions
                ]
                state["G_eigenvector_blocks"] = [
                    item[1] for item in decompositions
                ]
        if self.damping_strategy == "pi":
            if dims == 1:
                state["combined_diagonal"] = (
                    a_diag * g[0, 0] + self.state["damping"]
                ).astype(mx.float32)
            elif vocab_size == 1:
                combined = [
                    a_diag[0] * block
                    + self.state["damping"]
                    * mx.eye(block.shape[0], dtype=mx.float32)
                    for block in g_blocks
                ]
                if self.decomposition == "cholesky":
                    state["combined_cholesky_blocks"] = [
                        self._factor_cholesky(block) for block in combined
                    ]
                else:
                    decompositions = [
                        self._factor_eigh(block) for block in combined
                    ]
                    state["combined_eigenvalue_blocks"] = [
                        item[0] for item in decompositions
                    ]
                    state["combined_eigenvector_blocks"] = [
                        item[1] for item in decompositions
                    ]
        return state

    def _embedding_damping_values(self, a_diag, g, curvature_scale=1.0):
        damping = self.state["damping"].astype(mx.float32) / mx.maximum(
            mx.array(curvature_scale, mx.float32),
            mx.array(1e-12, mx.float32),
        )
        if self.damping_strategy == "uniform":
            return damping, damping
        mean_a = mx.mean(a_diag)
        mean_g = mx.trace(g) / g.shape[0]
        pi_candidate = mx.sqrt(
            mx.maximum(mean_a, mx.array(1e-12, mx.float32))
            / mx.maximum(mean_g, mx.array(1e-12, mx.float32))
        )
        zero_factor = mx.logical_or(mean_a <= 1e-12, mean_g <= 1e-12)
        pi = mx.where(zero_factor, mx.array(1.0, mx.float32), pi_candidate)
        root = mx.sqrt(damping)
        return root * pi, root / pi

    def init(self, parameters, model=None):
        model = model or self._model
        if model is None:
            self._fallback.init(parameters)
            self._state["fallback"] = self._fallback.state
            self._initialized = True
            self._state_assigned = False
            return
        self._model = model
        estimated_bytes = self.estimate_state_size_bytes(
            model=model, parameters=parameters
        )
        self._estimated_state_size_bytes = estimated_bytes
        validate_state_memory_limit(
            estimated_bytes, self.max_state_size_bytes
        )
        layers = self._find_layers(model)
        flat_params = dict(tree_flatten(parameters))
        supported = set()
        layer_states = {}
        for path, layer in layers.items():
            prefix = f"{path}." if path else ""
            weight_path = prefix + "weight"
            if weight_path not in flat_params:
                continue
            weight = flat_params[weight_path]
            out_dims = weight.shape[0]
            if _layer_kind(layer) == "embedding":
                layer_states[_state_key(path)] = self._new_embedding_state(
                    weight.shape[0], weight.shape[1]
                )
                supported.add(weight_path)
                continue
            in_dims = math.prod(weight.shape[1:])
            has_bias = "bias" in layer and prefix + "bias" in flat_params
            in_dims += int(has_bias)
            out_sizes, in_sizes = self._partition_sizes(
                path, out_dims, in_dims
            )
            partitioned = len(out_sizes) > 1 or len(in_sizes) > 1
            if partitioned:
                layer_states[_state_key(path)] = self._new_partitioned_state(
                    out_dims, in_dims, out_sizes, in_sizes
                )
            else:
                layer_states[_state_key(path)] = self._new_layer_state(
                    out_dims, in_dims
                )
            supported.add(weight_path)
            if has_bias:
                supported.add(prefix + "bias")

        fallback_flat = [(p, v) for p, v in flat_params.items() if p not in supported]
        if fallback_flat:
            self._fallback.init(tree_unflatten(fallback_flat))
        self._state["layers"] = layer_states
        self._state["fallback"] = self._fallback.state
        self._initialized = True
        self._state_assigned = False
        if not self._initializing_during_update:
            clear_observations(model, hard=True)

    def load_state(self, state, model: nn.Module):
        """Restore optimizer state and validate it against ``model``."""
        self.register(model)
        estimated_bytes = self.estimate_state_size_bytes(model=model)
        self._estimated_state_size_bytes = estimated_bytes
        validate_state_memory_limit(
            estimated_bytes, self.max_state_size_bytes
        )
        # Copy Python containers so two live optimizers cannot mutate each
        # other's state dictionaries. MLX arrays are immutable and may safely
        # be shared until replaced by the next functional update.
        state = tree_unflatten(list(tree_flatten(state)))
        if not isinstance(state, dict):
            raise ValueError("Incompatible optimizer root state schema")
        # ``tree_flatten`` intentionally omits empty containers. Recreate the
        # one valid empty dynamic subtree before validating the root schema.
        state.setdefault("layers", {})
        if set(state) != self._checkpoint_root_keys:
            missing = sorted(self._checkpoint_root_keys - set(state))
            extra = sorted(set(state) - self._checkpoint_root_keys)
            raise ValueError(
                "Incompatible optimizer root state schema; "
                f"missing={missing}, extra={extra}"
            )
        root_leaves = dict(
            tree_flatten(
                {
                    key: value
                    for key, value in state.items()
                    if key not in {"layers", "fallback"}
                }
            )
        )
        if set(root_leaves) != set(self._checkpoint_root_leaf_schema):
            raise ValueError("Incompatible optimizer root state schema")
        for path, (expected_shape, expected_dtype) in (
            self._checkpoint_root_leaf_schema.items()
        ):
            leaf = root_leaves[path]
            if (
                not hasattr(leaf, "shape")
                or not hasattr(leaf, "dtype")
                or leaf.shape != expected_shape
                or leaf.dtype != expected_dtype
            ):
                raise ValueError(
                    f"Incompatible optimizer root state leaf '{path}'"
                )
        flat_params = dict(tree_flatten(model.trainable_parameters()))
        expected = {
            path: layer
            for path, layer in self._find_layers(model).items()
            if (f"{path}." if path else "") + "weight" in flat_params
        }
        expected_keys = {_state_key(path) for path in expected}
        actual_keys = set(state.get("layers", {}))
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"Incompatible K-FAC layer state; missing={missing}, extra={extra}"
            )
        for path, layer in expected.items():
            key = _state_key(path)
            if key not in state.get("layers", {}):
                raise ValueError(f"Missing K-FAC state for layer '{path}'")
            prefix = f"{path}." if path else ""
            weight = flat_params.get(prefix + "weight")
            if weight is None:
                continue
            expected_out = weight.shape[0]
            expected_in = math.prod(weight.shape[1:]) + int(
                "bias" in layer and prefix + "bias" in flat_params
            )
            layer_state = state["layers"][key]
            if _layer_kind(layer) == "embedding":
                expected_state = self._new_embedding_state(
                    expected_out, expected_in
                )
            else:
                out_sizes, in_sizes = self._partition_sizes(
                    path, expected_out, expected_in
                )
                should_partition = len(out_sizes) > 1 or len(in_sizes) > 1
                if should_partition:
                    expected_state = self._new_partitioned_state(
                        expected_out,
                        expected_in,
                        out_sizes,
                        in_sizes,
                    )
                else:
                    expected_state = self._new_layer_state(
                        expected_out, expected_in
                    )
            actual_leaves = dict(tree_flatten(layer_state))
            expected_leaves = dict(tree_flatten(expected_state))
            if set(actual_leaves) != set(expected_leaves):
                raise ValueError(
                    f"Incompatible state schema for layer '{path}'"
                )
            for leaf_path, expected_leaf in expected_leaves.items():
                actual_leaf = actual_leaves[leaf_path]
                if (
                    actual_leaf.shape != expected_leaf.shape
                    or actual_leaf.dtype != expected_leaf.dtype
                ):
                    raise ValueError(
                        f"Incompatible state leaf '{leaf_path}' for layer '{path}'"
                    )
        supported_paths = set()
        for path, layer in expected.items():
            prefix = f"{path}." if path else ""
            supported_paths.add(prefix + "weight")
            if "bias" in layer and prefix + "bias" in flat_params:
                supported_paths.add(prefix + "bias")
        fallback_parameters = {
            path: parameter
            for path, parameter in flat_params.items()
            if path not in supported_paths
        }
        actual_fallback = dict(tree_flatten(state.get("fallback", {})))
        expected_fallback_paths = {"step", "learning_rate"}
        for path in fallback_parameters:
            expected_fallback_paths.add(f"{path}.m")
            expected_fallback_paths.add(f"{path}.v")
        if set(actual_fallback) != expected_fallback_paths:
            raise ValueError("Incompatible AdamW fallback state schema")
        for path, (expected_shape, expected_dtype) in (
            self._checkpoint_fallback_root_schema.items()
        ):
            leaf = actual_fallback[path]
            if (
                not hasattr(leaf, "shape")
                or not hasattr(leaf, "dtype")
                or leaf.shape != expected_shape
                or leaf.dtype != expected_dtype
            ):
                raise ValueError(
                    f"Incompatible AdamW fallback state leaf '{path}'"
                )
        for path, parameter in fallback_parameters.items():
            for moment in ("m", "v"):
                leaf = actual_fallback[f"{path}.{moment}"]
                if leaf.shape != parameter.shape or leaf.dtype != mx.float32:
                    raise ValueError(
                        f"Incompatible AdamW fallback state for '{path}'"
                    )
        finite_checks = [
            (path, mx.all(mx.isfinite(leaf)))
            for path, leaf in tree_flatten(state)
        ]
        mx.eval([check for _, check in finite_checks])
        nonfinite = [
            path for path, check in finite_checks if not bool(check)
        ]
        if nonfinite:
            raise ValueError(
                "Checkpoint optimizer state contains non-finite values at "
                + ", ".join(nonfinite)
            )
        restored_damping = float(state["damping"])
        if not self.min_damping <= restored_damping <= self.max_damping:
            raise ValueError(
                "Checkpoint damping must satisfy configured bounds "
                f"[{self.min_damping}, {self.max_damping}]"
            )
        self.state = state
        self._initialized = True
        self._state_assigned = False

    def adapt_damping(self, reduction_ratio):
        """Adjust damping from an actual/predicted reduction ratio."""
        if not self.adaptive_damping:
            return self.state["damping"]
        self.state["damping_adaptation_step"] = (
            self.state["damping_adaptation_step"] + 1
        )
        due = (
            self.state["damping_adaptation_step"]
            % self.damping_adaptation_interval
        ) == 0
        ratio = mx.array(reduction_ratio, mx.float32)
        decay = self.damping_adaptation_decay
        minimum = mx.array(self.min_damping, mx.float32)
        maximum = mx.array(self.max_damping, mx.float32)
        current = mx.where(
            mx.isfinite(self.state["damping"]),
            self.state["damping"],
            mx.array(self.damping, mx.float32),
        )
        current = mx.clip(current, minimum, maximum)
        lower = current / decay
        upper = current * decay
        proposed = mx.where(
            ratio < 0.25,
            lower,
            mx.where(ratio > 0.75, upper, current),
        )
        proposed = mx.clip(
            mx.where(mx.isfinite(proposed), proposed, maximum),
            minimum,
            maximum,
        )
        self.state["damping"] = mx.where(
            due, proposed, current
        ).astype(mx.float32)
        self.state["force_refresh"] = mx.logical_or(
            self.state["force_refresh"], due
        )
        return self.state["damping"]

    def _aggregate_stats(self, numerator, count, kind, path):
        callback_error = None
        if self.factor_aggregator is not None:
            original_numerator = numerator
            original_count = count
            try:
                if self._factor_aggregator_uses_stats:
                    numerator, count = self.factor_aggregator(
                        numerator, count, kind, path
                    )
                else:
                    factor = numerator / mx.maximum(
                        count, mx.array(1.0, mx.float32)
                    )
                    factor = self.factor_aggregator(factor, kind, path)
                    numerator = factor.astype(mx.float32) * count
                if numerator.shape != original_numerator.shape or count.size != 1:
                    raise ValueError(
                        "factor_aggregator must preserve numerator shape "
                        "and return a scalar count"
                    )
            except Exception as error:
                callback_error = error
                numerator = mx.zeros_like(original_numerator)
                count = mx.zeros_like(original_count)

        distributed = (
            self.distributed_group is not None
            and self.distributed_group.size() > 1
        )
        if callback_error is not None and not distributed:
            raise callback_error
        if distributed:
            if self.factor_aggregator is not None:
                error_count = mx.distributed.all_sum(
                    mx.array(callback_error is not None, mx.uint32),
                    group=self.distributed_group,
                )
                # This is an eager coordination barrier. A Python callback
                # cannot fail dynamically inside an already compiled graph.
                if int(error_count) > 0:
                    if callback_error is not None:
                        raise callback_error
                    raise RuntimeError(
                        "factor_aggregator failed on another distributed rank"
                    )
            numerator = mx.distributed.all_sum(
                numerator, group=self.distributed_group
            )
            count = mx.distributed.all_sum(
                count, group=self.distributed_group
            )
        factor = (
            numerator / mx.maximum(count, mx.array(1.0, mx.float32))
        ).astype(mx.float32)
        return factor, count.astype(mx.float32)

    def _empty_observed_factors(self, path, state):
        """Contribute zero local statistics while preserving collective order."""
        zero_count = mx.array(0.0, mx.float32)
        if "A_diag" in state:
            batch_a, a_count = self._aggregate_stats(
                mx.zeros_like(state["A_diag"]),
                zero_count,
                "A_diag",
                path,
            )
            if "G_blocks" in state:
                batch_g = []
                for index, block in enumerate(state["G_blocks"]):
                    factor, _ = self._aggregate_stats(
                        mx.zeros_like(block),
                        zero_count,
                        f"G.{index}",
                        path,
                    )
                    batch_g.append(factor)
            else:
                batch_g, _ = self._aggregate_stats(
                    mx.zeros_like(state["G"]), zero_count, "G", path
                )
        elif "A_blocks" in state:
            batch_a = []
            for index, block in enumerate(state["A_blocks"]):
                factor, a_count = self._aggregate_stats(
                    mx.zeros_like(block),
                    zero_count,
                    f"A.{index}",
                    path,
                )
                batch_a.append(factor)
            batch_g = []
            for index, block in enumerate(state["G_blocks"]):
                factor, _ = self._aggregate_stats(
                    mx.zeros_like(block),
                    zero_count,
                    f"G.{index}",
                    path,
                )
                batch_g.append(factor)
        else:
            batch_a, a_count = self._aggregate_stats(
                mx.zeros_like(state["A"]), zero_count, "A", path
            )
            batch_g, _ = self._aggregate_stats(
                mx.zeros_like(state["G"]), zero_count, "G", path
            )

        batch_count = mx.distributed.all_sum(
            zero_count, group=self.distributed_group
        )
        curvature_scale = a_count / mx.maximum(
            batch_count, mx.array(1.0, mx.float32)
        )
        return batch_a, batch_g, a_count, curvature_scale

    def _observed_factors(self, path, layer, state, mask=None):
        if _layer_kind(layer) == "embedding":
            return self._observed_embedding_factors(path, layer, state, mask)
        activations = []
        gradients = []
        weights = []
        observations = layer.kfac_observations
        first_activation = observations[0][0]
        batch_count = mx.array(
            1 if first_activation.ndim == 1 else first_activation.shape[0],
            mx.float32,
        )
        masks = mask if isinstance(mask, (list, tuple)) else [mask] * len(observations)
        if len(masks) != len(observations):
            raise ValueError(f"Mask count does not match calls for layer '{path}'")
        kind = _layer_kind(layer)
        for (activation, output_gradient), observation_mask in zip(
            observations, masks
        ):
            reduction_scale = (
                activation.shape[0]
                if self.loss_reduction == "mean" and activation.ndim > 1
                else 1
            )
            if kind == "conv1d":
                activation = _conv1d_patches(activation, layer)
            elif kind == "conv2d":
                activation = _conv2d_patches(activation, layer)
            activation = activation.astype(mx.float32)
            output_gradient = (
                output_gradient.astype(mx.float32)
                * self.output_gradient_scale
                * reduction_scale
            )
            factor_input_width = (
                sum(block.shape[0] for block in state["A_blocks"])
                if "A_blocks" in state
                else state["A"].shape[0]
            )
            if factor_input_width == activation.shape[-1] + 1:
                activation = mx.concatenate(
                    [
                        activation,
                        mx.ones(
                            (*activation.shape[:-1], 1), dtype=mx.float32
                        ),
                    ],
                    axis=-1,
                )
            activations.append(activation.reshape(-1, activation.shape[-1]))
            gradients.append(
                output_gradient.reshape(-1, output_gradient.shape[-1])
            )
            if observation_mask is not None:
                weights.append(observation_mask.reshape(-1))
            elif any(item is not None for item in masks):
                weights.append(mx.ones((activations[-1].shape[0],)))
        activation = mx.concatenate(activations, axis=0)
        output_gradient = mx.concatenate(gradients, axis=0)
        if activation.shape[0] != output_gradient.shape[0]:
            raise ValueError(
                f"Activation/output sample counts differ for layer '{path}'"
            )
        sample_weights = mx.concatenate(weights) if weights else None
        if "A_blocks" in state:
            batch_a = []
            start = 0
            for index, block in enumerate(state["A_blocks"]):
                stop = start + block.shape[0]
                numerator, count = _matrix_stats(
                    activation[:, start:stop], sample_weights
                )
                factor, a_count = self._aggregate_stats(
                    numerator, count, f"A.{index}", path
                )
                batch_a.append(factor)
                start = stop
            batch_g = []
            start = 0
            for index, block in enumerate(state["G_blocks"]):
                stop = start + block.shape[0]
                numerator, count = _matrix_stats(
                    output_gradient[:, start:stop], sample_weights
                )
                factor, g_count = self._aggregate_stats(
                    numerator, count, f"G.{index}", path
                )
                batch_g.append(factor)
                start = stop
        else:
            a_numerator, a_count = _matrix_stats(activation, sample_weights)
            g_numerator, g_count = _matrix_stats(
                output_gradient, sample_weights
            )
            batch_a, a_count = self._aggregate_stats(
                a_numerator, a_count, "A", path
            )
            batch_g, g_count = self._aggregate_stats(
                g_numerator, g_count, "G", path
            )
        if self.distributed_group is not None and self.distributed_group.size() > 1:
            batch_count = mx.distributed.all_sum(
                batch_count, group=self.distributed_group
            )
        curvature_scale = a_count / mx.maximum(
            batch_count, mx.array(1.0, mx.float32)
        )
        return batch_a, batch_g, a_count, curvature_scale

    def _observed_embedding_factors(self, path, layer, state, mask=None):
        if getattr(layer, "_kfac_as_linear_used", False):
            raise ValueError(
                f"Embedding '{path}' was used for both lookup and linear "
                "projection; tied curvature roles require a dedicated solver"
            )
        observations = layer.kfac_observations
        masks = mask if isinstance(mask, (list, tuple)) else [mask] * len(observations)
        if len(masks) != len(observations):
            raise ValueError(f"Mask count does not match calls for layer '{path}'")
        ids_parts = []
        gradient_parts = []
        weight_parts = []
        for (token_ids, output_gradient), observation_mask in zip(
            observations, masks
        ):
            ids_parts.append(token_ids.reshape(-1))
            gradient_parts.append(
                (
                    output_gradient.astype(mx.float32)
                    * self.output_gradient_scale
                    * (
                        token_ids.shape[0]
                        if self.loss_reduction == "mean"
                        and token_ids.ndim > 0
                        else 1
                    )
                ).reshape(-1, output_gradient.shape[-1])
            )
            if observation_mask is None:
                weight_parts.append(
                    mx.ones((token_ids.size,), dtype=mx.float32)
                )
            else:
                weight_parts.append(observation_mask.astype(mx.float32).reshape(-1))
        token_ids = mx.concatenate(ids_parts)
        gradients = mx.concatenate(gradient_parts)
        weights = mx.concatenate(weight_parts)
        count = mx.sum(weights)
        token_counts = mx.zeros_like(state["A_diag"]).at[token_ids].add(weights)
        batch_a, a_count = self._aggregate_stats(
            token_counts, count, "A_diag", path
        )
        if "G_blocks" in state:
            batch_g = []
            start = 0
            for index, block in enumerate(state["G_blocks"]):
                stop = start + block.shape[0]
                g_numerator, g_count = _matrix_stats(
                    gradients[:, start:stop], weights
                )
                factor, _ = self._aggregate_stats(
                    g_numerator, g_count, f"G.{index}", path
                )
                batch_g.append(factor)
                start = stop
        else:
            g_numerator, g_count = _matrix_stats(gradients, weights)
            batch_g, _ = self._aggregate_stats(
                g_numerator, g_count, "G", path
            )
        first_ids = observations[0][0]
        batch_count = mx.array(
            1 if first_ids.ndim == 0 else first_ids.shape[0], mx.float32
        )
        if self.distributed_group is not None and self.distributed_group.size() > 1:
            batch_count = mx.distributed.all_sum(
                batch_count, group=self.distributed_group
            )
        curvature_scale = a_count / mx.maximum(
            batch_count, mx.array(1.0, mx.float32)
        )
        return batch_a, batch_g, a_count, curvature_scale

    def _refresh_combined_block_decomposition(
        self, state, matrices, refresh
    ):
        """Refresh an exact scalar-factor block decomposition."""
        if "combined_cholesky_blocks" in state:
            new_blocks = [
                self._factor_cholesky(matrix) for matrix in matrices
            ]
            state["combined_cholesky_blocks"] = (
                new_blocks
                if refresh is True
                else [
                    _select_refresh(refresh, new, old)
                    for new, old in zip(
                        new_blocks, state["combined_cholesky_blocks"]
                    )
                ]
            )
            return
        new_blocks = [self._factor_eigh(matrix) for matrix in matrices]
        new_values = [item[0] for item in new_blocks]
        new_vectors = [item[1] for item in new_blocks]
        if refresh is True:
            state["combined_eigenvalue_blocks"] = new_values
            state["combined_eigenvector_blocks"] = new_vectors
        else:
            state["combined_eigenvalue_blocks"] = [
                _select_refresh(refresh, new, old)
                for new, old in zip(
                    new_values, state["combined_eigenvalue_blocks"]
                )
            ]
            state["combined_eigenvector_blocks"] = [
                _select_refresh(refresh, new, old)
                for new, old in zip(
                    new_vectors, state["combined_eigenvector_blocks"]
                )
            ]

    def _refresh_embedding_combined(self, state, refresh):
        scale = state["curvature_scale"]
        damping = self.state["damping"]
        if "combined_diagonal" in state:
            candidate = (
                scale * state["A_diag"] * state["G"][0, 0] + damping
            )
            state["combined_diagonal"] = _select_refresh(
                refresh, candidate, state["combined_diagonal"]
            )
            return
        if not (
            "combined_cholesky_blocks" in state
            or "combined_eigenvalue_blocks" in state
        ):
            return
        g_blocks = (
            state["G_blocks"] if "G_blocks" in state else [state["G"]]
        )
        matrices = [
            scale * state["A_diag"][0] * block
            + damping * mx.eye(block.shape[0], dtype=mx.float32)
            for block in g_blocks
        ]
        self._refresh_combined_block_decomposition(
            state, matrices, refresh
        )

    def _refresh_partitioned_combined(self, state, refresh):
        if not (
            "combined_cholesky_blocks" in state
            or "combined_eigenvalue_blocks" in state
        ):
            return
        scale = state["curvature_scale"]
        damping = self.state["damping"]
        if sum(block.shape[0] for block in state["A_blocks"]) == 1:
            scalar = state["A_blocks"][0][0, 0]
            other_blocks = state["G_blocks"]
        else:
            scalar = state["G_blocks"][0][0, 0]
            other_blocks = state["A_blocks"]
        matrices = [
            scale * scalar * block
            + damping * mx.eye(block.shape[0], dtype=mx.float32)
            for block in other_blocks
        ]
        self._refresh_combined_block_decomposition(
            state, matrices, refresh
        )

    def _refresh_decomposition(self, state, refresh):
        if isinstance(refresh, bool) and not refresh:
            return
        state["cached_curvature_scale"] = _select_refresh(
            refresh,
            state["curvature_scale"],
            state["cached_curvature_scale"],
        )
        if "A_diag" in state:
            state["cached_A_diag"] = _select_refresh(
                refresh, state["A_diag"], state["cached_A_diag"]
            )
            if "G_blocks" in state:
                g_dims = sum(block.shape[0] for block in state["G_blocks"])
                mean_g = sum(
                    (mx.trace(block) for block in state["G_blocks"]),
                    mx.array(0.0, mx.float32),
                ) / g_dims
                synthetic_g = mx.eye(1, dtype=mx.float32) * mean_g
            else:
                synthetic_g = state["G"]
            damping_a, damping_g = self._embedding_damping_values(
                state["A_diag"], synthetic_g, state["curvature_scale"]
            )
            state["damping_A"] = _select_refresh(
                refresh, damping_a, state["damping_A"]
            )
            state["damping_G"] = _select_refresh(
                refresh, damping_g, state["damping_G"]
            )
            if "G_blocks" in state:
                damped = [
                    block
                    + damping_g
                    * mx.eye(block.shape[0], dtype=mx.float32)
                    for block in state["G_blocks"]
                ]
                if self.decomposition == "cholesky":
                    new_g = [
                        self._factor_cholesky(block) for block in damped
                    ]
                    state["G_cholesky_blocks"] = (
                        new_g
                        if refresh is True
                        else [
                            mx.where(refresh, new, old)
                            for new, old in zip(
                                new_g, state["G_cholesky_blocks"]
                            )
                        ]
                    )
                else:
                    decompositions = [
                        self._factor_eigh(block) for block in damped
                    ]
                    new_gv = [item[0] for item in decompositions]
                    new_gq = [item[1] for item in decompositions]
                    if refresh is True:
                        state["G_eigenvalue_blocks"] = new_gv
                        state["G_eigenvector_blocks"] = new_gq
                    else:
                        state["G_eigenvalue_blocks"] = [
                            mx.where(refresh, new, old)
                            for new, old in zip(
                                new_gv, state["G_eigenvalue_blocks"]
                            )
                        ]
                        state["G_eigenvector_blocks"] = [
                            mx.where(refresh, new, old)
                            for new, old in zip(
                                new_gq, state["G_eigenvector_blocks"]
                            )
                        ]
                self._refresh_embedding_combined(state, refresh)
                return
            g_damped = state["G"] + damping_g * mx.eye(
                state["G"].shape[0], dtype=mx.float32
            )
            if self.decomposition == "cholesky":
                new_g = self._factor_cholesky(g_damped)
                state["G_cholesky"] = (
                    new_g
                    if refresh is True
                    else mx.where(refresh, new_g, state["G_cholesky"])
                )
            else:
                new_gv, new_gq = self._factor_eigh(g_damped)
                if refresh is True:
                    state["G_eigenvalues"] = new_gv
                    state["G_eigenvectors"] = new_gq
                else:
                    state["G_eigenvalues"] = mx.where(
                        refresh, new_gv, state["G_eigenvalues"]
                    )
                    state["G_eigenvectors"] = mx.where(
                        refresh, new_gq, state["G_eigenvectors"]
                    )
            self._refresh_embedding_combined(state, refresh)
            return
        if "A_blocks" in state:
            a_dims = sum(block.shape[0] for block in state["A_blocks"])
            g_dims = sum(block.shape[0] for block in state["G_blocks"])
            mean_a = sum(
                (mx.trace(block) for block in state["A_blocks"]),
                mx.array(0.0, mx.float32),
            ) / a_dims
            mean_g = sum(
                (mx.trace(block) for block in state["G_blocks"]),
                mx.array(0.0, mx.float32),
            ) / g_dims
            damping = self.state["damping"] / mx.maximum(
                state["curvature_scale"], mx.array(1e-12, mx.float32)
            )
            if self.damping_strategy == "pi":
                pi_candidate = mx.sqrt(
                    mx.maximum(mean_a, mx.array(1e-12, mx.float32))
                    / mx.maximum(mean_g, mx.array(1e-12, mx.float32))
                )
                zero_factor = mx.logical_or(
                    mean_a <= 1e-12, mean_g <= 1e-12
                )
                pi = mx.where(
                    zero_factor, mx.array(1.0, mx.float32), pi_candidate
                )
                damping_a = mx.sqrt(damping) * pi
                damping_g = mx.sqrt(damping) / pi
            else:
                damping_a = damping
                damping_g = damping
            state["damping_A"] = _select_refresh(
                refresh, damping_a, state["damping_A"]
            )
            state["damping_G"] = _select_refresh(
                refresh, damping_g, state["damping_G"]
            )
            a_damped = [
                block + damping_a * mx.eye(block.shape[0], dtype=mx.float32)
                for block in state["A_blocks"]
            ]
            g_damped = [
                block + damping_g * mx.eye(block.shape[0], dtype=mx.float32)
                for block in state["G_blocks"]
            ]
            if self.decomposition == "cholesky":
                new_a = [
                    self._factor_cholesky(block) for block in a_damped
                ]
                new_g = [
                    self._factor_cholesky(block) for block in g_damped
                ]
                if refresh is True:
                    state["A_cholesky_blocks"] = new_a
                    state["G_cholesky_blocks"] = new_g
                else:
                    state["A_cholesky_blocks"] = [
                        mx.where(refresh, new, old)
                        for new, old in zip(new_a, state["A_cholesky_blocks"])
                    ]
                    state["G_cholesky_blocks"] = [
                        mx.where(refresh, new, old)
                        for new, old in zip(new_g, state["G_cholesky_blocks"])
                    ]
            else:
                new_a = [self._factor_eigh(block) for block in a_damped]
                new_g = [self._factor_eigh(block) for block in g_damped]
                new_av, new_aq = zip(*new_a)
                new_gv, new_gq = zip(*new_g)
                if refresh is True:
                    state["A_eigenvalue_blocks"] = list(new_av)
                    state["A_eigenvector_blocks"] = list(new_aq)
                    state["G_eigenvalue_blocks"] = list(new_gv)
                    state["G_eigenvector_blocks"] = list(new_gq)
                else:
                    for key, new_values in (
                        ("A_eigenvalue_blocks", new_av),
                        ("A_eigenvector_blocks", new_aq),
                        ("G_eigenvalue_blocks", new_gv),
                        ("G_eigenvector_blocks", new_gq),
                    ):
                        state[key] = [
                            mx.where(refresh, new, old)
                            for new, old in zip(new_values, state[key])
                        ]
            self._refresh_partitioned_combined(state, refresh)
            return
        damping_a, damping_g = self._damping_values(
            state["A"], state["G"], state["curvature_scale"]
        )
        state["damping_A"] = _select_refresh(
            refresh, damping_a, state["damping_A"]
        )
        state["damping_G"] = _select_refresh(
            refresh, damping_g, state["damping_G"]
        )
        a_damped = state["A"] + damping_a * mx.eye(
            state["A"].shape[0], dtype=mx.float32
        )
        g_damped = state["G"] + damping_g * mx.eye(
            state["G"].shape[0], dtype=mx.float32
        )
        if self.decomposition == "cholesky":
            new_a = self._factor_cholesky(a_damped)
            new_g = self._factor_cholesky(g_damped)
            if refresh is True:
                state["A_cholesky"] = new_a
                state["G_cholesky"] = new_g
            else:
                state["A_cholesky"] = mx.where(
                    refresh, new_a, state["A_cholesky"]
                ).astype(mx.float32)
                state["G_cholesky"] = mx.where(
                    refresh, new_g, state["G_cholesky"]
                ).astype(mx.float32)
        else:
            new_av, new_aq = self._factor_eigh(a_damped)
            new_gv, new_gq = self._factor_eigh(g_damped)
            if refresh is True:
                state["A_eigenvalues"], state["A_eigenvectors"] = new_av, new_aq
                state["G_eigenvalues"], state["G_eigenvectors"] = new_gv, new_gq
            else:
                state["A_eigenvalues"] = mx.where(
                    refresh, new_av, state["A_eigenvalues"]
                )
                state["A_eigenvectors"] = mx.where(
                    refresh, new_aq, state["A_eigenvectors"]
                )
                state["G_eigenvalues"] = mx.where(
                    refresh, new_gv, state["G_eigenvalues"]
                )
                state["G_eigenvectors"] = mx.where(
                    refresh, new_gq, state["G_eigenvectors"]
                )
        if "combined_cholesky" in state or "combined_eigenvalues" in state:
            if state["A"].shape[0] == 1:
                combined = (
                    state["curvature_scale"] * state["A"][0, 0] * state["G"]
                    + self.state["damping"]
                    * mx.eye(state["G"].shape[0], dtype=mx.float32)
                )
            else:
                combined = (
                    state["curvature_scale"] * state["G"][0, 0] * state["A"]
                    + self.state["damping"]
                    * mx.eye(state["A"].shape[0], dtype=mx.float32)
                )
            if self.decomposition == "cholesky":
                new_combined = self._factor_cholesky(combined)
                state["combined_cholesky"] = _select_refresh(
                    refresh, new_combined, state["combined_cholesky"]
                )
            else:
                new_values, new_vectors = self._factor_eigh(combined)
                state["combined_eigenvalues"] = _select_refresh(
                    refresh, new_values, state["combined_eigenvalues"]
                )
                state["combined_eigenvectors"] = _select_refresh(
                    refresh, new_vectors, state["combined_eigenvectors"]
                )

    def _precondition(self, state, gradient):
        if "A_diag" in state:
            if "combined_diagonal" in state:
                return gradient / state["combined_diagonal"][:, None]
            if (
                "combined_cholesky_blocks" in state
                or "combined_eigenvalue_blocks" in state
            ):
                columns = []
                start = 0
                decompositions = (
                    state["combined_cholesky_blocks"]
                    if self.decomposition == "cholesky"
                    else state["combined_eigenvalue_blocks"]
                )
                for index, decomposition in enumerate(decompositions):
                    stop = start + decomposition.shape[0]
                    block_gradient = gradient[:, start:stop]
                    if self.decomposition == "cholesky":
                        block_natural = _solve_cholesky(
                            decomposition, block_gradient.T
                        ).T
                    else:
                        block_natural = _solve_eigen(
                            decomposition,
                            state["combined_eigenvector_blocks"][index],
                            block_gradient.T,
                        ).T
                    columns.append(block_natural)
                    start = stop
                return mx.concatenate(columns, axis=1)
            if "G_blocks" in state:
                columns = []
                start = 0
                for index, block in enumerate(state["G_blocks"]):
                    stop = start + block.shape[0]
                    block_gradient = gradient[:, start:stop]
                    if self.decomposition == "cholesky":
                        block_right = _solve_cholesky(
                            state["G_cholesky_blocks"][index],
                            block_gradient.T,
                        ).T
                    else:
                        block_right = _solve_eigen(
                            state["G_eigenvalue_blocks"][index],
                            state["G_eigenvector_blocks"][index],
                            block_gradient.T,
                        ).T
                    columns.append(block_right)
                    start = stop
                right = mx.concatenate(columns, axis=1)
            elif self.decomposition == "cholesky":
                right = _solve_cholesky(state["G_cholesky"], gradient.T).T
            else:
                right = _solve_eigen(
                    state["G_eigenvalues"],
                    state["G_eigenvectors"],
                    gradient.T,
                ).T
            natural = right / (
                state["cached_A_diag"][:, None] + state["damping_A"]
            )
        elif "A_blocks" in state:
            if (
                "combined_cholesky_blocks" in state
                or "combined_eigenvalue_blocks" in state
            ):
                decompositions = (
                    state["combined_cholesky_blocks"]
                    if self.decomposition == "cholesky"
                    else state["combined_eigenvalue_blocks"]
                )
                scalar_input = (
                    sum(block.shape[0] for block in state["A_blocks"]) == 1
                )
                pieces = []
                start = 0
                for index, decomposition in enumerate(decompositions):
                    stop = start + decomposition.shape[0]
                    if scalar_input:
                        block_gradient = gradient[start:stop, :]
                    else:
                        block_gradient = gradient[:, start:stop]
                    if self.decomposition == "cholesky":
                        block_natural = _solve_cholesky(
                            decomposition,
                            block_gradient
                            if scalar_input
                            else block_gradient.T,
                        )
                    else:
                        block_natural = _solve_eigen(
                            decomposition,
                            state["combined_eigenvector_blocks"][index],
                            block_gradient
                            if scalar_input
                            else block_gradient.T,
                        )
                    pieces.append(
                        block_natural
                        if scalar_input
                        else block_natural.T
                    )
                    start = stop
                return mx.concatenate(
                    pieces, axis=0 if scalar_input else 1
                )
            rows = []
            row_start = 0
            for g_index, g_block in enumerate(state["G_blocks"]):
                row_stop = row_start + g_block.shape[0]
                columns = []
                col_start = 0
                for a_index, a_block in enumerate(state["A_blocks"]):
                    col_stop = col_start + a_block.shape[0]
                    block_gradient = gradient[
                        row_start:row_stop, col_start:col_stop
                    ]
                    if self.decomposition == "cholesky":
                        block_natural = precondition_linear_gradient(
                            block_gradient,
                            state["G_cholesky_blocks"][g_index],
                            state["A_cholesky_blocks"][a_index],
                        )
                    else:
                        left = _solve_eigen(
                            state["G_eigenvalue_blocks"][g_index],
                            state["G_eigenvector_blocks"][g_index],
                            block_gradient,
                        )
                        block_natural = _solve_eigen(
                            state["A_eigenvalue_blocks"][a_index],
                            state["A_eigenvector_blocks"][a_index],
                            left.T,
                        ).T
                    columns.append(block_natural)
                    col_start = col_stop
                rows.append(mx.concatenate(columns, axis=1))
                row_start = row_stop
            natural = mx.concatenate(rows, axis=0)
        elif "combined_cholesky" in state or "combined_eigenvalues" in state:
            if state["A"].shape[0] == 1:
                if self.decomposition == "cholesky":
                    natural = _solve_cholesky(
                        state["combined_cholesky"], gradient
                    )
                else:
                    natural = _solve_eigen(
                        state["combined_eigenvalues"],
                        state["combined_eigenvectors"],
                        gradient,
                    )
            else:
                if self.decomposition == "cholesky":
                    natural = _solve_cholesky(
                        state["combined_cholesky"], gradient.T
                    ).T
                else:
                    natural = _solve_eigen(
                        state["combined_eigenvalues"],
                        state["combined_eigenvectors"],
                        gradient.T,
                    ).T
            return natural
        elif self.decomposition == "cholesky":
            natural = precondition_linear_gradient(
                gradient, state["G_cholesky"], state["A_cholesky"]
            )
        else:
            left = _solve_eigen(
                state["G_eigenvalues"], state["G_eigenvectors"], gradient
            )
            natural = _solve_eigen(
                state["A_eigenvalues"], state["A_eigenvectors"], left.T
            ).T
        return natural / mx.maximum(
            state["cached_curvature_scale"], mx.array(1e-12, mx.float32)
        )

    def _curvature_product(self, state, direction):
        """Apply the cached damped block curvature to an update direction."""
        scale = state["cached_curvature_scale"]
        if "A_diag" in state:
            if "combined_diagonal" in state:
                return state["combined_diagonal"][:, None] * direction
            if (
                "combined_cholesky_blocks" in state
                or "combined_eigenvalue_blocks" in state
            ):
                columns = []
                start = 0
                decompositions = (
                    state["combined_cholesky_blocks"]
                    if self.decomposition == "cholesky"
                    else state["combined_eigenvalue_blocks"]
                )
                for index, decomposition in enumerate(decompositions):
                    stop = start + decomposition.shape[0]
                    block_direction = direction[:, start:stop]
                    if self.decomposition == "cholesky":
                        product = _multiply_cholesky(
                            decomposition, block_direction.T
                        ).T
                    else:
                        product = _multiply_eigen(
                            decomposition,
                            state["combined_eigenvector_blocks"][index],
                            block_direction.T,
                        ).T
                    columns.append(product)
                    start = stop
                return mx.concatenate(columns, axis=1)
            if "G_blocks" in state:
                columns = []
                start = 0
                for index, block in enumerate(state["G_blocks"]):
                    stop = start + block.shape[0]
                    block_direction = direction[:, start:stop]
                    if self.decomposition == "cholesky":
                        product = _multiply_cholesky(
                            state["G_cholesky_blocks"][index],
                            block_direction.T,
                        ).T
                    else:
                        product = _multiply_eigen(
                            state["G_eigenvalue_blocks"][index],
                            state["G_eigenvector_blocks"][index],
                            block_direction.T,
                        ).T
                    columns.append(product)
                    start = stop
                right = mx.concatenate(columns, axis=1)
            elif self.decomposition == "cholesky":
                right = _multiply_cholesky(
                    state["G_cholesky"], direction.T
                ).T
            else:
                right = _multiply_eigen(
                    state["G_eigenvalues"],
                    state["G_eigenvectors"],
                    direction.T,
                ).T
            return (
                scale
                * (state["cached_A_diag"] + state["damping_A"])[:, None]
                * right
            )
        if "A_blocks" in state:
            if (
                "combined_cholesky_blocks" in state
                or "combined_eigenvalue_blocks" in state
            ):
                decompositions = (
                    state["combined_cholesky_blocks"]
                    if self.decomposition == "cholesky"
                    else state["combined_eigenvalue_blocks"]
                )
                scalar_input = (
                    sum(block.shape[0] for block in state["A_blocks"]) == 1
                )
                pieces = []
                start = 0
                for index, decomposition in enumerate(decompositions):
                    stop = start + decomposition.shape[0]
                    block_direction = (
                        direction[start:stop, :]
                        if scalar_input
                        else direction[:, start:stop]
                    )
                    rhs = (
                        block_direction
                        if scalar_input
                        else block_direction.T
                    )
                    if self.decomposition == "cholesky":
                        product = _multiply_cholesky(decomposition, rhs)
                    else:
                        product = _multiply_eigen(
                            decomposition,
                            state["combined_eigenvector_blocks"][index],
                            rhs,
                        )
                    pieces.append(product if scalar_input else product.T)
                    start = stop
                return mx.concatenate(
                    pieces, axis=0 if scalar_input else 1
                )
            rows = []
            row_start = 0
            for g_index, g_block in enumerate(state["G_blocks"]):
                row_stop = row_start + g_block.shape[0]
                columns = []
                col_start = 0
                for a_index, a_block in enumerate(state["A_blocks"]):
                    col_stop = col_start + a_block.shape[0]
                    block = direction[
                        row_start:row_stop, col_start:col_stop
                    ]
                    if self.decomposition == "cholesky":
                        left = _multiply_cholesky(
                            state["G_cholesky_blocks"][g_index], block
                        )
                        product = _multiply_cholesky(
                            state["A_cholesky_blocks"][a_index], left.T
                        ).T
                    else:
                        left = _multiply_eigen(
                            state["G_eigenvalue_blocks"][g_index],
                            state["G_eigenvector_blocks"][g_index],
                            block,
                        )
                        product = _multiply_eigen(
                            state["A_eigenvalue_blocks"][a_index],
                            state["A_eigenvector_blocks"][a_index],
                            left.T,
                        ).T
                    columns.append(product)
                    col_start = col_stop
                rows.append(mx.concatenate(columns, axis=1))
                row_start = row_stop
            return scale * mx.concatenate(rows, axis=0)
        if "combined_cholesky" in state or "combined_eigenvalues" in state:
            if state["A"].shape[0] == 1:
                if self.decomposition == "cholesky":
                    return _multiply_cholesky(
                        state["combined_cholesky"], direction
                    )
                return _multiply_eigen(
                    state["combined_eigenvalues"],
                    state["combined_eigenvectors"],
                    direction,
                )
            if self.decomposition == "cholesky":
                return _multiply_cholesky(
                    state["combined_cholesky"], direction.T
                ).T
            return _multiply_eigen(
                state["combined_eigenvalues"],
                state["combined_eigenvectors"],
                direction.T,
            ).T
        if self.decomposition == "cholesky":
            left = _multiply_cholesky(state["G_cholesky"], direction)
            product = _multiply_cholesky(
                state["A_cholesky"], left.T
            ).T
        else:
            left = _multiply_eigen(
                state["G_eigenvalues"],
                state["G_eigenvectors"],
                direction,
            )
            product = _multiply_eigen(
                state["A_eigenvalues"],
                state["A_eigenvectors"],
                left.T,
            ).T
        return scale * product

    def update(
        self,
        model: nn.Module,
        gradients: dict,
        *,
        masks=None,
        refresh=None,
        gradient_weight=None,
    ):
        def operation():
            updates = self._apply_gradients_impl(
                gradients,
                model,
                model=model,
                masks=masks,
                refresh=refresh,
                gradient_weight=gradient_weight,
            )
            model.update(updates)

        return self._run_update_transaction(model, operation)

    def apply_gradients(
        self,
        gradients: dict,
        parameters: dict,
        *,
        model=None,
        masks=None,
        refresh=None,
        gradient_weight=None,
    ):
        resolved_model = model or (
            parameters if isinstance(parameters, nn.Module) else None
        )
        resolved_model = resolved_model or self._model
        if resolved_model is None:
            raise ValueError("KFAC requires a model")

        return self._run_update_transaction(
            resolved_model,
            lambda: self._apply_gradients_impl(
                gradients,
                parameters,
                model=resolved_model,
                masks=masks,
                refresh=refresh,
                gradient_weight=gradient_weight,
            ),
        )

    @staticmethod
    def _clone_state_containers(value):
        """Copy tree containers while sharing immutable MLX array leaves."""
        if isinstance(value, dict):
            return {
                key: KFAC._clone_state_containers(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [KFAC._clone_state_containers(item) for item in value]
        if isinstance(value, tuple):
            return tuple(KFAC._clone_state_containers(item) for item in value)
        return value

    def _run_update_transaction(self, model, operation):
        previous_state_reference = self._state
        previous_state = self._clone_state_containers(self._state)
        previous_fallback = self._clone_state_containers(
            self._fallback.state
        )
        previous_model_state = self._clone_state_containers(model.state)
        previous_initialized = self._initialized
        previous_state_assigned = self._state_assigned
        previous_fallback_initialized = getattr(
            self._fallback, "_initialized", False
        )
        previous_model = self._model
        previous_partition_overrides = self._clone_state_containers(
            self._partition_overrides
        )
        try:
            return operation()
        except BaseException:
            # Bypass a user override that may itself have caused a partial
            # ``model.update`` failure.
            nn.Module.update(model, previous_model_state)
            if isinstance(previous_state, dict) and "fallback" in previous_state:
                previous_state["fallback"] = previous_fallback
            if isinstance(previous_state_reference, dict) and isinstance(
                previous_state, dict
            ):
                previous_state_reference.clear()
                previous_state_reference.update(previous_state)
                self._state = previous_state_reference
            else:
                self._state = previous_state_reference
            self._fallback.state = previous_fallback
            self._initialized = previous_initialized
            self._state_assigned = previous_state_assigned
            self._fallback._initialized = previous_fallback_initialized
            self._model = previous_model
            self._partition_overrides = previous_partition_overrides
            raise
        finally:
            clear_observations(model)

    @staticmethod
    def _gradient_schema_fingerprint(flat_gradients):
        value = 0
        for path, gradient in sorted(flat_gradients.items()):
            description = f"{path}:{gradient.shape}:{gradient.dtype}"
            for byte in description.encode():
                value = (value * 257 + byte) % 1_000_003
        return value

    def _aggregate_distributed_gradients(
        self, flat_gradients, gradient_weight
    ):
        group = self.distributed_group
        size = group.size()

        conversion_error = False
        if gradient_weight is None:
            local_mode = 0
            local_weight = mx.array(1.0, mx.float32)
        else:
            local_mode = 1
            try:
                candidate = mx.array(gradient_weight, mx.float32)
                if candidate.size != 1:
                    raise ValueError("gradient_weight must be a scalar")
                local_weight = candidate.reshape(())
            except (TypeError, ValueError):
                conversion_error = True
                local_weight = mx.array(0.0, mx.float32)

        local_invalid = mx.logical_or(
            mx.array(conversion_error),
            mx.logical_or(
                mx.logical_not(mx.isfinite(local_weight)),
                local_weight < 0,
            ),
        )
        mode_count = mx.distributed.all_sum(
            mx.array(local_mode, mx.uint32), group=group
        )
        invalid_count = mx.distributed.all_sum(
            local_invalid.astype(mx.uint32), group=group
        )
        safe_local_weight = mx.where(
            local_invalid, mx.array(0.0, mx.float32), local_weight
        )
        global_weight = mx.distributed.all_sum(
            safe_local_weight, group=group
        )

        try:
            mode_value = int(mode_count)
            invalid_value = int(invalid_count)
            weight_value = float(global_weight)
        except (RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Unable to coordinate distributed gradient weights while "
                "tracing. Pre-aggregate a fixed gradient tree and set "
                "aggregate_distributed_gradients=False for mx.compile."
            ) from error
        else:
            if mode_value not in {0, size}:
                raise ValueError(
                    "gradient_weight must be provided on every rank or none"
                )
            if invalid_value:
                raise ValueError(
                    "gradient_weight must be finite and non-negative"
                )
            if not math.isfinite(weight_value) or weight_value <= 0:
                raise ValueError(
                    "The global gradient_weight must be positive and finite"
                )
            effective_weight = safe_local_weight
            denominator = global_weight

        return {
            path: (
                mx.distributed.all_sum(
                    gradient.astype(mx.float32) * effective_weight,
                    group=group,
                )
                / denominator
            )
            for path, gradient in sorted(flat_gradients.items())
        }

    def _validate_distributed_gradient_schema(self, flat_gradients):
        """Coordinate a fixed schema before any rank-dependent collectives."""
        group = self.distributed_group
        size = group.size()
        local_count = len(flat_gradients)
        local_fingerprint = self._gradient_schema_fingerprint(flat_gradients)
        global_count = mx.distributed.all_sum(
            mx.array(local_count, mx.uint32), group=group
        )
        global_fingerprint = mx.distributed.all_sum(
            mx.array(local_fingerprint, mx.uint32), group=group
        )
        local_schema_mismatch = mx.logical_or(
            global_count != local_count * size,
            global_fingerprint != local_fingerprint * size,
        )
        mismatch_count = mx.distributed.all_sum(
            local_schema_mismatch.astype(mx.uint32), group=group
        )
        try:
            mismatch_value = int(mismatch_count)
        except (RuntimeError, TypeError, ValueError) as error:
            # Do not proceed to per-gradient or per-layer collectives unless
            # every rank has coherently validated the static gradient tree.
            raise RuntimeError(
                "Unable to validate distributed gradient schemas while "
                "tracing. Compile with a fixed pre-aggregated gradient tree."
            ) from error
        if mismatch_value:
            raise ValueError("Distributed gradient trees differ across ranks")

    def _apply_gradients_impl(
        self,
        gradients: dict,
        parameters: dict,
        *,
        model=None,
        masks=None,
        refresh=None,
        gradient_weight=None,
    ):
        model = model or (parameters if isinstance(parameters, nn.Module) else None)
        model = model or self._model
        if model is None:
            raise ValueError("KFAC requires a model")
        self._model = model
        if not self._initialized:
            if self._state_assigned or (
                isinstance(self.state, dict) and self.state.get("layers")
            ):
                self.load_state(self.state, model)
            else:
                self._initializing_during_update = True
                try:
                    # The trainable tree excludes frozen parameters while
                    # retaining every possible conditional/fallback leaf, so
                    # ownership and AdamW state stay static across steps.
                    self.init(model.trainable_parameters(), model=model)
                finally:
                    self._initializing_during_update = False

        layers = self._find_layers(model)
        flat_grads = dict(tree_flatten(gradients))
        distributed = (
            self.distributed_group is not None
            and self.distributed_group.size() > 1
        )
        if distributed:
            self._validate_distributed_gradient_schema(flat_grads)
            if self.aggregate_distributed_gradients:
                flat_grads = self._aggregate_distributed_gradients(
                    flat_grads, gradient_weight
                )
        flat_params = dict(tree_flatten(parameters))
        supported = set()
        candidates = {}
        masks = masks or {}

        for name, scheduler in self._schedulers.items():
            self.state[name] = scheduler(self.step)
        self.state["step"] = self.step + 1
        if refresh is None:
            try:
                # Eager path: select on the host so non-refresh steps do not
                # construct CPU decompositions at all.
                refresh = (
                    (int(self.step) - 1) % self.inverse_update_interval
                ) == 0 or bool(self.state["force_refresh"])
            except (RuntimeError, ValueError):
                # Compiled path: MLX 0.32 has no lazy conditional. Build the
                # candidate and select it with an array predicate.
                refresh = mx.logical_or(
                    # The initial counter is zero in KFAC-JAX, so refresh the
                    # first observed step and then every configured period.
                    (
                        (self.step - 1) % self.inverse_update_interval
                    )
                    == 0,
                    self.state["force_refresh"],
                )
        else:
            refresh = bool(refresh)

        for path, layer in layers.items():
            prefix = f"{path}." if path else ""
            weight_path = prefix + "weight"
            if weight_path not in flat_grads:
                continue
            state = self.state["layers"][_state_key(path)]
            try:
                batch_a, batch_g, sample_count, curvature_scale = self._observed_factors(
                    path, layer, state, masks.get(path)
                )
            except MissingObservationError:
                if (
                    self.distributed_group is not None
                    and self.distributed_group.size() > 1
                ):
                    # Every rank must execute the same factor collectives.
                    # A rank on which this conditional layer was inactive
                    # contributes zero local numerators and counts.
                    (
                        batch_a,
                        batch_g,
                        sample_count,
                        curvature_scale,
                    ) = self._empty_observed_factors(path, state)
                else:
                    # Ownership is static: a conditional supported layer with
                    # no observation is skipped rather than reassigned to
                    # AdamW. A requested refresh still updates its cached
                    # decomposition (for example after damping adaptation).
                    self._refresh_decomposition(state, refresh)
                    supported.add(weight_path)
                    bias_path = prefix + "bias"
                    if bias_path in flat_grads:
                        supported.add(bias_path)
                    continue
            decay = self.factor_decay
            has_samples = sample_count > 0
            old_scale = state["curvature_scale"]
            next_curvature_scale = (
                decay * old_scale
                + (1 - decay) * curvature_scale.astype(mx.float32)
            )

            def update_factor(old_factor, batch_factor):
                numerator = (
                    decay * old_scale * old_factor
                    + (1 - decay)
                    * curvature_scale.astype(mx.float32)
                    * batch_factor
                )
                return (
                    numerator
                    / mx.maximum(
                        next_curvature_scale,
                        mx.array(1e-12, mx.float32),
                    )
                ).astype(mx.float32)

            if "A_diag" in state:
                next_a = update_factor(state["A_diag"], batch_a)
                state["A_diag"] = mx.where(
                    has_samples, next_a, state["A_diag"]
                )
                if "G_blocks" in state:
                    state["G_blocks"] = [
                        mx.where(
                            has_samples,
                            update_factor(old, new),
                            old,
                        )
                        for old, new in zip(state["G_blocks"], batch_g)
                    ]
                else:
                    next_g = update_factor(state["G"], batch_g)
                    state["G"] = mx.where(has_samples, next_g, state["G"])
            elif "A_blocks" in state:
                state["A_blocks"] = [
                    mx.where(
                        has_samples,
                        update_factor(old, new),
                        old,
                    )
                    for old, new in zip(state["A_blocks"], batch_a)
                ]
                state["G_blocks"] = [
                    mx.where(
                        has_samples,
                        update_factor(old, new),
                        old,
                    )
                    for old, new in zip(state["G_blocks"], batch_g)
                ]
            else:
                next_a = update_factor(state["A"], batch_a)
                next_g = update_factor(state["G"], batch_g)
                state["A"] = mx.where(has_samples, next_a, state["A"])
                state["G"] = mx.where(has_samples, next_g, state["G"])
            state["curvature_scale"] = mx.where(
                has_samples,
                next_curvature_scale,
                state["curvature_scale"],
            )
            self._refresh_decomposition(state, refresh)

            weight = flat_params[weight_path]
            gradient = flat_grads[weight_path].astype(mx.float32).reshape(
                weight.shape[0], -1
            )
            bias_path = prefix + "bias"
            has_bias = "bias" in layer and bias_path in flat_grads
            if has_bias:
                gradient = mx.concatenate(
                    [gradient, flat_grads[bias_path].astype(mx.float32)[:, None]],
                    axis=1,
                )
            natural = self._precondition(state, gradient).astype(mx.float32)
            candidates[path] = (layer, state, gradient, natural)
            supported.add(weight_path)
            if has_bias:
                supported.add(bias_path)

        if refresh is not False:
            self.state["force_refresh"] = mx.zeros_like(
                self.state["force_refresh"]
            )

        directions = {
            path: self.momentum * state["velocity"] + natural
            for path, (_, state, _, natural) in candidates.items()
        }
        if candidates:
            if self.kl_clip_mode == "natural":
                quadratic = sum(
                    (
                        mx.sum(candidate[2] * candidate[3])
                        for candidate in candidates.values()
                    ),
                    mx.array(0.0, mx.float32),
                )
            else:
                quadratic = sum(
                    (
                        mx.sum(
                            directions[path]
                            * self._curvature_product(
                                candidate[1], directions[path]
                            )
                        )
                        for path, candidate in candidates.items()
                    ),
                    mx.array(0.0, mx.float32),
                )
            if self.kl_clip is None:
                # Keep the captured output connected to its prior state under
                # mx.compile; a fresh primitive-less constant cannot replace a
                # captured output.
                kl_scale = mx.ones_like(self.state["last_kl_scale"])
            else:
                lr32 = self.learning_rate.astype(mx.float32)
                denominator = mx.maximum(
                    lr32 * lr32 * quadratic, mx.array(1e-16, mx.float32)
                )
                kl_scale = mx.minimum(
                    mx.array(1.0, mx.float32),
                    mx.sqrt(mx.array(self.kl_clip, mx.float32) / denominator),
                )
            self.state["last_kl_scale"] = kl_scale.astype(mx.float32)

        updates = {}
        for path, (layer, state, gradient, natural) in candidates.items():
            prefix = f"{path}." if path else ""
            weight_path = prefix + "weight"
            weight = flat_params[weight_path]
            if self.kl_clip_mode == "natural":
                velocity = self.momentum * state["velocity"] + (
                    self.state["last_kl_scale"] * natural
                )
                applied_update = velocity
            else:
                velocity = directions[path]
                applied_update = self.state["last_kl_scale"] * velocity
            state["velocity"] = velocity.astype(mx.float32)
            lr = self.learning_rate.astype(weight.dtype)
            flat_width = math.prod(weight.shape[1:])
            weight_update = applied_update[:, :flat_width].reshape(weight.shape)
            updates[weight_path] = (
                weight * (1 - lr * self.weight_decay)
                - lr * weight_update.astype(weight.dtype)
            )
            bias_path = prefix + "bias"
            if bias_path in supported:
                bias = flat_params[bias_path]
                updates[bias_path] = (
                    bias * (1 - lr * self.weight_decay)
                    - lr * applied_update[:, -1].astype(bias.dtype)
                )

        fallback_grads = [(p, g) for p, g in flat_grads.items() if p not in supported]
        if fallback_grads:
            # AdamW should observe the global optimizer step even when fallback
            # leaves only occur conditionally.
            self._fallback.state["step"] = self.step - 1
            self._fallback.state["learning_rate"] = self.learning_rate
            fallback_updates = self._fallback.apply_gradients(
                tree_unflatten(fallback_grads), parameters
            )
            updates.update(tree_flatten(fallback_updates))
            self._state["fallback"] = self._fallback.state
        self._fallback.state["step"] = self.step
        self._state["fallback"] = self._fallback.state

        updates = tree_unflatten(list(updates.items()))
        try:
            # In eager mode, materialize all state and parameter updates before
            # committing the transaction so lazy decomposition/collective
            # failures are caught by the rollback path. Tracers cannot be
            # converted to int, so compiled graphs retain normal lazy capture.
            int(self.step)
        except (RuntimeError, TypeError, ValueError):
            pass
        else:
            _materialize_finite_arrays(
                updates=updates,
                optimizer_state=self.state,
            )
        return updates
