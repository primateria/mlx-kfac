# mlx-kfac

Native Kronecker-factored approximate curvature for MLX. The implementation
uses MLX operations only: there are no JAX/PyTorch dependencies, explicit
matrix inverses, or custom Metal kernels.

## Installation and compatibility

Install the published package with:

```bash
python -m pip install mlx-kfac
```

The supported release range is:

- Python 3.10 through 3.14
- MLX 0.25.2 through the 0.32 release series
- Apple silicon with macOS 14 or newer

The compatibility floor and current MLX release are separate CI targets, and
the Python matrix runs against MLX 0.32. The package dependency is capped
below MLX 0.33 because MLX is still pre-1.0; support for a new MLX minor is
opened after its test matrix passes.

MLX 0.32 officially requires native Python 3.10 or newer and macOS 14 or
newer on Apple silicon. MLX also publishes Linux CPU/CUDA and Windows
backends, but those backends are not yet release-qualified for mlx-kfac.
See the [official MLX installation requirements][mlx-install] for the current
upstream platform details.

[mlx-install]: https://ml-explore.github.io/mlx/build/html/install.html

## Supported curvature blocks

- Ordinary `mlx.nn.Linear`
- `mlx.nn.Conv1d` and `mlx.nn.Conv2d` with `groups=1`
- `mlx.nn.Embedding` using a diagonal token-frequency factor
- All four Linear projections inside `mlx.nn.MultiHeadAttention`
- Optional head-aware and maximum-size factor partitioning

Unsupported parameters, including normalization parameters and grouped
convolutions, are updated by an internal AdamW optimizer.

## Basic use

Pass the model when constructing the optimizer **before** creating or invoking
the differentiated training function. Registration instruments supported
modules in place while preserving their parameters and paths.

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_kfac import KFAC

model = nn.Linear(4, 2)
optimizer = KFAC(
    learning_rate=1e-2,
    model=model,
    damping=1e-3,
    factor_decay=0.95,
    inverse_update_interval=10,
    loss_reduction="mean",
)

def loss(x, y):
    return mx.mean((model(x) - y) ** 2)

loss_and_grad = nn.value_and_grad(model, loss)
value, gradients = loss_and_grad(x, y)
optimizer.update(model, gradients)
mx.eval(model.parameters(), optimizer.state)
```

`KFACLinear` remains available for compatibility, but ordinary `nn.Linear`
instances are preferred.

The installed package version is available without importing MLX:

```python
from importlib.metadata import version

print(version("mlx-kfac"))
```

## Curvature and stabilization options

The optimizer maintains float32 EMA activation and output-gradient factors.
Weights use the MLX `[out, in]` orientation and are preconditioned as
`G^-1 D A^-1` using cached decompositions and linear solves.

Available options include:

- `damping_strategy="uniform"` or trace-based `"pi"`
- `decomposition="cholesky"` or `"eigh"`
- bounded adaptive damping through `min_damping` and `max_damping`
- a relative eigenvalue floor for stable decompositions
- preconditioned-gradient `momentum`
- global `kl_clip`
- adaptive damping through `adapt_damping(reduction_ratio)`
- `max_factor_size` for two-dimensional block partitioning (default: 1024)
- `attention_head_blocks=True` for per-head projection factors

Biases are included exactly once through a homogeneous activation coordinate.
Embedding state stores `A_diag` with shape `[vocabulary]`, never a dense
vocabulary-by-vocabulary matrix.

Before allocating factors, KFAC estimates persistent optimizer array state
(including float32 factors, decompositions, velocities, and fallback moments)
and rejects projections above `max_state_size_bytes` (default: 2 GiB):

```python
projected_bytes = optimizer.estimate_state_size_bytes(model)
optimizer.init(model.trainable_parameters(), model=model)
assert optimizer.estimated_state_size_bytes == projected_bytes
```

Set either safety limit to `None` only when the resulting unpartitioned
factors and optimizer state are known to fit in memory.

`loss_reduction="sum"` (the default) treats captured cotangents as
per-example/summed gradients. Use `"mean"` when the objective is averaged over
the leading batch dimension; K-FAC then restores batch-duplication-invariant
output-gradient factors. `output_gradient_scale` is available for custom loss
normalizations.

## Repeated calls and padding

Sequence/spatial uses and repeated calls of one module are aggregated with an
explicit curvature repetition scale. Factor masks are keyed by module path:

```python
optimizer.update(
    model,
    gradients,
    masks={
        "attention.query_proj": query_mask,
        "attention.key_proj": key_mask,
        "attention.value_proj": key_mask,
        "attention.out_proj": query_mask,
    },
)
```

For a module invoked multiple times, provide a list of masks in call order.
Masks affect curvature statistics; the loss itself must apply the same mask so
the already-aggregated parameter gradient also excludes padded positions.

## Compilation

Initialize optimizer state, then capture both model and optimizer state. The
recommended helper compiles separate static refresh and cached graphs, so
decompositions are skipped between scheduled refreshes:

```python
optimizer.init(model.trainable_parameters(), model=model)

def step(x, y, *, refresh):
    value, gradients = loss_and_grad(x, y)
    optimizer.update(model, gradients, refresh=refresh)
    return value

captured = [model.state, optimizer.state]
step = optimizer.compile_step(step, inputs=captured, outputs=captured)
```

The returned `CompiledKFACStep` owns the refresh schedule and selects between
the two compiled graphs on the host. Each invocation is materialized inside a
model/optimizer transaction so a decomposition or collective failure rolls
back captured state before it is exposed to the next step. MLX 0.32 has no
lazy conditional primitive, so a single directly compiled graph with a
dynamic refresh predicate computes a candidate decomposition on every
invocation. If graphs are compiled or controlled manually instead, pass
static `refresh=False` on cached steps and `refresh=True` on scheduled refresh
steps.

Activation/output-cotangent capture must be consumed in the same differentiated
step. Compile the whole training step as above; compiling `value_and_grad`
separately and calling the optimizer afterward is not supported by MLX custom
VJP side-effect capture.

## Checkpoint restoration

Optimizer state is an MLX-compatible tree. Restore it after constructing an
optimizer with the same structural configuration:

```python
restored = KFAC(model=new_model, max_factor_size=256)
restored.load_state(saved_optimizer_state, new_model)
```

Direct assignment through `restored.state = saved_optimizer_state` is also
validated on the next update. Shape-incompatible factor state raises an error.

## Distributed statistics and gradients

Pass an initialized MLX distributed group to aggregate factor Gram sums and
sample counts globally before normalization:

```python
group = mx.distributed.init()
optimizer = KFAC(model=model, distributed_group=group)
```

Gradients are all-reduced as rank means by default. For unequal local batch
sizes and locally mean-reduced losses, pass the local count:

```python
optimizer.update(model, gradients, gradient_weight=local_batch_size)
```

This produces the globally weighted gradient mean. A custom
`factor_aggregator(numerator, count, kind, path)` can be used when collectives
are managed externally.

Distributed gradient schemas are coordinated before any rank-dependent
collectives, including when `aggregate_distributed_gradients=False`. During
compilation that validation must complete coherently while tracing; if the MLX
backend cannot do so, pre-aggregate a fixed gradient tree and construct KFAC
with `aggregate_distributed_gradients=False`. Factor statistics may still use
`distributed_group`.

## Explicit limitations

- Grouped/depthwise convolutions use AdamW fallback.
- Aliased supported modules and tied embedding/output-projection parameters
  are rejected because they combine incompatible curvature roles.
- Instrumented Python module objects are not intended for pickle/deepcopy;
  save model parameters and optimizer tree state instead.
- Cholesky, eigendecomposition, and triangular solves use MLX's CPU stream on
  MLX 0.32; parameters and the rest of the graph may remain on the GPU.

## Versioning and development

mlx-kfac follows [Semantic Versioning](https://semver.org/). While the version
is below 1.0, a minor release may change optimizer internals or checkpoint
schemas. Public names exported from `mlx_kfac` are kept stable within a minor
series, and checkpoint incompatibilities raise an explicit validation error.
See the [changelog][project-changelog] for release-level compatibility notes.

For a development checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Source distributions contain the complete test suite, including the
two-process distributed workers; wheels contain only the importable package
and license metadata. Maintainer release steps are documented in
[RELEASING.md][release-guide].

[project-changelog]: https://github.com/primateria/mlx-kfac/blob/main/CHANGELOG.md
[release-guide]: https://github.com/primateria/mlx-kfac/blob/main/RELEASING.md
