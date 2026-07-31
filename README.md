# mlx-kfac

Native Kronecker-factored approximate curvature for MLX. The implementation
uses MLX operations only: there are no JAX/PyTorch dependencies, explicit
matrix inverses, or custom Metal kernels.

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

## Curvature and stabilization options

The optimizer maintains float32 EMA activation and output-gradient factors.
Weights use the MLX `[out, in]` orientation and are preconditioned as
`G^-1 D A^-1` using cached decompositions and linear solves.

Available options include:

- `damping_strategy="uniform"` or trace-based `"pi"`
- `decomposition="cholesky"` or `"eigh"`
- preconditioned-gradient `momentum`
- global `kl_clip`
- adaptive damping through `adapt_damping(reduction_ratio)`
- `max_factor_size` for two-dimensional block partitioning
- `attention_head_blocks=True` for per-head projection factors

Biases are included exactly once through a homogeneous activation coordinate.
Embedding state stores `A_diag` with shape `[vocabulary]`, never a dense
vocabulary-by-vocabulary matrix.

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

Initialize optimizer state, then capture both model and optimizer state:

```python
optimizer.init(model.trainable_parameters(), model=model)

def step(x, y):
    value, gradients = loss_and_grad(x, y)
    optimizer.update(model, gradients)
    return value

captured = [model.state, optimizer.state]
step = mx.compile(step, inputs=captured, outputs=captured)
```

MLX 0.32 has no lazy conditional primitive. The default compiled path computes
a candidate decomposition and selects it at `inverse_update_interval`
boundaries. To truly skip decomposition work in host-controlled or separately
compiled paths, pass the static flag `refresh=False`; use `refresh=True` on
scheduled refresh steps.

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

## Explicit limitations

- Grouped/depthwise convolutions use AdamW fallback.
- Aliased supported modules and tied embedding/output-projection parameters
  are rejected because they combine incompatible curvature roles.
- Instrumented Python module objects are not intended for pickle/deepcopy;
  save model parameters and optimizer tree state instead.
- Cholesky, eigendecomposition, and triangular solves use MLX's CPU stream on
  MLX 0.32; parameters and the rest of the graph may remain on the GPU.
