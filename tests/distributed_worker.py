"""Two-rank integration probe run by ``mlx.launch``."""

import mlx.core as mx
import mlx.nn as nn

from mlx_kfac import KFAC


group = mx.distributed.init(backend="ring")
rank = group.rank()

layer = nn.Linear(2, 1, bias=False)
layer.weight = mx.zeros_like(layer.weight)
optimizer = KFAC(
    0.01,
    model=layer,
    factor_decay=0.0,
    inverse_update_interval=1,
    distributed_group=group,
    loss_reduction="mean",
)

if rank == 0:
    x = mx.array([[1.0, 2.0]])
    upstream = mx.array([[2.0]])
else:
    x = mx.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    upstream = mx.array([[1.0], [3.0], [4.0]])


def loss(values):
    return mx.mean(layer(values) * upstream)


_, gradients = nn.value_and_grad(layer, loss)(x)
optimizer.update(layer, gradients, gradient_weight=x.shape[0])
state = optimizer.state["layers"]["__root__"]
all_x = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
all_g = mx.array([[2.0], [1.0], [3.0], [4.0]])
expected_a = all_x.T @ all_x / 4
expected_g = all_g.T @ all_g / 4
mx.eval(state)
if not bool(mx.allclose(state["A"], expected_a)):
    raise AssertionError(f"rank {rank}: distributed A mismatch")
if not bool(mx.allclose(state["G"], expected_g)):
    raise AssertionError(f"rank {rank}: distributed G mismatch")
global_gradient = all_g.T @ all_x / 4
expected_natural = mx.linalg.solve(
    expected_g + 1e-3 * mx.eye(1),
    global_gradient,
    stream=mx.cpu,
)
expected_natural = mx.linalg.solve(
    expected_a + 1e-3 * mx.eye(2),
    expected_natural.T,
    stream=mx.cpu,
).T
if not bool(mx.allclose(layer.weight, -0.01 * expected_natural, atol=1e-5)):
    raise AssertionError(f"rank {rank}: distributed parameter mismatch")
print(f"rank={rank} distributed_kfac=ok")
