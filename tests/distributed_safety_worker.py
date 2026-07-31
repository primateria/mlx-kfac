"""Distributed safety probes run under ``mlx.launch``."""

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_kfac import KFAC
from mlx_kfac.capture import MissingObservationError
from mlx_kfac.optimizer import _state_key


group = mx.distributed.init(backend="ring")
rank = group.rank()


class Conditional(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 1, bias=False)
        self.second = nn.Linear(2, 1, bias=False)
        self.first.weight = mx.zeros_like(self.first.weight)
        self.second.weight = mx.zeros_like(self.second.weight)

    def __call__(self, x, use_first):
        return self.first(x) if use_first else self.second(x)


model = Conditional()
optimizer = KFAC(
    0.01,
    model=model,
    factor_decay=0.0,
    inverse_update_interval=1,
    distributed_group=group,
)
if rank == 0:
    x = mx.array([[1.0, 2.0]])
    upstream = mx.array([[2.0]])
    use_first = True
else:
    x = mx.array([[3.0, 4.0]])
    upstream = mx.array([[3.0]])
    use_first = False

_, gradients = nn.value_and_grad(
    model,
    lambda values: mx.sum(model(values, use_first) * upstream),
)(x)
optimizer.update(model, gradients)
first_state = optimizer.state["layers"][_state_key("first")]
second_state = optimizer.state["layers"][_state_key("second")]
mx.eval(model.parameters(), optimizer.state)
if not bool(
    mx.allclose(first_state["A"], mx.array([[1.0, 2.0], [2.0, 4.0]]))
):
    raise AssertionError(f"rank {rank}: divergent first A mismatch")
if not bool(mx.allclose(first_state["G"], mx.array([[4.0]]))):
    raise AssertionError(f"rank {rank}: divergent first G mismatch")
if not bool(
    mx.allclose(second_state["A"], mx.array([[9.0, 12.0], [12.0, 16.0]]))
):
    raise AssertionError(f"rank {rank}: divergent second A mismatch")
if not bool(mx.allclose(second_state["G"], mx.array([[9.0]]))):
    raise AssertionError(f"rank {rank}: divergent second G mismatch")


def expect_invalid_weight(local_weight):
    layer = nn.Linear(2, 1, bias=False)
    optimizer = KFAC(
        0.0,
        model=layer,
        distributed_group=group,
        inverse_update_interval=1,
    )
    optimizer.init(layer.trainable_parameters(), model=layer)
    values = mx.ones((1, 2))
    _, gradients = nn.value_and_grad(
        layer, lambda inputs: mx.sum(layer(inputs))
    )(values)
    try:
        optimizer.update(
            layer, gradients, gradient_weight=local_weight
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"rank {rank}: invalid gradient_weight did not fail"
        )
    if int(optimizer.state["step"]) != 0:
        raise AssertionError(f"rank {rank}: failed update changed step")
    try:
        _ = layer.kfac_observations
    except MissingObservationError:
        pass
    else:
        raise AssertionError(f"rank {rank}: failed update left capture")


expect_invalid_weight(-1.0 if rank == 0 else 1.0)
expect_invalid_weight(0.0)
expect_invalid_weight(float("inf") if rank == 1 else 1.0)
expect_invalid_weight(None if rank == 0 else 1.0)

def expect_schema_mismatch(aggregate_gradients):
    layer = nn.Linear(2, 1, bias=False)
    optimizer = KFAC(
        0.0,
        model=layer,
        distributed_group=group,
        aggregate_distributed_gradients=aggregate_gradients,
        inverse_update_interval=1,
    )
    optimizer.init(layer.trainable_parameters(), model=layer)
    values = mx.ones((1, 2))
    _, gradients = nn.value_and_grad(
        layer, lambda inputs: mx.sum(layer(inputs))
    )(values)
    if rank == 0:
        gradients = {}
    try:
        optimizer.update(layer, gradients)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"rank {rank}: gradient schema mismatch did not fail "
            f"with aggregation={aggregate_gradients}"
        )
    if int(optimizer.state["step"]) != 0:
        raise AssertionError(f"rank {rank}: schema failure changed step")
    try:
        _ = layer.kfac_observations
    except MissingObservationError:
        pass
    else:
        raise AssertionError(f"rank {rank}: schema failure left capture")


expect_schema_mismatch(True)
expect_schema_mismatch(False)


def aggregate_or_fail(numerator, count, kind, path):
    if rank == 0:
        raise RuntimeError("rank-local aggregation failure")
    return numerator, count


layer = nn.Linear(2, 1, bias=False)
optimizer = KFAC(
    0.0,
    model=layer,
    distributed_group=group,
    factor_aggregator=aggregate_or_fail,
    inverse_update_interval=1,
)
optimizer.init(layer.trainable_parameters(), model=layer)
values = mx.ones((1, 2))
_, gradients = nn.value_and_grad(
    layer, lambda inputs: mx.sum(layer(inputs))
)(values)
try:
    optimizer.update(layer, gradients)
except RuntimeError:
    pass
else:
    raise AssertionError(f"rank {rank}: callback failure was not coordinated")
if int(optimizer.state["step"]) != 0:
    raise AssertionError(f"rank {rank}: callback failure changed step")
try:
    _ = layer.kfac_observations
except MissingObservationError:
    pass
else:
    raise AssertionError(f"rank {rank}: callback failure left capture")

optimizer.factor_aggregator = None
optimizer._factor_aggregator_uses_stats = False
_, gradients = nn.value_and_grad(
    layer, lambda inputs: mx.sum(layer(inputs))
)(values)
optimizer.update(layer, gradients)
mx.eval(layer.parameters(), optimizer.state)
if int(optimizer.state["step"]) != 1:
    raise AssertionError(f"rank {rank}: recovery step failed")

marker_dir = os.environ.get("MLX_KFAC_MARKER_DIR")
if marker_dir:
    Path(marker_dir, f"rank-{rank}.ok").touch()
print(f"rank={rank} distributed_safety=ok")
