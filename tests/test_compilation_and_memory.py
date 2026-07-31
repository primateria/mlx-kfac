import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import mlx_kfac.runtime as runtime_module

from mlx_kfac import (
    CompiledKFACStep,
    KFAC,
    compile_kfac_step,
    estimate_kfac_state_bytes,
)
from mlx_kfac.capture import MissingObservationError
from mlx_kfac.runtime import validate_state_memory_limit


class CompilationAndMemoryTests(unittest.TestCase):
    def test_dual_graph_dispatches_only_scheduled_refresh_steps(self):
        layer = nn.Linear(2, 1, bias=False)
        optimizer = KFAC(
            0.0,
            model=layer,
            factor_decay=0.0,
            inverse_update_interval=3,
        )
        optimizer.init(layer.trainable_parameters(), model=layer)

        def loss(values):
            return mx.sum(layer(values) ** 2)

        value_and_grad = nn.value_and_grad(layer, loss)

        def step(values, *, refresh):
            objective, gradients = value_and_grad(values)
            optimizer.update(layer, gradients, refresh=refresh)
            return objective

        captured = [layer.state, optimizer.state]
        compiled_step = compile_kfac_step(
            optimizer,
            step,
            inputs=captured,
            outputs=captured,
        )
        self.assertIsInstance(compiled_step, CompiledKFACStep)
        for scale in (1.0, 2.0, 3.0, 4.0, 5.0):
            compiled_step(mx.ones((2, 2)) * scale)
        mx.eval(layer.parameters(), optimizer.state)

        self.assertEqual(compiled_step.refresh_calls, 2)
        self.assertEqual(compiled_step.cached_calls, 3)
        self.assertFalse(compiled_step.last_refresh)
        self.assertEqual(int(optimizer.state["step"]), 5)

    def test_compile_dispatcher_requires_refresh_keyword(self):
        layer = nn.Linear(1, 1)
        optimizer = KFAC(model=layer)
        optimizer.init(layer.trainable_parameters(), model=layer)
        with self.assertRaisesRegex(TypeError, "refresh"):
            compile_kfac_step(
                optimizer,
                lambda values: layer(values),
                inputs=[layer.state, optimizer.state],
                outputs=[layer.state, optimizer.state],
            )

    def test_compiled_dispatch_rolls_back_failed_runtime_invocation(self):
        layer = nn.Linear(2, 1, bias=False)
        optimizer = KFAC(
            0.01,
            model=layer,
            factor_decay=0.0,
            inverse_update_interval=3,
        )
        optimizer.init(layer.trainable_parameters(), model=layer)

        def loss(values):
            return mx.sum(layer(values) ** 2)

        value_and_grad = nn.value_and_grad(layer, loss)

        def step(values, *, refresh):
            objective, gradients = value_and_grad(values)
            optimizer.update(layer, gradients, refresh=refresh)
            return objective

        captured = [layer.state, optimizer.state]
        compiled_step = compile_kfac_step(
            optimizer,
            step,
            inputs=captured,
            outputs=captured,
        )
        values = mx.ones((2, 2))
        compiled_step(values)
        compiled_step(values)
        state_before = {
            path: mx.array(leaf)
            for path, leaf in tree_flatten(optimizer.state)
        }
        weight_before = mx.array(layer.weight)
        _, _ = value_and_grad(values)
        self.assertTrue(layer.kfac_observations)

        real_materialize = runtime_module._materialize_finite_arrays
        materialized = {}

        def fail_after_materialization(**trees):
            real_materialize(**trees)
            materialized["step"] = int(optimizer.state["step"])
            materialized["weight"] = mx.array(layer.weight)
            raise RuntimeError("injected compiled runtime failure")

        with mock.patch.object(
            compiled_step,
            "_materialize",
            side_effect=fail_after_materialization,
        ):
            with self.assertRaisesRegex(RuntimeError, "compiled runtime"):
                compiled_step(values)

        mx.eval(layer.weight, optimizer.state)
        self.assertEqual(materialized["step"], 3)
        self.assertFalse(
            bool(mx.array_equal(materialized["weight"], weight_before))
        )
        self.assertTrue(bool(mx.array_equal(layer.weight, weight_before)))
        restored = dict(tree_flatten(optimizer.state))
        self.assertEqual(set(restored), set(state_before))
        for path, expected in state_before.items():
            self.assertTrue(
                bool(mx.array_equal(restored[path], expected)),
                f"optimizer state changed at {path}",
            )
        with self.assertRaises(MissingObservationError):
            _ = layer.kfac_observations
        self.assertEqual(compiled_step.refresh_calls, 1)
        self.assertEqual(compiled_step.cached_calls, 1)
        self.assertFalse(compiled_step.last_refresh)

    def test_compiled_dispatch_rejects_nonfinite_state_before_commit(self):
        layer = nn.Linear(2, 1, bias=False)
        layer.weight = mx.zeros_like(layer.weight)
        optimizer = KFAC(
            0.01,
            model=layer,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        optimizer.init(layer.trainable_parameters(), model=layer)

        def loss(values):
            return mx.sum(layer(values))

        value_and_grad = nn.value_and_grad(layer, loss)

        def step(values, *, refresh):
            objective, gradients = value_and_grad(values)
            optimizer.update(layer, gradients, refresh=refresh)
            return objective

        captured = [layer.state, optimizer.state]
        compiled_step = optimizer.compile_step(
            step,
            inputs=captured,
            outputs=captured,
        )
        before = mx.array(layer.weight)
        with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
            compiled_step(mx.ones((1, 2)) * 1e20)

        mx.eval(layer.weight, optimizer.state)
        self.assertTrue(bool(mx.array_equal(layer.weight, before)))
        self.assertEqual(int(optimizer.state["step"]), 0)
        self.assertEqual(compiled_step.refresh_calls, 0)
        self.assertEqual(compiled_step.cached_calls, 0)

    def test_partitioning_reduces_estimated_state_and_limit_fails_early(self):
        layer = nn.Linear(16, 32, bias=True)
        unpartitioned = KFAC(model=layer, max_factor_size=None)
        partitioned = KFAC(model=layer, max_factor_size=8)
        full_bytes = estimate_kfac_state_bytes(unpartitioned, layer)
        partitioned_bytes = estimate_kfac_state_bytes(partitioned, layer)

        self.assertLess(partitioned_bytes, full_bytes)
        with self.assertRaisesRegex(MemoryError, "max_state_size_bytes"):
            validate_state_memory_limit(
                partitioned_bytes,
                partitioned_bytes - 1,
            )

    def test_safe_factor_default_and_integrated_memory_guard(self):
        layer = nn.Linear(16, 32, bias=True)
        optimizer = KFAC(
            model=layer,
            max_state_size_bytes=128,
        )
        self.assertEqual(optimizer.max_factor_size, 1024)
        with self.assertRaisesRegex(MemoryError, "Projected K-FAC"):
            optimizer.init(layer.trainable_parameters(), model=layer)
        self.assertFalse(optimizer._initialized)
        self.assertGreater(
            optimizer.estimated_state_size_bytes,
            optimizer.max_state_size_bytes,
        )

    def test_state_estimate_is_not_smaller_than_allocated_array_buffers(self):
        layer = nn.Linear(16, 32, bias=True)
        optimizer = KFAC(model=layer, max_state_size_bytes=None)
        estimate = optimizer.estimate_state_size_bytes(layer)
        optimizer.init(layer.trainable_parameters(), model=layer)
        actual = sum(leaf.nbytes for _, leaf in tree_flatten(optimizer.state))

        self.assertGreaterEqual(estimate, actual)


if __name__ == "__main__":
    unittest.main()
