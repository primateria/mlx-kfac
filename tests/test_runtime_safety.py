import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_kfac import KFAC
from mlx_kfac.capture import MissingObservationError


class RuntimeSafetyTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        actual = dict(tree_flatten(actual))
        expected = dict(tree_flatten(expected))
        self.assertEqual(set(actual), set(expected))
        mx.eval(actual, expected)
        for path in actual:
            self.assertTrue(
                bool(mx.array_equal(actual[path], expected[path])),
                f"state changed at {path}",
            )

    @staticmethod
    def snapshot(state):
        return {
            path: mx.array(value)
            for path, value in tree_flatten(state)
        }

    def test_failed_update_rolls_back_state_and_clears_capture(self):
        layer = nn.Linear(2, 1, bias=False)
        optimizer = KFAC(
            0.1,
            model=layer,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        optimizer.init(layer.trainable_parameters(), model=layer)
        mx.eval(layer.parameters(), optimizer.state)
        state_before = self.snapshot(optimizer.state)
        weight_before = layer.weight

        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        _, gradients = nn.value_and_grad(
            layer, lambda values: mx.sum(layer(values))
        )(x)
        with mock.patch.object(
            optimizer,
            "_precondition",
            side_effect=RuntimeError("injected precondition failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                optimizer.update(layer, gradients)

        self.assert_state_equal(optimizer.state, state_before)
        self.assertTrue(bool(mx.array_equal(layer.weight, weight_before)))
        with self.assertRaises(MissingObservationError):
            _ = layer.kfac_observations

        _, gradients = nn.value_and_grad(
            layer, lambda values: mx.sum(layer(values))
        )(x)
        optimizer.update(layer, gradients)
        mx.eval(layer.parameters(), optimizer.state)
        self.assertEqual(int(optimizer.state["step"]), 1)
        self.assertFalse(bool(mx.array_equal(layer.weight, weight_before)))

    def test_partial_model_update_is_rolled_back(self):
        class PartiallyFailingLinear(nn.Linear):
            def __init__(self):
                super().__init__(2, 1)
                self.fail_update = False

            def update(self, parameters):
                if self.fail_update:
                    self.weight = parameters["weight"]
                    raise RuntimeError("partial model update")
                return super().update(parameters)

        layer = PartiallyFailingLinear()
        optimizer = KFAC(
            0.1,
            model=layer,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        optimizer.init(layer.trainable_parameters(), model=layer)
        mx.eval(layer.state, optimizer.state)
        model_before = self.snapshot(layer.state)
        state_before = self.snapshot(optimizer.state)
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        _, gradients = nn.value_and_grad(
            layer, lambda values: mx.sum(layer(values))
        )(x)

        layer.fail_update = True
        with self.assertRaisesRegex(RuntimeError, "partial model"):
            optimizer.update(layer, gradients)

        self.assert_state_equal(layer.state, model_before)
        self.assert_state_equal(optimizer.state, state_before)
        with self.assertRaises(MissingObservationError):
            _ = layer.kfac_observations

        layer.fail_update = False
        _, gradients = nn.value_and_grad(
            layer, lambda values: mx.sum(layer(values))
        )(x)
        optimizer.update(layer, gradients)
        self.assertEqual(int(optimizer.state["step"]), 1)


if __name__ == "__main__":
    unittest.main()
