import math
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_kfac import KFAC
from mlx_kfac.optimizer import _cholesky, _eigh, _solve_cholesky, _solve_eigen


class NumericalHardeningTests(unittest.TestCase):
    def assert_finite(self, value):
        mx.eval(value)
        self.assertTrue(bool(mx.all(mx.isfinite(value))))

    def test_nonfinite_and_unrepresentable_scalar_options_are_rejected(self):
        cases = (
            ("learning_rate", math.nan),
            ("damping", math.inf),
            ("factor_decay", math.nan),
            ("momentum", math.inf),
            ("kl_clip", math.nan),
            ("weight_decay", math.inf),
            ("fallback_eps", math.nan),
            ("output_gradient_scale", math.inf),
            ("damping_adaptation_decay", math.nan),
            ("min_damping", math.nan),
            ("max_damping", math.inf),
            ("decomposition_relative_eigenvalue_floor", math.nan),
            ("damping", 1e-50),
            ("damping", 1e39),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    KFAC(**{name: value})

    def test_damping_bounds_and_relative_floor_are_validated(self):
        invalid = (
            {"damping": 1e-9},
            {"damping": 2.0, "max_damping": 1.0},
            {"min_damping": 2.0, "damping": 1.0},
            {"decomposition_relative_eigenvalue_floor": 0.0},
            {"decomposition_relative_eigenvalue_floor": 2.0},
        )
        for options in invalid:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    KFAC(**options)

    def test_adaptive_damping_clips_to_configured_finite_bounds(self):
        optimizer = KFAC(
            damping=1.0,
            adaptive_damping=True,
            damping_adaptation_interval=1,
            damping_adaptation_decay=0.1,
            min_damping=0.5,
            max_damping=2.0,
        )
        self.assertAlmostEqual(float(optimizer.adapt_damping(0.0)), 2.0)
        self.assertAlmostEqual(float(optimizer.adapt_damping(0.0)), 2.0)
        self.assertAlmostEqual(float(optimizer.adapt_damping(1.0)), 0.5)
        self.assertAlmostEqual(float(optimizer.adapt_damping(1.0)), 0.5)

    def test_scale_aware_eigen_floor_and_singular_cache_solve(self):
        matrix = mx.array([[1.0, 1.0], [1.0, 1.0]], mx.float32) * 1e12
        values, vectors = _eigh(matrix, relative_floor=1e-6)
        mx.eval(values, vectors)
        self.assert_finite(values)
        self.assertGreaterEqual(float(mx.min(values)), 1.9e6)

        rhs = mx.ones((2, 1), mx.float32)
        eigen_solution = _solve_eigen(values, vectors, rhs)
        self.assert_finite(eigen_solution)

        # A legacy or externally supplied singular cache must not reach MLX's
        # raw triangular inversion with a zero pivot.
        triangular_solution = _solve_cholesky(
            mx.zeros((2, 2), mx.float32), rhs
        )
        self.assert_finite(triangular_solution)

    def test_cholesky_projects_nonfinite_semidefinite_input_to_finite_spd(self):
        matrix = mx.array(
            [[math.inf, math.nan], [math.nan, 0.0]], mx.float32
        )
        factor = _cholesky(matrix)
        solution = _solve_cholesky(factor, mx.ones((2, 1), mx.float32))
        self.assert_finite(factor)
        self.assert_finite(solution)
        self.assertTrue(bool(mx.all(mx.diag(factor) > 0)))

    def test_rank_deficient_extreme_scale_update_is_finite(self):
        x = mx.array(
            [[1e6, 1e6], [2e6, 2e6]], dtype=mx.float32
        )
        upstream = mx.array([[1.0, 1.0], [2.0, 2.0]], mx.float32)
        for decomposition in ("cholesky", "eigh"):
            with self.subTest(decomposition=decomposition):
                layer = nn.Linear(2, 2, bias=False)
                layer.weight = mx.zeros_like(layer.weight)
                optimizer = KFAC(
                    1e-3,
                    model=layer,
                    damping=1e-3,
                    damping_strategy="pi",
                    decomposition=decomposition,
                    factor_decay=0.0,
                    inverse_update_interval=1,
                )
                _, gradients = nn.value_and_grad(
                    layer,
                    lambda values: mx.sum(layer(values) * upstream),
                )(x)
                optimizer.update(layer, gradients)
                mx.eval(layer.parameters(), optimizer.state)
                self.assert_finite(layer.weight)
                state = optimizer.state["layers"]["__root__"]
                decomposition_value = (
                    state["A_cholesky"]
                    if decomposition == "cholesky"
                    else state["A_eigenvalues"]
                )
                self.assert_finite(decomposition_value)

    def test_float32_factor_overflow_rolls_back_instead_of_committing_nan(self):
        layer = nn.Linear(2, 1, bias=False)
        layer.weight = mx.zeros_like(layer.weight)
        optimizer = KFAC(
            0.01,
            model=layer,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        optimizer.init(layer.trainable_parameters(), model=layer)
        state_before = {
            path: mx.array(value)
            for path, value in tree_flatten(optimizer.state)
        }
        weight_before = mx.array(layer.weight)
        values = mx.ones((1, 2)) * 1e20
        _, gradients = nn.value_and_grad(
            layer, lambda inputs: mx.sum(layer(inputs))
        )(values)

        with self.assertRaisesRegex(FloatingPointError, "Non-finite"):
            optimizer.update(layer, gradients)

        mx.eval(layer.weight, optimizer.state)
        self.assertTrue(bool(mx.array_equal(layer.weight, weight_before)))
        restored = dict(tree_flatten(optimizer.state))
        for path, expected in state_before.items():
            self.assertTrue(bool(mx.array_equal(restored[path], expected)))


if __name__ == "__main__":
    unittest.main()
