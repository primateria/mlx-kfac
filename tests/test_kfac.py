import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_kfac import KFAC, KFACLinear, precondition_linear_gradient


def as_float(x):
    mx.eval(x)
    return float(x)


class TestKFAC(unittest.TestCase):
    def assertArrayClose(self, actual, expected, atol=1e-5, rtol=1e-5):
        mx.eval(actual, expected)
        self.assertTrue(
            bool(mx.allclose(actual, expected, atol=atol, rtol=rtol)),
            f"\nactual={actual}\nexpected={expected}",
        )

    def test_precondition_matches_explicit_kronecker_inverse(self):
        # vec in column-major convention:
        # vec(G^-1 D A^-1) = (A^-T kron G^-1) vec(D).
        a = mx.array([[2.0, 0.3], [0.3, 1.4]], dtype=mx.float32)
        g = mx.array([[1.7, -0.2], [-0.2, 0.9]], dtype=mx.float32)
        d = mx.array([[0.4, -1.2], [2.0, 0.7]], dtype=mx.float32)
        actual = precondition_linear_gradient(
            d,
            mx.linalg.cholesky(g, stream=mx.cpu),
            mx.linalg.cholesky(a, stream=mx.cpu),
        )
        kron = mx.kron(a, g)
        vec_d = d.T.reshape(-1, 1)
        expected = mx.linalg.solve(kron, vec_d, stream=mx.cpu).reshape(2, 2).T
        self.assertArrayClose(actual, expected)

    def test_left_right_orientation_for_out_in_weight(self):
        a = mx.diag(mx.array([2.0, 4.0]))
        g = mx.diag(mx.array([5.0, 10.0, 20.0]))
        d = mx.ones((3, 2))
        actual = precondition_linear_gradient(
            d,
            mx.linalg.cholesky(g, stream=mx.cpu),
            mx.linalg.cholesky(a, stream=mx.cpu),
        )
        expected = mx.array(
            [[1 / 10, 1 / 20], [1 / 20, 1 / 40], [1 / 40, 1 / 80]]
        )
        self.assertArrayClose(actual, expected)

    def test_bias_homogeneous_augmentation(self):
        layer = KFACLinear(2, 1, bias=True)
        layer.weight = mx.zeros_like(layer.weight)
        layer.bias = mx.zeros_like(layer.bias)
        optimizer = KFAC(
            learning_rate=1.0,
            damping=1.0,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])

        def loss(inputs):
            return mx.sum(layer(inputs))

        _, grads = nn.value_and_grad(layer, loss)(x)
        optimizer.update(layer, grads)
        mx.eval(layer.parameters(), optimizer.state)

        augmented = mx.concatenate([x, mx.ones((2, 1))], axis=1)
        a = augmented.T @ augmented / 2 + mx.eye(3)
        # dL/dy is one for each sample, so G=1; damping makes it 2.
        d = mx.concatenate([grads["weight"], grads["bias"][:, None]], axis=1)
        expected = -mx.linalg.solve(
            a,
            mx.linalg.solve(mx.array([[2.0]]), d, stream=mx.cpu).T,
            stream=mx.cpu,
        ).T
        self.assertArrayClose(layer.weight, expected[:, :2])
        self.assertArrayClose(layer.bias, expected[:, 2])

    def test_factor_ema_updates(self):
        layer = KFACLinear(2, 1, bias=False)
        optimizer = KFAC(
            learning_rate=0.0,
            damping=0.1,
            factor_decay=0.75,
            inverse_update_interval=3,
        )
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])

        def loss(inputs):
            return mx.sum(layer(inputs))

        _, grads = nn.value_and_grad(layer, loss)(x)
        optimizer.update(layer, grads)
        state = optimizer.state["layers"]["__root__"]
        expected_a = 0.75 * mx.eye(2) + 0.25 * (x.T @ x / 2)
        expected_g = mx.ones((1, 1))
        self.assertArrayClose(state["A"], expected_a)
        self.assertArrayClose(state["G"], expected_g)
        self.assertEqual(state["A"].dtype, mx.float32)
        self.assertEqual(state["A_cholesky"].dtype, mx.float32)

    def test_unsupported_parameter_uses_adamw(self):
        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = KFACLinear(1, 1, bias=False)
                self.scale = mx.array(2.0)

            def __call__(self, x):
                return self.linear(x) * self.scale

        model = Mixed()
        optimizer = KFAC(learning_rate=0.01)

        def loss(x):
            return mx.sum(model(x))

        _, grads = nn.value_and_grad(model, loss)(mx.ones((2, 1)))
        old_scale = model.scale
        optimizer.update(model, grads)

        reference = optim.AdamW(0.01, weight_decay=0.0)
        expected = reference.apply_gradients({"scale": grads["scale"]}, {"scale": old_scale})
        self.assertArrayClose(model.scale, expected["scale"])

    def test_compile_preserves_parameters_and_optimizer_state(self):
        layer = KFACLinear(2, 1)
        optimizer = KFAC(
            learning_rate=0.01, damping=0.1, inverse_update_interval=2
        )

        def loss(x, y):
            return mx.mean((layer(x) - y) ** 2)

        value_and_grad = nn.value_and_grad(layer, loss)
        # Initialize state before it is captured by compile.
        x = mx.array([[1.0, -1.0], [2.0, 0.5]])
        y = mx.array([[0.2], [1.0]])
        _, grads = value_and_grad(x, y)
        optimizer.init(layer.trainable_parameters(), model=layer)

        def step(inputs, targets):
            value, step_grads = value_and_grad(inputs, targets)
            optimizer.update(layer, step_grads)
            return value

        captured_state = [layer.state, optimizer.state]
        step = mx.compile(step, inputs=captured_state, outputs=captured_state)
        before = layer.weight
        first = step(x, y)
        second = step(x, y)
        mx.eval(first, second, layer.parameters(), optimizer.state)
        self.assertFalse(bool(mx.allclose(before, layer.weight)))
        self.assertEqual(int(optimizer.state["step"]), 2)
        self.assertTrue(bool(mx.all(mx.isfinite(optimizer.state["layers"]["__root__"]["A"]))))

    def test_tiny_regression_loss_decreases(self):
        mx.random.seed(7)
        layer = KFACLinear(2, 1)
        optimizer = KFAC(
            learning_rate=0.03,
            damping=0.1,
            factor_decay=0.9,
            inverse_update_interval=2,
        )
        x = mx.random.normal((32, 2))
        y = x @ mx.array([[1.5], [-2.0]]) + 0.3

        def loss(inputs, targets):
            return mx.mean((layer(inputs) - targets) ** 2)

        value_and_grad = nn.value_and_grad(layer, loss)
        initial = loss(x, y)
        for _ in range(60):
            value, grads = value_and_grad(x, y)
            optimizer.update(layer, grads)
            mx.eval(layer.parameters(), optimizer.state)
        final = loss(x, y)
        self.assertLess(as_float(final), as_float(initial) * 0.2)


if __name__ == "__main__":
    unittest.main()
