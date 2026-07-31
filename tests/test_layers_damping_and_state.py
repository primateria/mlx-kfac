import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from mlx_kfac import KFAC
import mlx_kfac.optimizer as optimizer_module


class LayerDampingAndStateTests(unittest.TestCase):
    def assert_close(self, actual, expected, atol=1e-5, rtol=1e-5):
        mx.eval(actual, expected)
        self.assertTrue(
            bool(mx.allclose(actual, expected, atol=atol, rtol=rtol)),
            f"\nactual={actual}\nexpected={expected}",
        )

    def test_plain_linear_is_registered_in_place_and_idempotently(self):
        layer = nn.Linear(2, 3)
        weight = layer.weight
        x = mx.array([[1.0, 2.0]])
        before = layer(x)
        optimizer = KFAC(0.0, model=layer)
        optimizer.register(layer)
        after = layer(x)
        self.assertIs(layer.weight, weight)
        self.assertIsInstance(layer, nn.Linear)
        self.assert_close(before, after)

        _, grads = nn.value_and_grad(layer, lambda z: mx.sum(layer(z)))(x)
        optimizer.update(layer, grads)
        self.assertIn("__root__", optimizer.state["layers"])

    def test_unbatched_linear_has_unit_repetition_scale(self):
        layer = nn.Linear(2, 1, bias=False)
        layer.weight = mx.zeros_like(layer.weight)
        optimizer = KFAC(
            1.0,
            model=layer,
            damping=1.0,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        x = mx.array([1.0, 2.0])
        _, gradients = nn.value_and_grad(
            layer, lambda values: 3.0 * mx.sum(layer(values))
        )(x)
        optimizer.update(layer, gradients)
        state = optimizer.state["layers"]["__root__"]
        self.assert_close(state["curvature_scale"], mx.array(1.0))
        a = x[:, None] @ x[None, :] + mx.eye(2)
        expected = -mx.linalg.solve(
            a,
            mx.linalg.solve(
                mx.array([[10.0]]), gradients["weight"], stream=mx.cpu
            ).T,
            stream=mx.cpu,
        ).T
        self.assert_close(layer.weight, expected)

    def test_nested_paths_round_trip_through_mlx_tree_utilities(self):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(2, 3), nn.Linear(3, 1)]

            def __call__(self, x):
                return self.layers[1](mx.maximum(self.layers[0](x), 0))

        model = MLP()
        optimizer = KFAC(0.0, model=model)
        x = mx.ones((2, 2))
        _, grads = nn.value_and_grad(
            model, lambda z: mx.sum(model(z))
        )(x)
        optimizer.update(model, grads)
        flat = tree_flatten(optimizer.state)
        restored = tree_unflatten(flat)
        self.assertEqual(
            [path for path, _ in flat],
            [path for path, _ in tree_flatten(restored)],
        )
        self.assertEqual(len(optimizer.state["layers"]), 2)
        self.assertFalse(any(".0" in key for key in optimizer.state["layers"]))

    def test_conv2d_patch_factor_and_bias_augmentation(self):
        layer = nn.Conv2d(1, 1, 2, bias=True)
        optimizer = KFAC(
            0.0, model=layer, factor_decay=0.0, inverse_update_interval=1
        )
        x = mx.arange(1, 10, dtype=mx.float32).reshape(1, 3, 3, 1)
        _, grads = nn.value_and_grad(
            layer, lambda z: mx.sum(layer(z))
        )(x)
        optimizer.update(layer, grads)
        state = optimizer.state["layers"]["__root__"]
        patches = mx.array(
            [
                [1, 2, 4, 5, 1],
                [2, 3, 5, 6, 1],
                [4, 5, 7, 8, 1],
                [5, 6, 8, 9, 1],
            ],
            dtype=mx.float32,
        )
        self.assert_close(state["A"], patches.T @ patches / 4)
        self.assert_close(state["G"], mx.ones((1, 1)))

    def test_conv1d_patch_factor(self):
        layer = nn.Conv1d(1, 1, 2, bias=False)
        optimizer = KFAC(
            0.0, model=layer, factor_decay=0.0, inverse_update_interval=1
        )
        x = mx.array([[[1.0], [2.0], [3.0], [4.0]]])
        _, grads = nn.value_and_grad(
            layer, lambda z: mx.sum(layer(z))
        )(x)
        optimizer.update(layer, grads)
        patches = mx.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        state = optimizer.state["layers"]["__root__"]
        self.assert_close(state["A"], patches.T @ patches / 3)
        self.assert_close(state["curvature_scale"], mx.array(3.0))

    def test_conv2d_stride_padding_dilation_patch_order(self):
        layer = nn.Conv2d(
            2, 1, (2, 2), stride=(2, 1), padding=(1, 1), dilation=(2, 1),
            bias=False,
        )
        optimizer = KFAC(
            0.0, model=layer, factor_decay=0.0, inverse_update_interval=1
        )
        x = mx.arange(1, 1 + 1 * 4 * 3 * 2, dtype=mx.float32).reshape(
            1, 4, 3, 2
        )
        _, grads = nn.value_and_grad(
            layer, lambda z: mx.sum(layer(z))
        )(x)
        optimizer.update(layer, grads)

        padded = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        rows = []
        out = layer(x)
        for oh in range(out.shape[1]):
            for ow in range(out.shape[2]):
                row = []
                for kh in range(2):
                    for kw in range(2):
                        row.extend(
                            padded[0, oh * 2 + kh * 2, ow + kw, :].tolist()
                        )
                rows.append(row)
        patches = mx.array(rows)
        state = optimizer.state["layers"]["__root__"]
        self.assert_close(state["A"], patches.T @ patches / patches.shape[0])

    def test_pi_adjusted_damping(self):
        layer = nn.Linear(2, 2, bias=False)
        optimizer = KFAC(
            0.0,
            model=layer,
            damping=0.16,
            damping_strategy="pi",
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        x = mx.array([[1.0, 0.0], [0.0, 3.0]])
        upstream = mx.array([[2.0, 0.0], [0.0, 1.0]])

        def loss(inputs):
            return mx.sum(layer(inputs) * upstream)

        _, grads = nn.value_and_grad(layer, loss)(x)
        optimizer.update(layer, grads)
        state = optimizer.state["layers"]["__root__"]
        mean_a = mx.trace(state["A"]) / 2
        mean_g = mx.trace(state["G"]) / 2
        pi = mx.sqrt(mean_a / mean_g)
        self.assert_close(state["damping_A"], 0.4 * pi)
        self.assert_close(state["damping_G"], 0.4 / pi)

    def test_mean_loss_scaling_is_batch_duplication_invariant(self):
        factors = []
        for batch_size in (1, 4):
            layer = nn.Linear(1, 1, bias=False)
            layer.weight = mx.zeros_like(layer.weight)
            optimizer = KFAC(
                0.0,
                model=layer,
                factor_decay=0.0,
                inverse_update_interval=1,
                loss_reduction="mean",
            )
            x = mx.ones((batch_size, 1))
            _, gradients = nn.value_and_grad(
                layer, lambda values: mx.mean(layer(values))
            )(x)
            optimizer.update(layer, gradients)
            factors.append(optimizer.state["layers"]["__root__"]["G"])
        self.assert_close(factors[0], factors[1])

    def test_repeated_pi_damping_includes_curvature_scale(self):
        layer = nn.Linear(2, 2, bias=False)
        layer.weight = mx.zeros_like(layer.weight)
        optimizer = KFAC(
            1.0,
            model=layer,
            damping=1.0,
            damping_strategy="pi",
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        x = mx.array([[[1.0, 0.0], [0.0, 2.0]]])
        upstream = mx.array([[[1.0, 0.0], [0.0, 3.0]]])
        _, gradients = nn.value_and_grad(
            layer,
            lambda values: mx.sum(layer(values) * upstream),
        )(x)
        optimizer.update(layer, gradients)
        state = optimizer.state["layers"]["__root__"]
        self.assert_close(state["A"], mx.diag(mx.array([0.5, 2.0])))
        self.assert_close(state["G"], mx.diag(mx.array([0.5, 4.5])))
        self.assert_close(state["curvature_scale"], mx.array(2.0))
        self.assert_close(
            layer.weight,
            -mx.diag(mx.array([1.0 / 3.0, 6.0 / 27.5])),
        )

    def test_pi_damping_scalar_factor_uses_exact_damped_solve(self):
        layer = nn.Linear(1, 2, bias=False)
        layer.weight = mx.zeros_like(layer.weight)
        optimizer = KFAC(
            1.0,
            model=layer,
            damping=0.5,
            damping_strategy="pi",
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        x = mx.array([[2.0]])
        upstream = mx.array([[1.0, 3.0]])
        _, gradients = nn.value_and_grad(
            layer,
            lambda values: mx.sum(layer(values) * upstream),
        )(x)
        optimizer.update(layer, gradients)
        # A is scalar 4, so solve exactly with 4*G + .5*I.
        g = upstream.T @ upstream
        expected = -mx.linalg.solve(
            4.0 * g + 0.5 * mx.eye(2),
            gradients["weight"],
            stream=mx.cpu,
        )
        self.assert_close(layer.weight, expected)

    def test_pi_damping_balances_a_zero_factor(self):
        layer = nn.Linear(2, 2, bias=False)
        optimizer = KFAC(
            0.0,
            model=layer,
            damping=1.0,
            damping_strategy="pi",
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        x = mx.zeros((2, 2))
        upstream = mx.array([[1.0, 0.0], [0.0, 2.0]])
        _, gradients = nn.value_and_grad(
            layer,
            lambda values: mx.sum(layer(values) * upstream),
        )(x)
        optimizer.update(layer, gradients)
        state = optimizer.state["layers"]["__root__"]
        self.assert_close(state["A"], mx.zeros((2, 2)))
        self.assert_close(state["damping_A"], mx.array(1.0))
        self.assert_close(state["damping_G"], mx.array(1.0))

    def test_cholesky_and_eigh_updates_match(self):
        x = mx.array([[1.0, -2.0], [3.0, 0.5]])
        upstream = mx.array([[0.4, -1.0], [2.0, 0.7]])
        first = nn.Linear(2, 2, bias=True)
        second = nn.Linear(2, 2, bias=True)
        second.update(first.parameters())
        optimizers = [
            KFAC(
                0.1, model=first, damping=0.2, factor_decay=0.0,
                inverse_update_interval=1, decomposition="cholesky",
            ),
            KFAC(
                0.1, model=second, damping=0.2, factor_decay=0.0,
                inverse_update_interval=1, decomposition="eigh",
            ),
        ]
        for layer, optimizer in zip((first, second), optimizers):
            _, grads = nn.value_and_grad(
                layer, lambda z, layer=layer: mx.sum(layer(z) * upstream)
            )(x)
            optimizer.update(layer, grads)
            mx.eval(layer.parameters(), optimizer.state)
        self.assert_close(first.weight, second.weight, atol=2e-5)
        self.assert_close(first.bias, second.bias, atol=2e-5)

    def test_momentum_recurrence_and_global_kl_clip(self):
        layer = nn.Linear(1, 1, bias=False)
        layer.weight = mx.zeros_like(layer.weight)
        optimizer = KFAC(
            0.5,
            model=layer,
            damping=1.0,
            factor_decay=0.0,
            inverse_update_interval=1,
            momentum=0.5,
            kl_clip=0.01,
        )
        x = mx.ones((1, 1))
        velocities = []
        scales = []
        for _ in range(2):
            _, grads = nn.value_and_grad(
                layer, lambda z: mx.sum(layer(z))
            )(x)
            optimizer.update(layer, grads)
            state = optimizer.state["layers"]["__root__"]
            mx.eval(state, optimizer.state["last_kl_scale"])
            velocities.append(state["velocity"])
            scales.append(optimizer.state["last_kl_scale"])
        # Natural gradient is 1 / ((1+1)*(1+1)) = .25.
        expected_scale = min(1.0, (0.01 / (0.5**2 * 0.25)) ** 0.5)
        self.assert_close(scales[0], mx.array(expected_scale))
        self.assert_close(velocities[0], mx.array([[expected_scale * 0.25]]))
        self.assert_close(
            velocities[1], 0.5 * velocities[0] + expected_scale * 0.25
        )

    def test_update_mode_kl_clip_bounds_momentum_direction(self):
        layer = nn.Linear(1, 1, bias=False)
        layer.weight = mx.zeros_like(layer.weight)
        optimizer = KFAC(
            0.5,
            model=layer,
            damping=1.0,
            factor_decay=0.0,
            inverse_update_interval=1,
            momentum=0.5,
            kl_clip=0.01,
            kl_clip_mode="update",
        )
        x = mx.ones((1, 1))
        for _ in range(2):
            before = layer.weight
            _, gradients = nn.value_and_grad(
                layer, lambda values: mx.sum(layer(values))
            )(x)
            optimizer.update(layer, gradients)
            delta = layer.weight - before
            # Damped scalar curvature is (A+1)(G+1)=4.
            metric = mx.sum(delta * (4.0 * delta))
            mx.eval(metric)
            self.assertLessEqual(float(metric), 0.010001)

    def test_sequence_mask_and_reused_layer_calls(self):
        class Reused(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(2, 1, bias=False)

            def __call__(self, x, y):
                return self.proj(x), self.proj(y)

        model = Reused()
        optimizer = KFAC(
            0.0, model=model, factor_decay=0.0, inverse_update_interval=1
        )
        x = mx.array([[[1.0, 2.0], [9.0, 9.0]]])
        y = mx.array([[[3.0, 4.0], [5.0, 6.0]]])
        mask_x = mx.array([[1.0, 0.0]])
        mask_y = mx.array([[1.0, 1.0]])

        def loss(a, b):
            first, second = model(a, b)
            return mx.sum(first) + mx.sum(second)

        _, grads = nn.value_and_grad(model, loss)(x, y)
        optimizer.update(
            model, grads, masks={"proj": [mask_x, mask_y]}
        )
        state = next(iter(optimizer.state["layers"].values()))
        samples = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assert_close(state["A"], samples.T @ samples / 3)
        self.assert_close(state["G"], mx.ones((1, 1)))
        self.assert_close(state["curvature_scale"], mx.array(3.0))

    def test_repetition_scale_uses_factor_decay(self):
        layer = nn.Linear(1, 1, bias=False)
        optimizer = KFAC(
            0.0,
            model=layer,
            factor_decay=0.5,
            inverse_update_interval=1,
        )
        for repeats in (3, 1):
            x = mx.ones((1, repeats, 1))
            _, gradients = nn.value_and_grad(
                layer, lambda values: mx.sum(layer(values))
            )(x)
            optimizer.update(layer, gradients)
        state = optimizer.state["layers"]["__root__"]
        # Initial scale 1 -> 2 after repeats=3 -> 1.5 after repeats=1.
        self.assert_close(state["curvature_scale"], mx.array(1.5))

    def test_all_zero_mask_preserves_factors(self):
        layer = nn.Linear(2, 1, bias=False)
        optimizer = KFAC(
            0.0, model=layer, factor_decay=0.0, inverse_update_interval=1
        )
        optimizer.init(layer.trainable_parameters(), model=layer)
        before = optimizer.state["layers"]["__root__"]["A"]
        x = mx.ones((1, 3, 2))
        _, grads = nn.value_and_grad(
            layer, lambda z: mx.sum(layer(z))
        )(x)
        optimizer.update(layer, grads, masks={"": mx.zeros((1, 3))})
        self.assert_close(
            optimizer.state["layers"]["__root__"]["A"], before
        )

    def test_frozen_bias_and_grouped_conv_fallback(self):
        linear = nn.Linear(2, 1, bias=True)
        linear.freeze(keys="bias", recurse=False)
        optimizer = KFAC(0.0, model=linear)
        x = mx.ones((2, 2))
        _, grads = nn.value_and_grad(
            linear, lambda z: mx.sum(linear(z))
        )(x)
        optimizer.update(linear, grads)
        self.assertEqual(
            optimizer.state["layers"]["__root__"]["A"].shape, (2, 2)
        )

        conv = nn.Conv2d(2, 2, 1, groups=2, bias=False)
        before = conv.weight
        fallback = KFAC(0.01, model=conv, weight_decay=0.0)
        inp = mx.ones((1, 2, 2, 2))
        _, conv_grads = nn.value_and_grad(
            conv, lambda z: mx.sum(conv(z))
        )(inp)
        fallback.update(conv, conv_grads)
        reference = optim.AdamW(0.01, weight_decay=0.0)
        expected = reference.apply_gradients(
            conv_grads, {"weight": before}
        )
        self.assert_close(conv.weight, expected["weight"])

    def test_alternating_conditional_layers_keep_static_ownership(self):
        class Conditional(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(2, 1)
                self.second = nn.Linear(2, 1)

            def __call__(self, x, use_first):
                return self.first(x) if use_first else self.second(x)

        model = Conditional()
        optimizer = KFAC(0.01, model=model, inverse_update_interval=1)
        x = mx.ones((2, 2))
        for use_first in (True, False, True):
            _, gradients = nn.value_and_grad(
                model,
                lambda values, flag=use_first: mx.sum(model(values, flag)),
            )(x)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state)
        self.assertEqual(int(optimizer.state["step"]), 3)

    def test_factor_aggregator_runtime_error_is_not_swallowed(self):
        layer = nn.Linear(2, 1, bias=False)

        def fail(*args):
            raise RuntimeError("aggregation failed")

        optimizer = KFAC(
            0.01,
            model=layer,
            factor_aggregator=fail,
        )
        x = mx.ones((2, 2))
        _, gradients = nn.value_and_grad(
            layer, lambda values: mx.sum(layer(values))
        )(x)
        with self.assertRaisesRegex(RuntimeError, "aggregation failed"):
            optimizer.update(layer, gradients)

    def test_checkpoint_restore_and_incompatible_shape(self):
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        first = nn.Linear(2, 1)
        first_optimizer = KFAC(
            0.01, model=first, momentum=0.5, inverse_update_interval=1
        )

        def take_step(layer, optimizer):
            _, grads = nn.value_and_grad(
                layer, lambda z: mx.mean(layer(z) ** 2)
            )(x)
            optimizer.update(layer, grads)
            mx.eval(layer.parameters(), optimizer.state)

        take_step(first, first_optimizer)
        params = tree_unflatten(list(tree_flatten(first.parameters())))
        state = tree_unflatten(list(tree_flatten(first_optimizer.state)))

        second = nn.Linear(2, 1)
        second.update(params)
        second_optimizer = KFAC(
            0.01, model=second, momentum=0.5, inverse_update_interval=1
        )
        second_optimizer.load_state(state, second)
        take_step(first, first_optimizer)
        take_step(second, second_optimizer)
        self.assert_close(first.weight, second.weight)
        self.assert_close(first.bias, second.bias)

        wrong = nn.Linear(3, 1)
        wrong_optimizer = KFAC(0.01, model=wrong)
        with self.assertRaises(ValueError):
            wrong_optimizer.load_state(state, wrong)

    def test_direct_state_assignment_continues_without_reinitializing(self):
        x = mx.array([[1.0, 2.0]])
        first = nn.Linear(2, 1)
        first_optimizer = KFAC(
            0.01, model=first, momentum=0.5, inverse_update_interval=1
        )

        def take_step(layer, optimizer):
            _, gradients = nn.value_and_grad(
                layer, lambda z: mx.sum(layer(z) ** 2)
            )(x)
            optimizer.update(layer, gradients)
            mx.eval(layer.parameters(), optimizer.state)

        take_step(first, first_optimizer)
        second = nn.Linear(2, 1)
        second.update(first.parameters())
        second_optimizer = KFAC(
            0.01, model=second, momentum=0.5, inverse_update_interval=1
        )
        second_optimizer.state = tree_unflatten(
            list(tree_flatten(first_optimizer.state))
        )
        take_step(first, first_optimizer)
        take_step(second, second_optimizer)
        self.assert_close(first.weight, second.weight)
        self.assert_close(first.bias, second.bias)

    def test_direct_state_assignment_continues_fallback_only_model(self):
        x = mx.array([[1.0, 3.0], [2.0, -1.0]])
        target = mx.array([[2.0, -1.0], [0.5, 3.0]])
        first = nn.LayerNorm(2)
        first_optimizer = KFAC(0.01, model=first)

        def take_step(layer, optimizer):
            _, gradients = nn.value_and_grad(
                layer,
                lambda values: mx.sum(layer(values) * target),
            )(x)
            optimizer.update(layer, gradients)
            mx.eval(layer.parameters(), optimizer.state)

        take_step(first, first_optimizer)
        second = nn.LayerNorm(2)
        second.update(
            tree_unflatten(list(tree_flatten(first.parameters())))
        )
        second_optimizer = KFAC(0.01, model=second)
        # MLX tree round-tripping drops the legitimately empty ``layers``
        # container, so this exercises the ambiguous restore edge directly.
        second_optimizer.state = tree_unflatten(
            list(tree_flatten(first_optimizer.state))
        )
        take_step(first, first_optimizer)
        take_step(second, second_optimizer)
        self.assert_close(first.weight, second.weight)
        self.assert_close(first.bias, second.bias)
        self.assertEqual(
            int(first_optimizer.state["step"]),
            int(second_optimizer.state["step"]),
        )

    def test_load_state_rejects_schema_mismatch_atomically(self):
        layer = nn.Linear(2, 1)
        source = KFAC(
            0.01, model=layer, decomposition="cholesky"
        )
        source.init(layer.trainable_parameters(), model=layer)

        target_layer = nn.Linear(2, 1)
        target = KFAC(
            0.01, model=target_layer, decomposition="eigh"
        )
        before = target.state
        with self.assertRaisesRegex(ValueError, "schema"):
            target.load_state(source.state, target_layer)
        self.assertIs(target.state, before)

    def test_load_state_validates_root_schema_shape_and_dtype_atomically(self):
        layer = nn.Linear(2, 1)
        source = KFAC(0.01, model=layer)
        source.init(layer.trainable_parameters(), model=layer)

        missing = dict(source.state)
        missing.pop("force_refresh")
        wrong_shape = dict(source.state)
        wrong_shape["step"] = mx.zeros((1,), dtype=mx.uint64)
        wrong_dtype = dict(source.state)
        wrong_dtype["damping"] = source.state["damping"].astype(mx.float16)

        target_layer = nn.Linear(2, 1)
        target = KFAC(0.01, model=target_layer)
        before = target.state
        for name, invalid in (
            ("missing", missing),
            ("shape", wrong_shape),
            ("dtype", wrong_dtype),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "root state"):
                    target.load_state(invalid, target_layer)
                self.assertIs(target.state, before)

    def test_load_state_validates_fallback_root_leaf_dtype_atomically(self):
        source_layer = nn.LayerNorm(2)
        source = KFAC(0.01, model=source_layer)
        source.init(source_layer.trainable_parameters(), model=source_layer)
        invalid = tree_unflatten(list(tree_flatten(source.state)))
        invalid["fallback"]["step"] = mx.array(0.0, mx.float32)

        target_layer = nn.LayerNorm(2)
        target = KFAC(0.01, model=target_layer)
        before = target.state
        with self.assertRaisesRegex(ValueError, "fallback state leaf"):
            target.load_state(invalid, target_layer)
        self.assertIs(target.state, before)

    def test_load_state_rejects_nonfinite_values_and_damping_outside_bounds(self):
        source_layer = nn.Linear(2, 1)
        source = KFAC(0.01, model=source_layer)
        source.init(source_layer.trainable_parameters(), model=source_layer)
        invalid_factor = tree_unflatten(list(tree_flatten(source.state)))
        invalid_factor["layers"]["__root__"]["A"] = (
            mx.ones_like(invalid_factor["layers"]["__root__"]["A"])
            * float("inf")
        )
        invalid_damping = tree_unflatten(list(tree_flatten(source.state)))
        invalid_damping["damping"] = mx.array(float("nan"), mx.float32)
        out_of_bounds = tree_unflatten(list(tree_flatten(source.state)))
        out_of_bounds["damping"] = mx.array(1e-9, mx.float32)

        target_layer = nn.Linear(2, 1)
        target = KFAC(0.01, model=target_layer, min_damping=1e-8)
        before = target.state
        for name, invalid, message in (
            ("factor", invalid_factor, "non-finite"),
            ("damping", invalid_damping, "non-finite"),
            ("bounds", out_of_bounds, "bounds"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    target.load_state(invalid, target_layer)
                self.assertIs(target.state, before)

    def test_direct_state_assignment_validates_against_constructor_schema(self):
        source_layer = nn.Linear(2, 1)
        source = KFAC(0.01, model=source_layer)
        source.init(source_layer.trainable_parameters(), model=source_layer)
        invalid = dict(source.state)
        invalid.pop("last_kl_scale")

        target_layer = nn.Linear(2, 1)
        target = KFAC(0.01, model=target_layer)
        target.state = invalid
        x = mx.ones((1, 2))
        _, gradients = nn.value_and_grad(
            target_layer, lambda values: mx.sum(target_layer(values))
        )(x)
        with self.assertRaisesRegex(ValueError, "root state"):
            target.update(target_layer, gradients)
        self.assertIs(target.state, invalid)

    def test_load_state_copies_python_containers(self):
        first = nn.Linear(2, 1)
        first_optimizer = KFAC(0.01, model=first)
        first_optimizer.init(first.trainable_parameters(), model=first)
        second = nn.Linear(2, 1)
        second_optimizer = KFAC(0.01, model=second)
        second_optimizer.load_state(first_optimizer.state, second)
        self.assertIsNot(first_optimizer.state, second_optimizer.state)
        second_optimizer.state["step"] = mx.array(7, mx.uint64)
        self.assertEqual(int(first_optimizer.state["step"]), 0)

    def test_fully_frozen_supported_layer_checkpoint(self):
        class FrozenWithScale(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)
                self.linear.freeze()
                self.scale = mx.array(1.0)

            def __call__(self, x):
                return self.linear(x) * self.scale

        first = FrozenWithScale()
        optimizer = KFAC(0.01, model=first)
        optimizer.init(first.trainable_parameters(), model=first)
        state = tree_unflatten(list(tree_flatten(optimizer.state)))
        second = FrozenWithScale()
        restored = KFAC(0.01, model=second)
        restored.load_state(state, second)
        self.assertEqual(restored.state["layers"], {})

    def test_compiled_bfloat16_plain_linear_keeps_float32_state(self):
        layer = nn.Linear(2, 1)
        layer.apply(lambda value: value.astype(mx.bfloat16))
        optimizer = KFAC(
            0.01, model=layer, inverse_update_interval=2
        )
        x = mx.array([[1.0, 2.0], [2.0, -1.0]], mx.bfloat16)
        y = mx.array([[1.0], [-1.0]], mx.bfloat16)

        def loss(inputs, targets):
            residual = layer(inputs) - targets
            return mx.mean(residual.astype(mx.float32) ** 2)

        value_and_grad = nn.value_and_grad(layer, loss)
        optimizer.init(layer.trainable_parameters(), model=layer)

        def step(inputs, targets):
            value, grads = value_and_grad(inputs, targets)
            optimizer.update(layer, grads)
            return value

        captured = [layer.state, optimizer.state]
        step = mx.compile(step, inputs=captured, outputs=captured)
        step(x, y)
        step(x, y)
        mx.eval(layer.parameters(), optimizer.state)
        state = optimizer.state["layers"]["__root__"]
        self.assertEqual(layer.weight.dtype, mx.bfloat16)
        for key in ("A", "G", "A_cholesky", "G_cholesky", "velocity"):
            self.assertEqual(state[key].dtype, mx.float32)

    def test_mixed_precision_fallback_moments_are_float32(self):
        class WithNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.norm = nn.LayerNorm(2)

            def __call__(self, x):
                return self.norm(self.linear(x))

        model = WithNorm()
        model.apply(lambda value: value.astype(mx.bfloat16))
        optimizer = KFAC(0.01, model=model)
        x = mx.ones((2, 2), mx.bfloat16)
        _, gradients = nn.value_and_grad(
            model,
            lambda values: mx.sum(model(values).astype(mx.float32) ** 2),
        )(x)
        optimizer.update(model, gradients)
        for path, value in tree_flatten(optimizer.state["fallback"]):
            if path.endswith(".m") or path.endswith(".v"):
                self.assertEqual(value.dtype, mx.float32)

    def test_adaptive_damping_thresholds(self):
        optimizer = KFAC(
            damping=1.0, adaptive_damping=True,
            damping_adaptation_decay=0.5,
            damping_adaptation_interval=1,
        )
        self.assert_close(optimizer.adapt_damping(0.1), mx.array(2.0))
        self.assert_close(optimizer.adapt_damping(0.5), mx.array(2.0))
        self.assert_close(optimizer.adapt_damping(0.9), mx.array(1.0))

    def test_adaptive_damping_honors_interval(self):
        optimizer = KFAC(
            damping=1.0,
            adaptive_damping=True,
            damping_adaptation_decay=0.5,
            damping_adaptation_interval=2,
        )
        self.assert_close(optimizer.adapt_damping(0.1), mx.array(1.0))
        self.assert_close(optimizer.adapt_damping(0.1), mx.array(2.0))

    def test_adaptive_damping_forces_refresh_in_compiled_step(self):
        layer = nn.Linear(2, 2, bias=False)
        optimizer = KFAC(
            0.0,
            model=layer,
            damping=1.0,
            adaptive_damping=True,
            damping_adaptation_decay=0.5,
            damping_adaptation_interval=1,
            inverse_update_interval=100,
        )
        optimizer.init(layer.trainable_parameters(), model=layer)
        x = mx.eye(2)
        value_and_grad = nn.value_and_grad(
            layer, lambda values: mx.sum(layer(values))
        )

        def step(values):
            objective, gradients = value_and_grad(values)
            optimizer.update(layer, gradients)
            return objective

        captured = [layer.state, optimizer.state]
        step = mx.compile(step, inputs=captured, outputs=captured)
        step(x)
        before = optimizer.state["layers"]["__root__"]["A_cholesky"]
        optimizer.adapt_damping(0.1)
        step(x)
        after = optimizer.state["layers"]["__root__"]["A_cholesky"]
        mx.eval(before, after)
        self.assertFalse(bool(mx.allclose(before, after)))

    def test_nonrefresh_step_uses_one_consistent_cached_curvature(self):
        layer = nn.Linear(2, 1, bias=False)
        layer.weight = mx.zeros_like(layer.weight)
        optimizer = KFAC(
            0.1,
            model=layer,
            damping=0.5,
            factor_decay=0.0,
            inverse_update_interval=100,
            kl_clip_mode="update",
        )

        first_x = mx.array([[[1.0, 0.0], [0.0, 2.0]]])
        first_upstream = mx.array([[[1.0], [3.0]]])
        _, first_gradients = nn.value_and_grad(
            layer,
            lambda values: mx.sum(layer(values) * first_upstream),
        )(first_x)
        optimizer.update(layer, first_gradients, refresh=True)
        state = optimizer.state["layers"]["__root__"]
        cached_a = state["A_cholesky"]
        cached_g = state["G_cholesky"]
        cached_scale = state["cached_curvature_scale"]
        before = layer.weight

        second_x = mx.array(
            [[[4.0, 1.0], [2.0, 3.0], [1.0, -2.0], [5.0, 2.0]]]
        )
        second_upstream = mx.array([[[2.0], [1.0], [4.0], [-1.0]]])
        _, second_gradients = nn.value_and_grad(
            layer,
            lambda values: mx.sum(layer(values) * second_upstream),
        )(second_x)
        expected_natural = optimizer_module.precondition_linear_gradient(
            second_gradients["weight"].astype(mx.float32),
            cached_g,
            cached_a,
        ) / cached_scale
        optimizer.update(layer, second_gradients, refresh=False)

        self.assert_close(state["curvature_scale"], mx.array(4.0))
        self.assert_close(state["cached_curvature_scale"], cached_scale)
        self.assert_close(layer.weight, before - 0.1 * expected_natural)

        direction = mx.array([[0.25, -0.75]])
        cached_a_matrix = cached_a @ cached_a.T
        cached_g_matrix = cached_g @ cached_g.T
        expected_product = (
            cached_scale
            * cached_g_matrix
            @ direction
            @ cached_a_matrix
        )
        self.assert_close(
            optimizer._curvature_product(state, direction),
            expected_product,
        )

    def test_embedding_nonrefresh_uses_cached_diagonal_factor(self):
        embedding = nn.Embedding(3, 2)
        embedding.weight = mx.zeros_like(embedding.weight)
        optimizer = KFAC(
            0.1,
            model=embedding,
            damping=1.0,
            factor_decay=0.0,
            inverse_update_interval=100,
        )

        first_ids = mx.array([0, 0])
        first_upstream = mx.array([[1.0, 0.0], [0.0, 1.0]])
        _, first_gradients = nn.value_and_grad(
            embedding,
            lambda values: mx.sum(embedding(values) * first_upstream),
        )(first_ids)
        optimizer.update(embedding, first_gradients, refresh=True)
        state = optimizer.state["layers"]["__root__"]
        cached_a = state["cached_A_diag"]
        cached_g = state["G_cholesky"]
        cached_scale = state["cached_curvature_scale"]
        before = embedding.weight

        second_ids = mx.array([1, 1])
        second_upstream = mx.array([[2.0, 1.0], [1.0, 3.0]])
        _, second_gradients = nn.value_and_grad(
            embedding,
            lambda values: mx.sum(embedding(values) * second_upstream),
        )(second_ids)
        right = optimizer_module._solve_cholesky(
            cached_g, second_gradients["weight"].T
        ).T
        expected_natural = (
            right
            / (cached_a[:, None] + state["damping_A"])
            / cached_scale
        )
        optimizer.update(embedding, second_gradients, refresh=False)

        self.assert_close(state["A_diag"], mx.array([0.0, 1.0, 0.0]))
        self.assert_close(state["cached_A_diag"], cached_a)
        self.assert_close(embedding.weight, before - 0.1 * expected_natural)

    def test_inactive_conditional_layer_honors_forced_damping_refresh(self):
        class Conditional(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(2, 1, bias=False)
                self.second = nn.Linear(2, 1, bias=False)

            def __call__(self, x, use_first):
                return self.first(x) if use_first else self.second(x)

        model = Conditional()
        optimizer = KFAC(
            0.0,
            model=model,
            damping=1.0,
            factor_decay=0.0,
            inverse_update_interval=100,
            adaptive_damping=True,
            damping_adaptation_decay=0.5,
            damping_adaptation_interval=1,
        )
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])

        _, gradients = nn.value_and_grad(
            model, lambda values: mx.sum(model(values, False))
        )(x)
        optimizer.update(model, gradients)
        second_state = optimizer.state["layers"][
            optimizer_module._state_key("second")
        ]
        before = second_state["A_cholesky"]

        optimizer.adapt_damping(0.1)
        _, gradients = nn.value_and_grad(
            model, lambda values: mx.sum(model(values, True))
        )(x)
        optimizer.update(model, gradients)
        after = second_state["A_cholesky"]

        self.assert_close(second_state["damping_A"], mx.array(2.0))
        self.assert_close(second_state["damping_G"], mx.array(2.0))
        expected = mx.linalg.cholesky(
            second_state["A"] + 2.0 * mx.eye(2),
            stream=mx.cpu,
        )
        self.assert_close(after, expected)
        mx.eval(before, after)
        self.assertFalse(bool(mx.allclose(before, after)))

    def test_static_no_refresh_skips_decomposition(self):
        layer = nn.Linear(2, 1)
        optimizer = KFAC(0.0, model=layer, inverse_update_interval=3)
        optimizer.init(layer.trainable_parameters(), model=layer)
        x = mx.ones((2, 2))
        _, grads = nn.value_and_grad(
            layer, lambda z: mx.sum(layer(z))
        )(x)
        with mock.patch.object(
            optimizer_module,
            "_cholesky",
            side_effect=AssertionError("decomposition should be skipped"),
        ):
            optimizer.update(layer, grads, refresh=False)

    def test_eager_interval_skips_nonrefresh_decompositions(self):
        layer = nn.Linear(2, 1)
        optimizer = KFAC(0.0, model=layer, inverse_update_interval=3)
        optimizer.init(layer.trainable_parameters(), model=layer)
        x = mx.ones((2, 2))
        real_cholesky = optimizer_module._cholesky
        with mock.patch.object(
            optimizer_module, "_cholesky", wraps=real_cholesky
        ) as cholesky:
            _, gradients = nn.value_and_grad(
                layer, lambda values: mx.sum(layer(values))
            )(x)
            optimizer.update(layer, gradients)
            self.assertEqual(cholesky.call_count, 2)
            for _ in range(2):
                _, gradients = nn.value_and_grad(
                    layer, lambda values: mx.sum(layer(values))
                )(x)
                optimizer.update(layer, gradients)
            self.assertEqual(cholesky.call_count, 2)
            _, gradients = nn.value_and_grad(
                layer, lambda values: mx.sum(layer(values))
            )(x)
            optimizer.update(layer, gradients)
            self.assertEqual(cholesky.call_count, 4)

    def test_source_has_no_forbidden_framework_or_inverse(self):
        for path in (
            "src/mlx_kfac/capture.py",
            "src/mlx_kfac/linear.py",
            "src/mlx_kfac/optimizer.py",
        ):
            source = open(path, encoding="utf-8").read()
            self.assertNotIn("import jax", source)
            self.assertNotIn("import torch", source)
            self.assertNotIn("linalg.inv", source)


if __name__ == "__main__":
    unittest.main()
