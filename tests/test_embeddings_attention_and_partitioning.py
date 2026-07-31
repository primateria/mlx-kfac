import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from mlx_kfac import KFAC
from mlx_kfac.optimizer import _state_key


class EmbeddingAttentionAndPartitioningTests(unittest.TestCase):
    def assert_close(self, actual, expected, atol=1e-5, rtol=1e-5):
        mx.eval(actual, expected)
        self.assertTrue(
            bool(mx.allclose(actual, expected, atol=atol, rtol=rtol)),
            f"\nactual={actual}\nexpected={expected}",
        )

    def test_embedding_diagonal_factor_and_explicit_kron_update(self):
        embedding = nn.Embedding(4, 2)
        embedding.weight = mx.zeros_like(embedding.weight)
        optimizer = KFAC(
            1.0,
            model=embedding,
            damping=1.0,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        token_ids = mx.array([0, 2, 2, 3])
        upstream = mx.array(
            [[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0], [0.5, 1.0]]
        )

        def loss(ids):
            return mx.sum(embedding(ids) * upstream)

        _, gradients = nn.value_and_grad(embedding, loss)(token_ids)
        optimizer.update(embedding, gradients)
        state = optimizer.state["layers"]["__root__"]
        expected_a = mx.array([0.25, 0.0, 0.5, 0.25])
        expected_g = upstream.T @ upstream / 4
        self.assert_close(state["A_diag"], expected_a)
        self.assert_close(state["G"], expected_g)
        self.assertNotIn("A", state)
        self.assertEqual(state["A_diag"].ndim, 1)

        a_damped = expected_a + 1.0
        g_damped = expected_g + mx.eye(2)
        expected_natural = mx.linalg.solve(
            g_damped, gradients["weight"].T, stream=mx.cpu
        ).T / a_damped[:, None]
        self.assert_close(embedding.weight, -expected_natural)

    def test_pi_damping_scalar_embedding_factor_is_exact(self):
        embedding = nn.Embedding(3, 1)
        embedding.weight = mx.zeros_like(embedding.weight)
        optimizer = KFAC(
            1.0,
            model=embedding,
            damping=1.0,
            damping_strategy="pi",
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        token_ids = mx.array([0, 1, 2])
        upstream = mx.array([[1.0], [2.0], [3.0]])

        _, gradients = nn.value_and_grad(
            embedding,
            lambda ids: mx.sum(embedding(ids) * upstream),
        )(token_ids)
        optimizer.update(embedding, gradients)

        a_diag = mx.full((3,), 1.0 / 3.0)
        g_scalar = mx.sum(upstream**2) / 3.0
        expected = gradients["weight"] / (
            a_diag[:, None] * g_scalar + 1.0
        )
        self.assert_close(embedding.weight, -expected)

    def test_pi_damping_scalar_embedding_token_factor_is_exact_when_partitioned(self):
        embedding = nn.Embedding(1, 4)
        embedding.weight = mx.zeros_like(embedding.weight)
        optimizer = KFAC(
            1.0,
            model=embedding,
            damping=0.75,
            damping_strategy="pi",
            factor_decay=0.0,
            inverse_update_interval=1,
            max_factor_size=2,
        )
        token_ids = mx.array([0, 0])
        upstream = mx.array(
            [[1.0, 2.0, -1.0, 0.5], [3.0, -1.0, 2.0, 4.0]]
        )

        _, gradients = nn.value_and_grad(
            embedding,
            lambda ids: mx.sum(embedding(ids) * upstream),
        )(token_ids)
        optimizer.update(embedding, gradients)

        expected_parts = []
        for block_slice in (slice(0, 2), slice(2, 4)):
            block_upstream = upstream[:, block_slice]
            block_g = block_upstream.T @ block_upstream / 2
            combined = block_g + 0.75 * mx.eye(2)
            expected_parts.append(
                mx.linalg.solve(
                    combined,
                    gradients["weight"][:, block_slice].T,
                    stream=mx.cpu,
                ).T
            )
        expected = mx.concatenate(expected_parts, axis=1)
        self.assert_close(embedding.weight, -expected)

    def test_embedding_mask_duplicate_counts_and_unseen_rows(self):
        embedding = nn.Embedding(5, 2)
        original = embedding.weight
        optimizer = KFAC(
            0.1,
            model=embedding,
            damping=0.5,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        ids = mx.array([[0, 2, 2, 4]])
        mask = mx.array([[1.0, 1.0, 0.0, 0.0]])
        _, gradients = nn.value_and_grad(
            embedding,
            lambda values: mx.sum(
                embedding(values) * mask[..., None]
            ),
        )(ids)
        optimizer.update(embedding, gradients, masks={"": mask})
        state = optimizer.state["layers"]["__root__"]
        self.assert_close(
            state["A_diag"], mx.array([0.5, 0.0, 0.5, 0.0, 0.0])
        )
        mx.eval(embedding.weight, original)
        self.assert_close(embedding.weight[1], original[1])
        self.assert_close(embedding.weight[3], original[3])
        self.assert_close(embedding.weight[4], original[4])

    def test_scalar_embedding_lookup(self):
        embedding = nn.Embedding(3, 2)
        optimizer = KFAC(
            0.0,
            model=embedding,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        token = mx.array(1)
        _, gradients = nn.value_and_grad(
            embedding, lambda value: mx.sum(embedding(value))
        )(token)
        optimizer.update(embedding, gradients)
        state = optimizer.state["layers"]["__root__"]
        self.assert_close(state["A_diag"], mx.array([0.0, 1.0, 0.0]))
        self.assert_close(state["curvature_scale"], mx.array(1.0))

    def test_disabled_embedding_reuse_uses_adamw_fallback(self):
        embedding = nn.Embedding(4, 2)
        # Instrument the shared module through an earlier optimizer first.
        instrumenting_optimizer = KFAC(0.0, model=embedding)
        instrumenting_optimizer.init(
            embedding.trainable_parameters(), model=embedding
        )
        self.assertIn("__root__", instrumenting_optimizer.state["layers"])

        optimizer = KFAC(
            0.01, model=embedding, include_embeddings=False
        )
        token_ids = mx.array([0, 2, 2])
        _, gradients = nn.value_and_grad(
            embedding, lambda values: mx.sum(embedding(values))
        )(token_ids)
        before = embedding.weight
        reference = optim.AdamW(0.01, weight_decay=0.0)
        expected = reference.apply_gradients(
            {"weight": gradients["weight"]}, {"weight": before}
        )

        optimizer.update(embedding, gradients)
        self.assertEqual(optimizer.state["layers"], {})
        self.assert_close(embedding.weight, expected["weight"])

    def test_tied_embedding_projection_fails_explicitly(self):
        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(4, 2)

            def __call__(self, ids):
                values = self.embedding(ids)
                return self.embedding.as_linear(values)

        model = Tied()
        optimizer = KFAC(0.1, model=model)
        ids = mx.array([0, 1])
        _, gradients = nn.value_and_grad(
            model, lambda values: mx.sum(model(values))
        )(ids)
        with self.assertRaisesRegex(ValueError, "dedicated solver"):
            optimizer.update(model, gradients)

    def test_aliased_supported_module_fails_at_registration(self):
        class Aliased(nn.Module):
            def __init__(self):
                super().__init__()
                projection = nn.Linear(2, 2)
                self.first = projection
                self.second = projection

        with self.assertRaisesRegex(ValueError, "Aliased"):
            KFAC(model=Aliased())

    def test_shared_bias_parameter_fails_at_registration(self):
        class SharedBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(2, 2)
                self.second = nn.Linear(2, 2)
                self.second.bias = self.first.bias

        with self.assertRaisesRegex(ValueError, "Tied or aliased"):
            KFAC(model=SharedBias())

    def test_attention_registers_four_independent_projection_blocks(self):
        attention = nn.MultiHeadAttention(4, 2)
        optimizer = KFAC(
            0.0,
            model=attention,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        x = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4) / 10
        _, gradients = nn.value_and_grad(
            attention,
            lambda values: mx.sum(attention(values, values, values)),
        )(x)
        optimizer.update(attention, gradients)
        layers = optimizer._find_layers(attention)
        self.assertEqual(
            set(layers),
            {"query_proj", "key_proj", "value_proj", "out_proj"},
        )
        self.assertEqual(len(optimizer.state["layers"]), 4)
        a_factors = [
            optimizer.state["layers"][_state_key(name)]["A"]
            for name in ("query_proj", "key_proj", "value_proj")
        ]
        self.assert_close(a_factors[0], a_factors[1])
        self.assert_close(a_factors[1], a_factors[2])

    def test_cross_attention_factor_shapes(self):
        attention = nn.MultiHeadAttention(
            4,
            2,
            query_input_dims=3,
            key_input_dims=5,
            value_input_dims=6,
        )
        optimizer = KFAC(0.0, model=attention)
        q = mx.ones((2, 2, 3))
        k = mx.ones((2, 3, 5))
        v = mx.ones((2, 3, 6))

        def loss(queries, keys, values):
            return mx.sum(attention(queries, keys, values))

        _, gradients = nn.value_and_grad(attention, loss)(q, k, v)
        optimizer.update(attention, gradients)
        states = optimizer.state["layers"]
        self.assertEqual(states[_state_key("query_proj")]["A"].shape, (3, 3))
        self.assertEqual(states[_state_key("key_proj")]["A"].shape, (5, 5))
        self.assertEqual(states[_state_key("value_proj")]["A"].shape, (6, 6))

    def test_attention_head_aware_partitions(self):
        attention = nn.MultiHeadAttention(8, 4)
        optimizer = KFAC(
            0.0,
            model=attention,
            attention_head_blocks=True,
            inverse_update_interval=1,
        )
        x = mx.ones((1, 3, 8))
        _, gradients = nn.value_and_grad(
            attention,
            lambda values: mx.sum(attention(values, values, values)),
        )(x)
        optimizer.update(attention, gradients)
        states = optimizer.state["layers"]
        for name in ("query_proj", "key_proj", "value_proj"):
            state = states[_state_key(name)]
            self.assertEqual([x.shape[0] for x in state["G_blocks"]], [2] * 4)
        out_state = states[_state_key("out_proj")]
        self.assertEqual([x.shape[0] for x in out_state["A_blocks"]], [2] * 4)

    def test_attention_max_blocks_preserve_head_and_bias_boundaries(self):
        attention = nn.MultiHeadAttention(8, 2, bias=True)
        optimizer = KFAC(
            0.0,
            model=attention,
            attention_head_blocks=True,
            max_factor_size=3,
            inverse_update_interval=1,
        )
        x = mx.ones((1, 2, 8))
        _, gradients = nn.value_and_grad(
            attention,
            lambda values: mx.sum(attention(values, values, values)),
        )(x)
        optimizer.update(attention, gradients)
        states = optimizer.state["layers"]
        for name in ("query_proj", "key_proj", "value_proj"):
            self.assertEqual(
                [block.shape[0] for block in states[_state_key(name)]["G_blocks"]],
                [3, 1, 3, 1],
            )
        self.assertEqual(
            [
                block.shape[0]
                for block in states[_state_key("out_proj")]["A_blocks"]
            ],
            [3, 1, 3, 1, 1],
        )

    def test_attention_padding_factor_masks(self):
        attention = nn.MultiHeadAttention(4, 2)
        optimizer = KFAC(
            0.0,
            model=attention,
            factor_decay=0.0,
            inverse_update_interval=1,
        )
        x = mx.array(
            [[[1.0, 2.0, 3.0, 4.0], [50.0, 50.0, 50.0, 50.0]]]
        )
        valid = mx.array([[1.0, 0.0]])

        def loss(values):
            return mx.sum(attention(values, values, values))

        _, gradients = nn.value_and_grad(attention, loss)(x)
        masks = {
            "query_proj": valid,
            "key_proj": valid,
            "value_proj": valid,
            "out_proj": valid,
        }
        optimizer.update(attention, gradients, masks=masks)
        expected = x[:, :1, :].reshape(1, 4)
        query_a = optimizer.state["layers"][
            _state_key("query_proj")
        ]["A"]
        self.assert_close(query_a, expected.T @ expected)

    def test_compiled_attention_recompiles_for_sequence_length(self):
        attention = nn.MultiHeadAttention(4, 2)
        optimizer = KFAC(0.0, model=attention, inverse_update_interval=1)
        optimizer.init(attention.trainable_parameters(), model=attention)

        def loss(values):
            return mx.sum(attention(values, values, values))

        value_and_grad = nn.value_and_grad(attention, loss)

        def step(values):
            objective, gradients = value_and_grad(values)
            optimizer.update(attention, gradients)
            return objective

        captured = [attention.state, optimizer.state]
        step = mx.compile(step, inputs=captured, outputs=captured)
        step(mx.ones((1, 3, 4)))
        step(mx.ones((1, 2, 4)))
        mx.eval(attention.parameters(), optimizer.state)
        self.assertEqual(int(optimizer.state["step"]), 2)
        for state in optimizer.state["layers"].values():
            self.assertTrue(bool(mx.all(mx.isfinite(state["velocity"]))))

    def test_block_partition_matches_explicit_block_diagonal_solve(self):
        layer = nn.Linear(5, 6, bias=True)
        layer.weight = mx.zeros_like(layer.weight)
        layer.bias = mx.zeros_like(layer.bias)
        optimizer = KFAC(
            1.0,
            model=layer,
            damping=1.0,
            factor_decay=0.0,
            inverse_update_interval=1,
            max_factor_size=3,
        )
        x = mx.array(
            [[1, 2, 3, 4, 5], [2, 0, 1, -1, 3]], dtype=mx.float32
        )
        upstream = mx.arange(1, 13, dtype=mx.float32).reshape(2, 6) / 4

        def loss(inputs):
            return mx.sum(layer(inputs) * upstream)

        _, gradients = nn.value_and_grad(layer, loss)(x)
        optimizer.update(layer, gradients)
        state = optimizer.state["layers"]["__root__"]
        self.assertTrue(all(block.shape[0] <= 3 for block in state["A_blocks"]))
        self.assertTrue(all(block.shape[0] <= 3 for block in state["G_blocks"]))

        augmented = mx.concatenate([x, mx.ones((2, 1))], axis=1)
        grad = mx.concatenate(
            [gradients["weight"], gradients["bias"][:, None]], axis=1
        )
        expected_rows = []
        for out_slice in (slice(0, 3), slice(3, 6)):
            expected_columns = []
            g = (
                upstream[:, out_slice].T @ upstream[:, out_slice] / 2
                + mx.eye(3)
            )
            for in_slice in (slice(0, 3), slice(3, 6)):
                a = (
                    augmented[:, in_slice].T @ augmented[:, in_slice] / 2
                    + mx.eye(3)
                )
                left = mx.linalg.solve(
                    g, grad[out_slice, in_slice], stream=mx.cpu
                )
                expected_columns.append(
                    mx.linalg.solve(a, left.T, stream=mx.cpu).T
                )
            expected_rows.append(mx.concatenate(expected_columns, axis=1))
        expected = mx.concatenate(expected_rows, axis=0)
        self.assert_close(layer.weight, -expected[:, :5], atol=2e-5)
        self.assert_close(layer.bias, -expected[:, 5], atol=2e-5)

    def test_pi_damping_partitioned_whole_scalar_factor_is_exact(self):
        cases = (
            (
                nn.Linear(1, 4, bias=False),
                mx.array([[1.0], [2.0]]),
                mx.array(
                    [[1.0, 2.0, -1.0, 0.5], [3.0, -1.0, 2.0, 4.0]]
                ),
                "input",
            ),
            (
                nn.Linear(4, 1, bias=False),
                mx.array(
                    [[1.0, 2.0, -1.0, 0.5], [3.0, -1.0, 2.0, 4.0]]
                ),
                mx.array([[1.0], [2.0]]),
                "output",
            ),
        )
        for layer, x, upstream, scalar_side in cases:
            with self.subTest(scalar_side=scalar_side):
                layer.weight = mx.zeros_like(layer.weight)
                optimizer = KFAC(
                    1.0,
                    model=layer,
                    damping=0.75,
                    damping_strategy="pi",
                    factor_decay=0.0,
                    inverse_update_interval=1,
                    max_factor_size=2,
                )
                _, gradients = nn.value_and_grad(
                    layer,
                    lambda values: mx.sum(layer(values) * upstream),
                )(x)
                optimizer.update(layer, gradients)

                if scalar_side == "input":
                    scalar = mx.sum(x**2) / x.shape[0]
                    expected_parts = []
                    for block_slice in (slice(0, 2), slice(2, 4)):
                        block_upstream = upstream[:, block_slice]
                        block_g = (
                            block_upstream.T
                            @ block_upstream
                            / upstream.shape[0]
                        )
                        combined = scalar * block_g + 0.75 * mx.eye(2)
                        expected_parts.append(
                            mx.linalg.solve(
                                combined,
                                gradients["weight"][block_slice],
                                stream=mx.cpu,
                            )
                        )
                    expected = mx.concatenate(expected_parts, axis=0)
                else:
                    scalar = mx.sum(upstream**2) / upstream.shape[0]
                    expected_parts = []
                    for block_slice in (slice(0, 2), slice(2, 4)):
                        block_x = x[:, block_slice]
                        block_a = block_x.T @ block_x / x.shape[0]
                        combined = scalar * block_a + 0.75 * mx.eye(2)
                        expected_parts.append(
                            mx.linalg.solve(
                                combined,
                                gradients["weight"][:, block_slice].T,
                                stream=mx.cpu,
                            ).T
                        )
                    expected = mx.concatenate(expected_parts, axis=1)
                self.assert_close(layer.weight, -expected, atol=2e-5)

    def test_large_partition_limit_matches_unpartitioned(self):
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        first = nn.Linear(2, 2)
        second = nn.Linear(2, 2)
        second.update(first.parameters())
        optimizers = [
            KFAC(
                0.1, model=first, factor_decay=0.0,
                inverse_update_interval=1,
            ),
            KFAC(
                0.1, model=second, factor_decay=0.0,
                inverse_update_interval=1, max_factor_size=100,
            ),
        ]
        for layer, optimizer in zip((first, second), optimizers):
            _, gradients = nn.value_and_grad(
                layer, lambda z, layer=layer: mx.sum(layer(z))
            )(x)
            optimizer.update(layer, gradients)
        self.assert_close(first.weight, second.weight)
        self.assert_close(first.bias, second.bias)

    def test_partitioned_embedding_bounds_dense_factors_and_compiles(self):
        embedding = nn.Embedding(7, 5)
        optimizer = KFAC(
            0.01,
            model=embedding,
            max_factor_size=2,
            inverse_update_interval=1,
        )
        optimizer.init(embedding.trainable_parameters(), model=embedding)
        ids = mx.array([[0, 1, 2], [2, 3, 4]])

        def loss(values):
            return mx.mean(embedding(values) ** 2)

        value_and_grad = nn.value_and_grad(embedding, loss)

        def step(values):
            objective, gradients = value_and_grad(values)
            optimizer.update(embedding, gradients)
            return objective

        captured = [embedding.state, optimizer.state]
        step = mx.compile(step, inputs=captured, outputs=captured)
        initial = loss(ids)
        for _ in range(5):
            step(ids)
        final = loss(ids)
        mx.eval(embedding.parameters(), optimizer.state)
        state = optimizer.state["layers"]["__root__"]
        self.assertEqual(state["A_diag"].shape, (7,))
        self.assertTrue(all(block.shape[0] <= 2 for block in state["G_blocks"]))
        self.assertEqual(int(optimizer.state["step"]), 5)
        self.assertLess(float(final), float(initial))

    def test_partitioned_checkpoint_continuation(self):
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        first = nn.Linear(4, 4, bias=True)
        first_optimizer = KFAC(
            0.01,
            model=first,
            max_factor_size=2,
            momentum=0.5,
            inverse_update_interval=1,
        )

        def take_step(layer, optimizer):
            _, gradients = nn.value_and_grad(
                layer, lambda values: mx.sum(layer(values) ** 2)
            )(x)
            optimizer.update(layer, gradients)
            mx.eval(layer.parameters(), optimizer.state)

        take_step(first, first_optimizer)
        second = nn.Linear(4, 4, bias=True)
        second.update(first.parameters())
        second_optimizer = KFAC(
            0.01,
            model=second,
            max_factor_size=2,
            momentum=0.5,
            inverse_update_interval=1,
        )
        second_optimizer.load_state(
            tree_unflatten(list(tree_flatten(first_optimizer.state))), second
        )
        take_step(first, first_optimizer)
        take_step(second, second_optimizer)
        self.assert_close(first.weight, second.weight)
        self.assert_close(first.bias, second.bias)

    def test_embedding_state_cannot_load_into_shape_compatible_linear(self):
        embedding = nn.Embedding(3, 2)
        embedding_optimizer = KFAC(0.01, model=embedding)
        embedding_optimizer.init(
            embedding.trainable_parameters(), model=embedding
        )
        linear = nn.Linear(3, 2, bias=False)
        linear_optimizer = KFAC(0.01, model=linear)
        with self.assertRaisesRegex(ValueError, "schema"):
            linear_optimizer.load_state(embedding_optimizer.state, linear)

    def test_unequal_count_aggregator_matches_concatenated_statistics(self):
        local_x = mx.array([[1.0, 2.0]])
        remote_x = mx.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        local_g = mx.array([[2.0]])
        remote_g = mx.array([[1.0], [3.0], [4.0]])

        def aggregate(numerator, count, kind, path):
            if kind == "A":
                return numerator + remote_x.T @ remote_x, count + 3
            if kind == "G":
                return numerator + remote_g.T @ remote_g, count + 3
            return numerator, count

        layer = nn.Linear(2, 1, bias=False)
        optimizer = KFAC(
            0.0,
            model=layer,
            factor_decay=0.0,
            inverse_update_interval=1,
            factor_aggregator=aggregate,
        )

        def loss(values):
            return mx.sum(layer(values) * local_g)

        _, gradients = nn.value_and_grad(layer, loss)(local_x)
        optimizer.update(layer, gradients)
        state = optimizer.state["layers"]["__root__"]
        all_x = mx.concatenate([local_x, remote_x])
        all_g = mx.concatenate([local_g, remote_g])
        self.assert_close(state["A"], all_x.T @ all_x / 4)
        self.assert_close(state["G"], all_g.T @ all_g / 4)

    def test_compiled_causal_attention_and_fallback_training(self):
        class TinyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiHeadAttention(4, 2)
                self.norm = nn.LayerNorm(4)
                self.output = nn.Linear(4, 1)

            def __call__(self, x):
                mask = nn.MultiHeadAttention.create_additive_causal_mask(
                    x.shape[1]
                )
                return self.output(
                    self.norm(x + self.attention(x, x, x, mask))
                )

        mx.random.seed(11)
        model = TinyTransformer()
        optimizer = KFAC(
            0.02,
            model=model,
            damping=0.1,
            inverse_update_interval=2,
            max_factor_size=4,
        )
        x = mx.random.normal((8, 3, 4))
        target = mx.sum(x, axis=-1, keepdims=True)

        def loss(inputs, targets):
            return mx.mean((model(inputs) - targets) ** 2)

        value_and_grad = nn.value_and_grad(model, loss)
        optimizer.init(model.trainable_parameters(), model=model)

        def step(inputs, targets):
            objective, gradients = value_and_grad(inputs, targets)
            optimizer.update(model, gradients)
            return objective

        captured = [model.state, optimizer.state]
        step = mx.compile(step, inputs=captured, outputs=captured)
        initial = loss(x, target)
        for _ in range(25):
            step(x, target)
        final = loss(x, target)
        mx.eval(initial, final, model.parameters(), optimizer.state)
        self.assertLess(float(final), float(initial) * 0.7)
        fallback_paths = [path for path, _ in tree_flatten(optimizer.state["fallback"])]
        self.assertTrue(any("norm" in path for path in fallback_paths))


if __name__ == "__main__":
    unittest.main()
