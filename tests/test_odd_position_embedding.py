"""Positional embeddings retain the exact configurable policy embedding width."""
import math
import unittest

import torch

from libero.lifelong.models.modules import transformer_modules as transformer

class OddPositionTest(unittest.TestCase):
    def oracle(self, width, length, factor=1.):
        channels = 2 * math.ceil(width / 2)
        frequencies = [10 ** (-index / channels) for index in range(0, channels, 2)]
        rows = []
        for position in range(length):
            values = [math.sin(position * freq) for freq in frequencies]
            values += [math.cos(position * freq) for freq in frequencies]
            rows.append([value * factor for value in values[:width]])
        return torch.tensor(rows, dtype=torch.float32)

    def test_odd_width_matches_public_output_contract_and_analytic_basis(self):
        for width in (1, 3, 7, 15):
            with self.subTest(width=width):
                module = transformer.SinusoidalPositionEncoding(width)
                x = torch.zeros(2, 5, 3, width)
                actual = module(x)
                self.assertEqual(tuple(actual.shape), (5, width))
                self.assertEqual(module.output_size(width), actual.shape[-1])
                torch.testing.assert_close(actual, self.oracle(width, 5), rtol=1e-6, atol=1e-6)

    def test_even_width_basis_is_preserved(self):
        for width in (2, 8, 16):
            with self.subTest(width=width):
                module = transformer.SinusoidalPositionEncoding(width)
                torch.testing.assert_close(module(torch.zeros(2, 5, width)), self.oracle(width, 5), rtol=1e-6, atol=1e-6)

    def test_position_addition_and_real_decoder_backpropagate_at_odd_width(self):
        width = 7
        module = transformer.SinusoidalPositionEncoding(width)
        decoder = transformer.TransformerDecoder(
            input_size=width, num_layers=2, num_heads=2, head_output_size=4,
            mlp_hidden_size=11, dropout=0.)
        x = torch.randn(2, 4, 3, width, requires_grad=True)
        augmented = x + module(x).unsqueeze(1)
        decoder.compute_mask(x.shape)
        actual = decoder(augmented.reshape(2, 12, width))
        self.assertEqual(tuple(actual.shape), (2, 12, width))
        self.assertTrue(torch.isfinite(actual).all().item())
        actual.square().mean().backward()
        self.assertEqual(tuple(x.grad.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreater(x.grad.abs().sum().item(), 0.)

    def test_learnable_factor_has_analytic_gradient(self):
        width = 7
        module = transformer.SinusoidalPositionEncoding(width, factor_ratio=1.2)
        actual = module(torch.zeros(2, 5, width))
        torch.testing.assert_close(actual, self.oracle(width, 5, 1.2), rtol=1e-6, atol=1e-6)
        actual.sum().backward()
        torch.testing.assert_close(module.factor.grad, self.oracle(width, 5).sum().reshape(1), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
