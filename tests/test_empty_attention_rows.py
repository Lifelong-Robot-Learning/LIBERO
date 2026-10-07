import unittest

import torch

from libero.lifelong.models.modules import transformer_modules as modules

class EmptyAttentionRowsTest(unittest.TestCase):
    def attention(self):
        module = modules.Attention(4, num_heads=2, head_output_size=2, dropout=0.)
        with torch.no_grad():
            # Q=K=0 and V=x gives a uniform average over allowed values.
            module.qkv.weight.zero_()
            module.qkv.weight[8:12].copy_(torch.eye(4))
            module.output_layer[0].weight.copy_(torch.eye(4))
            module.output_layer[0].bias.zero_()
        return module

    def check_mask(self, mask, allowed):
        module = self.attention()
        x = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 7.).requires_grad_()
        output = module(x, mask)
        expected = torch.zeros_like(x)
        expected_gradient = torch.zeros_like(x)
        expected_weights = torch.zeros(2, 2, 3, 3)
        for batch in range(2):
            for row in range(3):
                keys = [key for key in range(3) if allowed[batch, row, key]]
                if keys:
                    expected[batch, row] = sum(x.detach()[batch, key] for key in keys) / len(keys)
                    for key in keys:
                        expected_gradient[batch, key] += 1. / len(keys)
                        expected_weights[batch, :, row, key] = 1. / len(keys)
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(module.att_weights, expected_weights, rtol=1e-6, atol=1e-7)
        self.assertTrue(torch.isfinite(output).all().item())
        output.sum().backward()
        torch.testing.assert_close(x.grad, expected_gradient, rtol=1e-6, atol=1e-7)
        for parameter in module.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_fully_padded_batch_element_has_zero_values_weights_and_gradient(self):
        mask = torch.tensor([[False, False, False], [True, False, True]])
        self.check_mask(mask, mask[:, None, :].expand(2, 3, 3))

    def test_per_batch_causal_masks_support_fully_hidden_query_rows(self):
        mask = torch.tensor([[[False, False, False], [True, False, False], [True, True, False]],
                             [[True, True, True], [False, False, False], [False, True, False]]])
        self.check_mask(mask, mask)

    def test_shared_causal_mask_handles_empty_rows_without_contaminating_valid_rows(self):
        mask = torch.tensor([[[False, False, False], [True, False, False], [True, True, True]]])
        self.check_mask(mask, mask.expand(2, 3, 3))

    def test_unmasked_and_nonempty_mask_controls_preserve_uniform_value_attention(self):
        allowed = torch.ones(2, 3, 3, dtype=torch.bool)
        self.check_mask(None, allowed)
        self.check_mask(torch.ones(2, 3, dtype=torch.bool), allowed)


if __name__ == '__main__':
    unittest.main()
