"""Stochastic depth applies to both residual branches of each decoder layer."""
import importlib.util
from pathlib import Path
import unittest
import torch

# Load the complete leaf module, avoiding policy/environment package startup.
source = Path(__file__).resolve().parents[1] / "libero/lifelong/models/modules/transformer_modules.py"
spec = importlib.util.spec_from_file_location("libero_transformer_modules", source)
transformer_modules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transformer_modules)
TransformerDecoder = transformer_modules.TransformerDecoder


class AttentionDropPathTest(unittest.TestCase):
    def decoder(self):
        torch.manual_seed(4)
        return TransformerDecoder(
            input_size=8, num_layers=2, num_heads=2, head_output_size=4,
            mlp_hidden_size=12, dropout=0.2,
        )

    def test_training_drops_attention_and_feedforward_residuals(self):
        for mask_mode in ("none", "explicit", "cached"):
            with self.subTest(mask_mode=mask_mode):
                decoder = self.decoder().train()
                decoder.drop_path.drop_prob = 1.
                x = torch.arange(32., dtype=torch.float32).reshape(2, 2, 8) / 10
                mask = None
                if mask_mode == "explicit":
                    mask = torch.tril(torch.ones(1, 2, 2))
                elif mask_mode == "cached":
                    decoder.compute_mask((2, 2, 1))
                torch.testing.assert_close(decoder(x, mask), x, rtol=0, atol=0)

    def test_dropped_residuals_have_identity_input_gradient(self):
        decoder = self.decoder().train()
        decoder.drop_path.drop_prob = 1.
        x = torch.randn(2, 3, 8, requires_grad=True)
        weights = torch.randn_like(x)
        (decoder(x) * weights).sum().backward()
        torch.testing.assert_close(x.grad, weights, rtol=0, atol=0)
        for parameter in decoder.parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=0)

    def test_evaluation_keeps_both_residuals(self):
        decoder = self.decoder().eval()
        decoder.drop_path.drop_prob = 1.
        x = torch.randn(2, 3, 8)
        expected = x
        for att_norm, att, ff_norm, ff in decoder.layers:
            expected = expected + att(att_norm(expected))
            expected = expected + ff(ff_norm(expected))
        torch.testing.assert_close(decoder(x), expected)


if __name__ == "__main__":
    unittest.main()

