"""All declared GMM hidden depths produce valid action distributions."""
import math
import unittest

import torch

from libero.lifelong.models import policy_head

class GMMFeatureWidthTest(unittest.TestCase):
    def head(self, depth):
        torch.manual_seed(11)
        head = policy_head.GMMHead(input_size=5, output_size=2, hidden_size=7,
                                   num_layers=depth, num_modes=3).double()
        with torch.no_grad():
            for module in head.modules():
                if isinstance(module, torch.nn.Linear):
                    module.weight.fill_(.1)
                    module.bias.copy_(torch.linspace(-.2, .3, module.out_features, dtype=torch.float64))
        return head

    def density(self, distribution, target):
        # Independent scalar mixture density, rather than MixtureSameFamily's
        # log_prob or the production loss implementation.
        weights = distribution.mixture_distribution.probs.detach().reshape(-1, 3)
        component = distribution.component_distribution.base_dist
        means = component.loc.detach().reshape(-1, 3, 2)
        scales = component.scale.detach().reshape(-1, 3, 2)
        rows = target.detach().reshape(-1, 2)
        values = []
        for row, row_weights, row_means, row_scales in zip(rows, weights, means, scales):
            probability = 0.
            for weight, mean, scale in zip(row_weights, row_means, row_scales):
                component_density = 1.
                for value, mu, sigma in zip(row, mean, scale):
                    z = (float(value) - float(mu)) / float(sigma)
                    component_density *= math.exp(-.5 * z * z) / (math.sqrt(2 * math.pi) * float(sigma))
                probability += float(weight) * component_density
            values.append(math.log(probability))
        return torch.tensor(values, dtype=torch.float64).reshape(target.shape[:-1])

    def test_zero_and_one_hidden_layer_accept_distinct_input_hidden_widths(self):
        for depth in (0, 1, 2):
            with self.subTest(depth=depth):
                head = self.head(depth)
                dist = head(torch.ones(3, 5, dtype=torch.float64))
                self.assertEqual(tuple(dist.batch_shape), (3,))
                self.assertEqual(tuple(dist.event_shape), (2,))
                target = torch.tensor([[.2, -.3], [.1, .2], [-.1, .4]], dtype=torch.float64)
                torch.testing.assert_close(dist.log_prob(target), self.density(dist, target), rtol=1e-12, atol=1e-12)

    def test_temporal_inputs_preserve_batch_and_action_dimensions(self):
        for depth in (0, 1, 2):
            with self.subTest(depth=depth):
                head = self.head(depth)
                x = torch.ones(2, 4, 5, dtype=torch.float64)
                dist = head(x)
                self.assertEqual(tuple(dist.batch_shape), (2, 4))
                self.assertEqual(tuple(dist.event_shape), (2,))
                target = torch.full((2, 4, 2), .2, dtype=torch.float64)
                torch.testing.assert_close(dist.log_prob(target), self.density(dist, target), rtol=1e-12, atol=1e-12)

    def test_all_loss_reductions_use_the_same_mixture_likelihood(self):
        for depth in (0, 1):
            with self.subTest(depth=depth):
                head = self.head(depth)
                head.loss_coef = 1.7
                dist = head(torch.ones(3, 5, dtype=torch.float64))
                target = torch.tensor([[.2, -.3], [.1, .2], [-.1, .4]], dtype=torch.float64)
                negative_logs = -self.density(dist, target) * 1.7
                torch.testing.assert_close(head.loss_fn(dist, target, 'none'), negative_logs, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(head.loss_fn(dist, target, 'mean'), negative_logs.mean(), rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(head.loss_fn(dist, target, 'sum'), negative_logs.sum(), rtol=1e-12, atol=1e-12)

    def test_input_gradient_matches_independent_density_difference(self):
        for depth in (0, 1):
            with self.subTest(depth=depth):
                head = self.head(depth)
                x = torch.ones(1, 5, dtype=torch.float64, requires_grad=True)
                target = torch.tensor([[.2, -.3]], dtype=torch.float64)
                head.loss_fn(head(x), target).backward()
                expected = torch.empty_like(x)
                step = 1e-6
                for index in range(5):
                    plus = x.detach().clone()
                    minus = x.detach().clone()
                    plus[0, index] += step
                    minus[0, index] -= step
                    positive_loss = -self.density(head(plus), target).mean()
                    negative_loss = -self.density(head(minus), target).mean()
                    expected[0, index] = (positive_loss - negative_loss) / (2 * step)
                torch.testing.assert_close(x.grad, expected, rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
