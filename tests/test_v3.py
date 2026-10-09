"""验证 V3 原始路径、注意力行为、参数预算和初值配对，不依赖真实样本。"""

import unittest

import torch

from converter.models import FaultModel
from converter.models_v3 import FaultModelV3, VARIANTS


class AttentionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.batch = {"current": torch.randn(8, 3, 20), "dc": torch.randn(8, 2, 20), "rms": torch.randn(8, 3)}

    def make_model(self, variant):
        torch.manual_seed(42)
        return FaultModelV3(variant)

    def test_b0_exact_old_concat(self):
        torch.manual_seed(42)
        old = FaultModel("concat")
        old_rng = torch.random.get_rng_state()
        new = self.make_model("b0_concat")
        self.assertTrue(torch.equal(old_rng, torch.random.get_rng_state()))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        self.assertTrue(all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items()))
        old.eval(); new.eval()
        self.assertTrue(torch.equal(old(self.batch)[0], new(self.batch)[0]))

    def test_paired_modules_and_rng(self):
        models = {name: self.make_model(name) for name in VARIANTS}
        rng = torch.random.get_rng_state()
        reference = models["b0_concat"].encoders.state_dict()
        for model in models.values():
            self.assertTrue(all(torch.equal(value, model.encoders.state_dict()[key]) for key, value in reference.items()))
            self.make_model(model.fusion)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for module in ("attention", "interaction_norm", "interaction_mlp"):
            left = getattr(models["b2_attention"], module).state_dict()
            right = getattr(models["b3_attention_skip"], module).state_dict()
            self.assertTrue(all(torch.equal(value, right[key]) for key, value in left.items()))
        left = models["b1_mlp_skip"].classifier.state_dict()
        right = models["b3_attention_skip"].classifier.state_dict()
        self.assertTrue(all(torch.equal(value, right[key]) for key, value in left.items()))

    def test_raw_skip_and_attention(self):
        model = self.make_model("b3_attention_skip").eval()
        with torch.no_grad():
            raw = torch.cat([model.encoders[name](self.batch[name]) for name in model.branches], dim=1)
            fused, weights = model.fused_features(self.batch, collect_attention=True)
            self.assertEqual(fused.shape, (8, 256))
            self.assertTrue(torch.equal(raw, fused[:, :192]))
            self.assertEqual(weights.shape, (8, 4, 3, 3))
            self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(8, 4, 3), atol=1e-6))
            self.assertTrue((weights.diagonal(dim1=-2, dim2=-1) > 0).all())
            self.assertGreater(float(weights.std(dim=0).max()), 0)
            logits, old_weights = model(self.batch)
            self.assertIsNone(old_weights)
            self.assertTrue(torch.equal(logits, model.classifier(fused)))
            normalized = model.interaction_norm(raw.reshape(8, 3, 64))
            attended, _ = model.attention(normalized, normalized, normalized,
                                          need_weights=True, average_attn_weights=False)
            self.assertTrue(torch.equal(fused[:, 192:], model.interaction_mlp(attended.flatten(1))))

    def test_shapes_parameter_budget_and_gradients(self):
        counts = {}
        for variant, width in zip(VARIANTS, (192, 256, 64, 256)):
            model = self.make_model(variant)
            fused, _ = model.fused_features(self.batch)
            self.assertEqual(fused.shape, (8, width))
            logits, _ = model(self.batch)
            torch.nn.functional.cross_entropy(logits, torch.arange(8)).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.norm()), 0, name)
            counts[variant] = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(list(counts.values()), [41320, 78488, 54056, 78632])
        self.assertLess(abs(counts["b1_mlp_skip"] / counts["b3_attention_skip"] - 1), 0.01)


if __name__ == "__main__":
    unittest.main()
