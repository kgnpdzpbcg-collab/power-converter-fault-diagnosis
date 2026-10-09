"""验证候选模型公平初值、输入边界、有效梯度与确定性，不按真实样本硬编码。"""

import unittest

import torch

from converter.models import FaultModel
from candidate.models import CandidateModel, VARIANTS


class CandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.batch = {"current": torch.randn(8, 3, 20), "dc": torch.randn(8, 2, 20), "rms": torch.randn(8, 3)}

    def make(self, variant):
        torch.manual_seed(42)
        return CandidateModel(variant=variant)

    def test_baseline_exact_old_model(self):
        torch.manual_seed(42)
        old = FaultModel("mean").eval(); rng = torch.random.get_rng_state()
        new = self.make("baseline").eval()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        self.assertTrue(all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items()))
        self.assertTrue(torch.equal(old(self.batch)[0], new(self.batch)[0]))

    def test_common_initialization_and_rng(self):
        base = self.make("baseline"); rng = torch.random.get_rng_state()
        for variant in VARIANTS[1:]:
            model = self.make(variant)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            for name in ("classifier",):
                self.assertTrue(all(torch.equal(v, getattr(model, name).state_dict()[k])
                                    for k, v in getattr(base, name).state_dict().items()))
            self.assertTrue(all(torch.equal(v, model.encoders['rms'].state_dict()[k])
                                for k, v in base.encoders['rms'].state_dict().items()))

    def test_shapes_counts_and_effective_gradients(self):
        counts = []
        for variant in VARIANTS:
            model = self.make(variant)
            logits, weights = model(self.batch)
            self.assertEqual(logits.shape, (8, 8))
            self.assertTrue(torch.equal(weights, torch.full_like(weights, 1/3)))
            torch.nn.functional.cross_entropy(logits, torch.arange(8)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            if variant == "multiscale":
                for name in ("path3", "path5", "project", "skip"):
                    self.assertGreater(float(sum(p.grad.square().sum() for p in getattr(model.encoders['current'], name).parameters())), 0)
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(counts, [24936, 27366, 27432])
        self.assertLess(abs(counts[1]/counts[2]-1), .01)

    def test_metadata_cannot_change_predictions(self):
        altered = {**self.batch, "label": torch.arange(8), "fine_label": torch.arange(8),
                   "sample_index": torch.arange(8), "timestamp": torch.arange(8), "block": torch.arange(8)}
        for variant in VARIANTS:
            model = self.make(variant).eval()
            self.assertTrue(torch.equal(model(self.batch)[0], model(altered)[0]))

    def test_repeated_optimizer_steps_exact(self):
        states = []
        for _ in range(2):
            model = self.make("multiscale")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(self.batch)[0], torch.arange(8))
                loss.backward(); optimizer.step()
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        self.assertTrue(all(torch.equal(v, states[1][k]) for k, v in states[0].items()))


if __name__ == "__main__":
    unittest.main()
