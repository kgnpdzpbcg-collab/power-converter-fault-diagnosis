"""验证第二轮实验的关键控制条件：共同初值、等权起点和可学习门控。"""

import unittest
from unittest.mock import patch

import torch

from converter.data import make_loaders
from converter.models_v2 import FaultModelV2
from converter.training_trace import EpochTrace


class ControlledModelTests(unittest.TestCase):
    """不依据某个真实样本写规则，以合成输入检查结构行为及随机数控制。"""
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.batch = {"current": torch.randn(8, 3, 20), "dc": torch.randn(8, 2, 20), "rms": torch.randn(8, 3)}

    def test_paired_initialization_and_rng(self):
        reference, rng = None, None
        for fusion, norm, uniform in (("mean", False, False), ("mean", True, False),
                                      ("dynamic", False, False), ("dynamic", False, True),
                                      ("dynamic", True, False), ("dynamic", True, True)):
            torch.manual_seed(42)
            model = FaultModelV2(fusion, branch_norm=norm, uniform_gate_init=uniform)
            common = {f"{name}.{key}": value for name in model.branches
                      for key, value in (model.encoders[name][0] if norm else model.encoders[name]).state_dict().items()}
            common.update({f"head.{key}": value for key, value in model.classifier.state_dict().items()})
            if reference is None:
                reference, rng = common, torch.random.get_rng_state()
            else:
                self.assertTrue(all(torch.equal(value, reference[key]) for key, value in common.items()))
                self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_uniform_start_can_learn(self):
        model = FaultModelV2("dynamic", branch_norm=True, uniform_gate_init=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        logits, weights = model(self.batch)
        self.assertTrue(torch.equal(weights, torch.full_like(weights, 1 / 3)))
        torch.nn.functional.cross_entropy(logits, torch.arange(8)).backward()
        self.assertGreater(float(model.gate[-1].weight.grad.norm()), 0)
        # 末层为零时，首批梯度不能传到门控首层；更新末层后应恢复传播。
        self.assertEqual(float(model.gate[0].weight.grad.norm()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logits, weights = model(self.batch)
        torch.nn.functional.cross_entropy(logits, torch.arange(8)).backward()
        self.assertGreater(float(model.gate[0].weight.grad.norm()), 0)
        self.assertGreater(float(weights.std(dim=0).max().detach()), 0)
        for encoder in model.encoders.values():
            self.assertGreater(float(encoder[-1].weight.grad.norm()), 0)

    def test_trace_does_not_change_gradient_or_rng(self):
        model = FaultModelV2("dynamic")
        logits, weights = model(self.batch)
        torch.nn.functional.cross_entropy(logits, torch.arange(8)).backward()
        rng = torch.random.get_rng_state()
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        trace = EpochTrace(model.branches)
        trace.observe(model, weights)
        record = trace.record(1)
        self.assertEqual(record["samples"], 8)
        self.assertAlmostEqual(sum(record[f"{name}_weight_mean"] for name in model.branches), 1, places=6)
        self.assertTrue(all(torch.equal(parameter.grad, value) for parameter, value in zip(model.parameters(), gradients)))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_validation_only_loaders_never_create_test_view(self):
        # 拦截 Dataset 构造，直接验证集合选择，而不是检查输出目录名称。
        constructed = []
        def dataset_factory(prepared_dir, split, protocol, samples):
            constructed.append(split)
            return list(range(8))
        with patch("converter.data.ConverterDataset", side_effect=dataset_factory):
            loaders = make_loaders("unused", samples={}, splits=("train", "val"))
        self.assertEqual(constructed, ["train", "val"])
        self.assertEqual(list(loaders), ["train", "val"])


if __name__ == "__main__":
    unittest.main()
