"""验证容易导致数据泄漏或错误结论的边界，使用合成数据而非真实样本硬编码。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from converter.data import normalize
from converter.metrics import classification_metrics
from converter.models import FaultModel
from converter.prepare import FAMILY_MAP, episode_blocks, fit_normalization, iter_records, nearest, split_blocks


class DataTests(unittest.TestCase):
    """流式解析、事件恢复绑定、标签映射与训练集统计的独立行为约束。"""
    def test_stream_across_chunks_and_reject_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            records = [{"nested": {"中文": [1, 2, 3]}, "text": "}, ["}, {"a": 4}]
            path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(list(iter_records(path, chunk_size=7)), records)
            for invalid in ('[{"a": 1}', '[{"a":1},]', '[{}] x', '[{} {}]'):
                path.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(iter_records(path, chunk_size=5))

    def test_nearest_only_time_and_earlier_tie(self):
        anchor = np.array([-1.0, 1.0, 2.1, 9.0])
        self.assertEqual(nearest(anchor, np.array([0.0, 2.0, 4.0])).tolist(), [0, 0, 1, 2])

    def test_blocks_bind_recovery_before_coarse_mapping(self):
        labels = np.array([0, 0, 1, 1, 0, 0, 2, 2, 3, 0])
        block, starts = episode_blocks(labels)
        self.assertEqual(starts.tolist(), [2, 6, 8])
        self.assertEqual(block.tolist(), [0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
        # 1、2、3 属于同一粗类，仍必须是三个故障事件。
        self.assertEqual(FAMILY_MAP[labels[starts]].tolist(), [1, 1, 1])

    def test_label_boundaries(self):
        ids = np.array([0, 1, 11, 12, 13, 19, 20, 31, 32, 38, 39, 45, 46, 47])
        self.assertEqual(FAMILY_MAP[ids].tolist(), [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])

    def test_split_reproducible_and_stratified(self):
        fine = np.repeat([1, 12, 47], 20)
        first = split_blocks(fine, 42)
        self.assertTrue(np.array_equal(first, split_blocks(fine, 42)))
        self.assertFalse(np.array_equal(first, split_blocks(fine, 123)))
        for label in [1, 12, 47]:
            self.assertEqual(np.bincount(first[fine == label], minlength=3).tolist(), [14, 3, 3])

    def test_train_only_scale_and_sensitivity_view(self):
        base = np.arange(24, dtype=np.float32)
        samples = {"split": np.repeat([0, 1, 2], 8), "label": np.tile(np.arange(8), 3),
                   "saturation_candidate": np.r_[np.zeros(8, dtype=bool), np.ones(16, dtype=bool)]}
        samples.update(current=np.tile(base[:, None, None], (1, 3, 20)),
                       dc=np.tile(base[:, None, None], (1, 2, 20)), rms=np.tile(base[:, None], (1, 3)))
        stats = fit_normalization(samples, "raw")
        self.assertEqual(stats["rms_median"].tolist(), [3.5] * 3)
        self.assertEqual(stats["rms_iqr"].tolist(), [3.5] * 3)
        samples["rms"][8:] = 1e9
        self.assertTrue(np.array_equal(stats["rms_median"], fit_normalization(samples, "raw")["rms_median"]))
        self.assertTrue(np.array_equal(stats["train_indices"], fit_normalization(samples, "exclude_saturation")["train_indices"]))
        transformed = normalize(samples["rms"], stats["rms_median"], stats["rms_iqr"], 10)
        self.assertEqual(float(transformed.max()), 10.0)
        self.assertEqual(float(samples["rms"][-1, 0]), 1e9)


class ModelTests(unittest.TestCase):
    """验证融合结构、权重和梯度，而不仅检查程序不报错。"""
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.batch = {"current": torch.randn(8, 3, 20), "dc": torch.randn(8, 2, 20), "rms": torch.randn(8, 3)}

    def test_logits_and_gate_gradients(self):
        for fusion in ("v0", "concat", "mean", "static", "dynamic"):
            model = FaultModel(fusion)
            logits, weights = model(self.batch)
            self.assertEqual(tuple(logits.shape), (8, 8))
            torch.nn.functional.cross_entropy(logits, torch.arange(8)).backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
            if weights is not None:
                self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(8)))
            if fusion == "dynamic":
                self.assertGreater(float(weights.detach().std(dim=0).max()), 0)
                self.assertGreater(float(model.gate[0].weight.grad.abs().sum()), 0)
            if fusion == "static":
                self.assertTrue(torch.allclose(weights[0], weights[-1]))

    def test_dynamic_single_and_pairs(self):
        for branches in (("rms",), ("current", "dc"), ("dc", "rms")):
            logits, weights = FaultModel("dynamic", branches)(self.batch)
            self.assertEqual(tuple(logits.shape), (8, 8))
            self.assertEqual(tuple(weights.shape), (8, len(branches)))

    def test_metrics_include_unpredicted_classes(self):
        confusion = np.zeros((8, 8), dtype=np.int64)
        confusion[:, 0] = 10
        metrics = classification_metrics(confusion)
        self.assertAlmostEqual(metrics["accuracy"], 0.125)
        self.assertAlmostEqual(metrics["macro_recall"], 0.125)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 9) / 8)


if __name__ == "__main__":
    unittest.main()
