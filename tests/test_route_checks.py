"""检查诊断的数学正确性与训练独占统计，不据真实样本设置规则。"""

import unittest
import numpy as np

from diagnostics.probes import (balanced_weights, fit_probe, predict_scores,
                               metrics, waveform_features, paired_event_bootstrap)


class FeasibilityTests(unittest.TestCase):
    def test_event_and_class_weights(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        blocks = np.array([1, 1, 2, 3, 3, 3])
        weights = balanced_weights(labels, blocks)
        self.assertAlmostEqual(weights[labels == 0].sum(), 0.5)
        self.assertAlmostEqual(weights[labels == 1].sum(), 0.5)
        self.assertAlmostEqual(weights[blocks == 1].sum(), weights[blocks == 2].sum())

    def test_probe_train_only_and_replay(self):
        x = np.array([[-2, 0], [-1, 0], [1, 0], [2, 0]], dtype=float)
        y = np.array([0, 0, 1, 1]); blocks = np.arange(4)
        for method in ("ridge", "prototype"):
            model = fit_probe(x, y, blocks, [0, 1], method)
            self.assertTrue(np.array_equal(predict_scores(model, x).argmax(1), y))
            before = {key: value.copy() for key, value in model.items()}
            predict_scores(model, x * 1000)
            self.assertTrue(all(np.array_equal(value, model[key]) for key, value in before.items()))
            self.assertTrue(np.isfinite(predict_scores(model, x)).all())

    def test_known_bin_and_parseval(self):
        t = np.arange(20)
        x = (4 + 2 * np.sin(2 * np.pi * 3 * t / 20))[None, None, :]
        result = waveform_features(x)
        self.assertEqual(int(result["spectral_shape"].argmax()), 2)
        self.assertAlmostEqual(result["spectral_shape"][0, 2], 1)
        expected = 20 * np.sum((x - x.mean(-1, keepdims=True)) ** 2)
        self.assertAlmostEqual(result["power"].sum(), expected)

    def test_shape_gain_offset_and_circular_shift_invariance(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(5, 3, 20))
        a = waveform_features(x)
        b = waveform_features(x * 1e6 + 12)
        self.assertTrue(np.allclose(a["spectral_shape"], b["spectral_shape"], atol=1e-12))
        self.assertTrue(np.allclose(a["time_shape"], b["time_shape"], atol=1e-12))
        c = waveform_features(np.roll(x, 3, axis=-1))
        self.assertTrue(np.allclose(a["spectral_shape"], c["spectral_shape"], atol=1e-12))

    def test_constant_waveform_and_nyquist(self):
        constant = waveform_features(np.ones((2, 3, 20)) * 17)
        self.assertTrue(np.array_equal(constant["spectral_shape"], np.zeros((2, 30))))
        self.assertTrue(np.array_equal(constant["time_shape"], np.zeros((2, 60))))
        nyquist = waveform_features(((-1.0) ** np.arange(20))[None, None, :])
        self.assertAlmostEqual(nyquist["spectral_shape"][0, -1], 1)
        self.assertAlmostEqual(nyquist["power"].sum(), 400)

    def test_float_confusion_not_truncated(self):
        result = metrics([[0.5, 0.25], [0, 0.25]])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertEqual(result["support"], [0.75, 0.25])

    def test_paired_bootstrap_same_predictions_zero_difference(self):
        y = np.tile([0, 1], 10)
        result = paired_event_bootstrap(y, y, y, np.repeat(np.arange(10), 2), [0, 1], repeats=20)
        self.assertEqual(result["percentile_95_interval"], [0, 0])


if __name__ == "__main__":
    unittest.main()
