"""RF输入边界测试：标签元数据隔离、三源拼接顺序及共同样本行顺序。"""

import unittest
import numpy as np

from baselines.rf import feature_matrix


class RFInputTests(unittest.TestCase):
    """固定小样本逐元素核对，避免传统模型意外引入标签或错位拼接。"""
    def setUp(self):
        self.inputs = {"current": np.arange(120).reshape(2,3,20),
                       "dc": np.arange(80).reshape(2,2,20)+1000,
                       "rms": np.arange(6).reshape(2,3)+2000,
                       "label": np.array([0,6]), "timestamp": np.array([8,9])}

    def test_exact_columns_and_shared_rows(self):
        """103维必须依次为60电流、40DC、3RMS，不能按分支改变记录归属。"""
        current = feature_matrix(self.inputs, "RF-Current")
        all_features = feature_matrix(self.inputs, "RF-All")
        np.testing.assert_array_equal(current, self.inputs["current"].reshape(2,60))
        np.testing.assert_array_equal(all_features[:,:60], current)
        np.testing.assert_array_equal(all_features[:,60:100], self.inputs["dc"].reshape(2,40))
        np.testing.assert_array_equal(all_features[:,100:], self.inputs["rms"])
        self.assertEqual(all_features.dtype, np.float32)

    def test_metadata_cannot_change_features(self):
        """改变label/时间/事件元数据不改变RF任一输入值。"""
        before = {m: feature_matrix(self.inputs, m) for m in ("RF-Current", "RF-All")}
        self.inputs.update(label=np.array([7,1]), timestamp=np.array([-1,-2]), block=np.array([6,3]))
        for model_name, features in before.items():
            np.testing.assert_array_equal(features, feature_matrix(self.inputs, model_name))


if __name__ == "__main__":
    unittest.main()
