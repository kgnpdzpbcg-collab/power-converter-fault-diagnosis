"""记录分支权重及梯度随训练的变化，观测过程不参与优化或模型选择。"""

import numpy as np


class EpochTrace:
    """累计一轮训练统计。

    权重均值、标准差和熵按样本汇总；梯度 L2 范数按训练批次取均值。
    梯度在 backward 后、optimizer.step 前读取，不改变梯度或随机数状态。
    梯度范数反映训练信号大小，不能直接解释为分支的物理重要性。
    """
    def __init__(self, branches):
        self.branches = tuple(branches)
        self.samples = self.batches = 0
        self.entropy = 0.0
        self.weight_sum = np.zeros(len(branches), dtype=np.float64)
        self.weight_squared_sum = self.weight_sum.copy()
        self.below = self.weight_sum.copy()
        self.gradient_sum = self.weight_sum.copy()

    def observe(self, model, weights):
        """weights 为本批前向返回的 [B,M]；仅接受拥有显式融合权重的模型。"""
        values = weights.detach().double().numpy()
        self.samples += len(values)
        self.batches += 1
        self.weight_sum += values.sum(axis=0)
        self.weight_squared_sum += (values ** 2).sum(axis=0)
        self.below += (values < 0.01).sum(axis=0)
        self.entropy += float(-(values * np.log(np.clip(values, 1e-12, 1))).sum())
        for index, name in enumerate(self.branches):
            squared = sum(float(parameter.grad.detach().double().square().sum())
                          for parameter in model.encoders[name].parameters() if parameter.grad is not None)
            self.gradient_sum[index] += squared ** 0.5

    def record(self, epoch):
        """返回可直接写 CSV 的一行；浮点舍入导致的极小负方差截为零。"""
        mean = self.weight_sum / self.samples
        std = np.sqrt(np.maximum(self.weight_squared_sum / self.samples - mean ** 2, 0))
        result = {"epoch": epoch, "samples": self.samples, "batches": self.batches,
                  "gate_entropy_mean": self.entropy / self.samples}
        for index, name in enumerate(self.branches):
            result.update({f"{name}_weight_mean": mean[index], f"{name}_weight_std": std[index],
                           f"{name}_fraction_weight_below_0_01": self.below[index] / self.samples,
                           f"{name}_gradient_l2_batch_mean": self.gradient_sum[index] / self.batches})
        return result
