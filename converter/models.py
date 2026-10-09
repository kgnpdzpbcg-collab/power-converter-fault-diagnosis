"""V0 与公平的融合对照；测量分支是特征编码专家，不使用稀疏路由。"""

import torch
from torch import nn


def waveform_encoder(channels):
    """输入 [B,C,20]，两层小卷积及全局平均池化输出 [B,64]。"""
    return nn.Sequential(nn.Conv1d(channels, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
                         nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten())


class FaultModel(nn.Module):
    """八类故障分类，提供四种融合以及冻结的 V0。

    v0：RMS 编码到 32 维，三源拼接 160 维；其他对照统一 RMS 到 64 维。
    concat：拼接；mean：等权平均；static：训练得到的全局 Softmax 权重；
    dynamic：用当前样本的分支表示计算 Softmax 权重，再加权求和。
    不把标签、时间戳或异常标记传入门控；门控权重不能解释为物理因果贡献。
    """
    def __init__(self, fusion="v0", branches=("current", "dc", "rms")):
        super().__init__()
        if fusion not in ("v0", "concat", "mean", "static", "dynamic"):
            raise ValueError(f"未知融合方式: {fusion}")
        if not branches or len(set(branches)) != len(branches) or any(name not in ("current", "dc", "rms") for name in branches):
            raise ValueError("测量组合必须包含不重复的 current/dc/rms")
        self.fusion, self.branches = fusion, tuple(branches)
        rms_dim = 32 if fusion == "v0" else 64
        encoders = {"current": lambda: waveform_encoder(3), "dc": lambda: waveform_encoder(2),
                    "rms": lambda: nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, rms_dim), nn.ReLU())}
        self.encoders = nn.ModuleDict({name: encoders[name]() for name in branches})
        dims = [rms_dim if name == "rms" else 64 for name in branches]
        # 单源时无需门控，避免引入 Softmax(1) 对应的无效参数。
        if fusion == "dynamic" and len(branches) > 1:
            self.gate = nn.Sequential(nn.Linear(sum(dims), 32), nn.ReLU(), nn.Linear(32, len(branches)))
        if fusion == "static":
            self.weight_logits = nn.Parameter(torch.zeros(len(branches)))
        fused_dim = sum(dims) if fusion in ("v0", "concat") else 64
        self.classifier = nn.Sequential(nn.Linear(fused_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 8))

    def forward(self, inputs):
        """返回 logits[B,8] 和可选权重[B,M]；外部交叉熵接收未 Softmax 的 logits。"""
        encoded = [self.encoders[name](inputs[name]) for name in self.branches]
        weights = None
        if self.fusion in ("v0", "concat"):
            fused = torch.cat(encoded, dim=1)
        else:
            stacked = torch.stack(encoded, dim=1)
            batch = stacked.shape[0]
            if self.fusion == "mean" or len(encoded) == 1:
                weights = stacked.new_full((batch, len(encoded)), 1.0 / len(encoded))
            elif self.fusion == "static":
                weights = self.weight_logits.softmax(dim=0).expand(batch, -1)
            else:
                weights = self.gate(torch.cat(encoded, dim=1)).softmax(dim=1)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(fused), weights
