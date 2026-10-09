"""第二轮受控融合实验：特征归一化与门控初始权重的二因素比较。"""

import torch
from torch import nn

from .models import FaultModel


class FaultModelV2(FaultModel):
    """复用首轮编码器及分类头，仅比较两个明确的改动。

    branch_norm=True 时，在每个 64 维分支表示后加入可学习 LayerNorm。
    uniform_gate_init=True 时，将门控末层置零，初始 Softmax 权重严格等权。
    输入和输出与 FaultModel 一致；支持 mean/dynamic，标签和事件元数据
    都不进入模型。归一化可能移除有用幅值信息，必须同时比较 mean 对照。

    先创建等权模型，保证同 seed 下各变体的编码器和分类头初值相同。
    门控在 fork_rng 中创建，避免它消耗随机数后改变 Dropout 的随机序列。
    该初始化顺序与首轮动态模型不同，所以要与本轮动态对照作因子比较。
    """
    def __init__(self, fusion="mean", branches=("current", "dc", "rms"),
                 branch_norm=False, uniform_gate_init=False):
        if fusion not in ("mean", "dynamic"):
            raise ValueError("第二轮仅比较 mean 与 dynamic")
        if uniform_gate_init and (fusion != "dynamic" or len(branches) < 2):
            raise ValueError("等权门控初始化只适用于多分支动态融合")
        super().__init__("mean", branches)
        self.fusion = fusion
        if branch_norm:
            for name in self.branches:
                # LayerNorm 的仿射参数初始为 1/0，不消耗随机数；不改变输入归一化。
                self.encoders[name] = nn.Sequential(self.encoders[name], nn.LayerNorm(64))
        if fusion == "dynamic" and len(self.branches) > 1:
            with torch.random.fork_rng(devices=[]):
                self.gate = nn.Sequential(nn.Linear(64 * len(self.branches), 32), nn.ReLU(),
                                          nn.Linear(32, len(self.branches)))
            if uniform_gate_init:
                nn.init.zeros_(self.gate[-1].weight)
                nn.init.zeros_(self.gate[-1].bias)
