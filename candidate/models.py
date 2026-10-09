"""只修改短波形编码；三源64维表示、等权平均、分类头与旧模型一致。"""

from copy import deepcopy

import torch
from torch import nn

from converter.models import FaultModel


VARIANTS = ("baseline", "wide", "multiscale")
DESCRIPTIONS = ("原等权平均", "普通宽卷积容量对照", "多尺度残差候选")


class MultiscaleEncoder(nn.Module):
    """输入[B,C,20]，非线性k3/k5并行＋输入残差，输出[B,64]。

    两分支各16通道，拼接后仍为32；尾部k3卷积和池化沿用基线初值。
    保留跨测量通道卷积，不使用大核、注意力或新的融合权重。
    """
    def __init__(self, channels, base_encoder):
        super().__init__()
        self.path3 = nn.Sequential(nn.Conv1d(channels, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU())
        self.path5 = nn.Sequential(nn.Conv1d(channels, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU())
        self.project = nn.Conv1d(32, 32, 1)
        self.skip = nn.Conv1d(channels, 32, 1)
        self.relu = nn.ReLU()
        self.tail = nn.Sequential(*(deepcopy(layer) for layer in list(base_encoder.children())[3:]))

    def forward(self, x):
        """投影并行局部特征后与原输入投影相加，长度保持20，再提取64维表示。"""
        features = torch.cat((self.path3(x), self.path5(x)), dim=1)
        return self.tail(self.relu(self.project(features) + self.skip(x)))


def wide_encoder(channels):
    """普通两层k3卷积，首层38通道，容量接近多尺度候选，输出仍为64维。"""
    return nn.Sequential(nn.Conv1d(channels, 38, 3, padding=1), nn.BatchNorm1d(38), nn.ReLU(),
                         nn.Conv1d(38, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten())


class CandidateModel(FaultModel):
    """复用旧forward，标签/时间/事件信息均不进入模型。

    先创建完全相同的原模型，使各变体RMS及分类头初值一致。
    替换编码器时隔离随机数消耗，训练Dropout起点仍与基线相同。
    不把不同形状的卷积参数声称为完全相同。
    """
    def __init__(self, fusion="mean", branches=("current", "dc", "rms"), variant="baseline"):
        if fusion != "mean" or variant not in VARIANTS:
            raise ValueError("候选实验只使用等权融合及预先固定的三个编码变体")
        super().__init__(fusion, branches)
        self.variant = variant
        if variant != "baseline":
            with torch.random.fork_rng(devices=[]):
                for name, channels in (("current", 3), ("dc", 2)):
                    if name in self.encoders:
                        self.encoders[name] = (wide_encoder(channels) if variant == "wide"
                                               else MultiscaleEncoder(channels, self.encoders[name]))
