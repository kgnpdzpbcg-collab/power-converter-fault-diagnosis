"""V3：三个全局特征 token 的自注意力，以及原始特征的跳跃拼接。"""

import torch
from torch import nn

from .models import FaultModel


VARIANTS = ("b0_concat", "b1_mlp_skip", "b2_attention", "b3_attention_skip")


class FaultModelV3(FaultModel):
    """共享原始编码器，隔离交互形式与是否保留原始特征。

    三个输入分支、输出 64 维和 token 顺序固定，不用于测量组合消融。
    B0 完全沿用旧 concat；B1/B3 头同初值；B2/B3 注意力同初值。
    新模块使用局部派生种子，构造后恢复全局 RNG，保持训练随机性可配对。
    """
    def __init__(self, fusion="b3_attention_skip", branches=("current", "dc", "rms")):
        if fusion not in VARIANTS or tuple(branches) != ("current", "dc", "rms"):
            raise ValueError("V3 必须使用 B0–B3 和固定 current/dc/rms 次序")
        super().__init__("concat", branches)
        self.fusion = fusion
        if fusion == "b0_concat":
            return
        seed = torch.initial_seed()
        # 相同输入尺寸的分类头逐元素相同；局部构造不消耗训练用 RNG。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((seed + 1) % (2**64))
            width = 64 if fusion == "b2_attention" else 256
            self.classifier = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                            nn.Dropout(0.2), nn.Linear(128, 8))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((seed + 2) % (2**64))
            # 归一化仅在交互路径；原始路径的编码器输出直接拼接。
            self.interaction_norm = nn.LayerNorm(64)
            if fusion == "b1_mlp_skip":
                self.interaction_mlp = nn.Sequential(nn.Linear(192, 112), nn.ReLU(),
                                                     nn.Linear(112, 64), nn.ReLU())
            else:
                self.attention = nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True)
                self.interaction_mlp = nn.Sequential(nn.Linear(192, 64), nn.ReLU())

    def fused_features(self, inputs, collect_attention=False):
        """返回融合特征和可选逐头矩阵；输入只包含三种测量，不读取标签。

        注意力实现始终计算逐头权重，从而避免 need_weights 两种内核导致
        诊断和标准前向出现数值差异。3×3 小矩阵开销有限；只在诊断时返回。
        对角线未屏蔽，每个 query 都可访问三个 key；Flatten 保留分支次序。
        """
        tokens = torch.stack([self.encoders[name](inputs[name]) for name in self.branches], dim=1)
        original = tokens.flatten(1)
        if self.fusion == "b0_concat":
            return original, None
        normalized = self.interaction_norm(tokens)
        attention_weights = None
        if self.fusion == "b1_mlp_skip":
            interaction = self.interaction_mlp(normalized.flatten(1))
        else:
            attended, weights = self.attention(normalized, normalized, normalized,
                                               need_weights=True, average_attn_weights=False)
            interaction = self.interaction_mlp(attended.flatten(1))
            if collect_attention:
                attention_weights = weights
        fused = interaction if self.fusion == "b2_attention" else torch.cat((original, interaction), dim=1)
        return fused, attention_weights

    def forward(self, inputs):
        """保持旧训练接口：(logits[B,8], None)，不把注意力当三源门控权重。"""
        fused, _ = self.fused_features(inputs)
        return self.classifier(fused), None
