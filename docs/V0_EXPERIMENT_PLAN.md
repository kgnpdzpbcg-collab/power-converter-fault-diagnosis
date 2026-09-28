# V0 实验方案：多源电气信号融合的功率电子变换器故障诊断

> 状态：**Frozen V0**
>
> 目标：形成一篇方法不复杂、实验设计可靠、可以支撑普通会议论文的电子设备/功率电子变换器故障诊断工作。
>
> V0 不追求复杂模型创新。核心研究问题是：**不同电气测量是否具有互补故障信息，简单的多源神经网络融合能否在合理的数据划分下改善粗粒度故障诊断。**

---

## 1. 研究问题

### 1.1 主问题

给定同一运行时刻附近的三类电气测量：

1. 三相瞬时电流；
2. DC 侧瞬时电压/电流；
3. 三相 RMS 电压；

是否可以通过一个轻量级多分支神经网络联合学习这些互补信息，从而比单一测量源获得更稳定的电子设备故障诊断性能？

### 1.2 主假设

若不同测量源对不同故障族的敏感性不同，则：

[
F_{fusion}=f(F_{current},F_{DC},F_{RMS})
]

应当比任一单源表示具有更好的宏平均分类性能，尤其体现在：

- Macro-F1；
- Macro-Recall；
- 少数故障大类；
- 混淆矩阵中的跨故障族误判。

### 1.3 V0 不解决的问题

V0 不以以下问题作为主贡献：

- 47 种细粒度故障逐类定位；
- open-set / unknown fault；
- domain adaptation；
- RUL；
- LLM / RAG / Agent；
- 复杂物理建模；
- 在线自适应；
- 新型 Transformer 架构。

这些全部留到后续工作，避免会议论文范围失控。

---

## 2. 数据集

### 2.1 数据来源

数据集：**Power Converter Fault Diagnosis Dataset**

对象：

- 2 MW two-level voltage source inverter (2L-VSI)
- 400/690 V，50 Hz 三相电网
- 1500 V DC bus
- Typhoon HIL606 实时 Hardware-in-the-Loop 仿真

公开数据包含：

- 1 个正常状态；
- 47 种故障/异常状态；
- 总计 48 个 `id_error`。

相关论文：

García-Campos et al., 2026, *Hybrid Fault-Space Restructuring for Machine Learning-Based Fault Diagnosis in Power Electronic Converters*.

### 2.2 实际公开文件

本地核验 `daset.zip` 后，压缩包只有三个 JSON：

| 文件 | 样本数 | `mea_value` 维度 | 结构解释 |
|---|---:|---:|---|
| `id_mea_10.json` | 58,166 | 3 | 三相 RMS 电压 |
| `id_mea_3.json` | 228,336 | 40 | DC 电压 + 电流，可视为 `2×20` |
| `id_mea_2.json` | 386,705 | 60 | 三相电流，可视为 `3×20` |

总数：

[
58,166 + 228,336 + 386,705 = 673,207
]

与配套论文报告的总样本数一致。

### 2.3 单条记录结构

核心字段：

```json
{
  "timestamp": "...",
  "device": "test1",
  "id_mea": 2,
  "id_error": 0,
  "mea_value": [...],
  "experiment_id": "experiment_3",
  "store_timestamp": "..."
}
```

机器学习所需核心字段：

[
x = 	exttt{mea_value},qquad y = 	exttt{id_error}
]

其中：

- `id_error = 0`: Normal
- `id_error = 1...47`: Fault / abnormal condition

---

## 3. 三种输入的统一表示

### 3.1 `id_mea = 2`

论文定义为 instantaneous three-phase grid currents。

原始：

[
x_2inmathbb{R}^{60}
]

按三相重构：

[
X_2inmathbb{R}^{3	imes20}
]

即：

```text
Ia: 20 points
Ib: 20 points
Ic: 20 points
```

模型输入 shape：

```text
[B, 3, 20]
```

禁止把 60 点错误理解成单通道连续 60 点时序。

### 3.2 `id_mea = 3`

论文定义为 instantaneous DC-link capacitor voltage and input current。

原始：

[
x_3inmathbb{R}^{40}
]

重构：

[
X_3inmathbb{R}^{2	imes20}
]

模型输入：

```text
[B, 2, 20]
```

### 3.3 `id_mea = 10`

三相 RMS voltage：

[
x_{10}inmathbb{R}^{3}
]

模型输入：

```text
[B, 3]
```

该分支视为低维状态量，不使用卷积。

---

## 4. 多源同步规则

### 4.1 对齐原则

以样本最少的 `id_mea=10` 为时间锚点。

对于每一条 `id_mea=10`：

1. 在 `id_mea=2` 中查找时间戳最近的样本；
2. 在 `id_mea=3` 中查找时间戳最近的样本；
3. 两个匹配均要求 `|Δt| < 0.2 s`；
4. 三条记录必须具有相同 `id_error`；
5. 不满足条件的样本直接舍弃，不进行人工修正。

### 4.2 本地核验结果

按照上述规则，可得到：

- 时间上三源均成功匹配：**58,155**
- 三源标签也完全一致：**58,153**
- 标签不一致：2

因此 V0 固定使用：

[
oxed{N=58,153}
]

个三源同步样本。

每个最终样本表示为：

```text
sample
├── current: [3,20]
├── dc_vi:   [2,20]
├── rms_v:   [3]
├── fine_label: 0...47
├── coarse_label: 0...7
├── timestamp
└── episode_id
```

---

## 5. 主任务：8 类粗粒度故障诊断

### 5.1 为什么不以 48 类作为主任务

原始论文已经显示，大量细粒度故障在单一测量空间中存在明显重叠。

本项目目标不是强行证明 47 种故障都可以可靠区分，而是模拟电子设备的一级故障筛查：

> 首先识别故障所属大类，再由后续检测完成精细定位。

因此 V0 采用物理含义明确的粗粒度标签，而不是依据神经网络结果事后合并。

### 5.2 固定标签映射

| Coarse ID | 类别 | 原始 `id_error` |
|---:|---|---|
| 0 | Normal | 0 |
| 1 | AC Short-Circuit | 1–11 |
| 2 | Weak Grid / Grid Impedance | 12 |
| 3 | Phase Disconnection / Open Circuit | 13–19 |
| 4 | Grid Electrical Abnormality | 20–31 |
| 5 | Harmonic Distortion | 32–38 |
| 6 | DC-Side Fault | 39–45 |
| 7 | IGBT Blocking Fault | 46–47 |

物理解释：

- **AC Short-Circuit**：相间、相地及组合短路；
- **Weak Grid**：电网阻抗变化；
- **Phase Disconnection**：单相、多相、三相断开；
- **Grid Electrical Abnormality**：幅值、尖峰、相序、相位、频率变化；
- **Harmonic Distortion**：3/5/7 次及组合谐波；
- **DC-Side Fault**：DC 母线尖峰、低压、DC 端短路；
- **IGBT Blocking Fault**：IGBT 开态/闭态 blocking。

### 5.3 三源同步数据的类别数量

本地对齐后的 58,153 条样本：

| 类别 | 样本数 |
|---|---:|
| Normal | 41,642 |
| AC Short-Circuit | 3,704 |
| Weak Grid | 417 |
| Phase Disconnection | 3,340 |
| Grid Electrical Abnormality | 3,898 |
| Harmonic Distortion | 2,184 |
| DC-Side Fault | 2,282 |
| IGBT Blocking Fault | 686 |
| **Total** | **58,153** |

Normal 占约 71.6%，因此 Accuracy 只能作为辅助指标。

---

## 6. Episode 重建与防泄漏划分

### 6.1 为什么禁止 sample-level random split

数据的 `id_error` 呈现连续运行段，同一次故障注入会产生多个连续、高度相关的样本。

若直接：

```python
train_test_split(rows)
```

则可能出现同一 Fault Episode 的部分时刻进入 Train、部分时刻进入 Test，产生明显 event-level information leakage。

### 6.2 Episode 定义

按照时间排序，以原始 `id_error` 的连续非零区段定义一个 fault episode。

本地核验 `id_mea=10`：

- fault episode 数量约：**2532**
- 单个 fault episode 中位长度：约 **7 个 record**

每个 fault episode 与其后的正常恢复区间绑定为同一个 `episode_block`，以避免 fault 后的恢复过渡样本被拆入不同数据集。

首个故障前的初始正常区间并入第一个 episode block。

### 6.3 固定划分

按 `episode_block` 而不是 sample 划分：

```text
Train : 70%
Val   : 15%
Test  : 15%
```

要求：

1. 一个 episode block 只能属于一个 split；
2. 按原始 `fine_label` 尽可能分层；
3. 固定第一版 manifest；
4. 后续所有模型使用完全相同的 manifest；
5. 不允许为了某个模型性能重新划分数据。

主 manifest：

```text
seed = 42
```

额外稳定性实验：

```text
seeds = [42, 123, 2026]
```

三 seed 用于模型训练随机性；数据 manifest 默认固定。

---

## 7. 数据质量 Gate 0

在训练任何网络前必须完成。

### G0-1 数据完整性

检查：

- NaN
- Inf
- 空向量
- 维度错误
- timestamp 缺失
- label 超出 0–47
- 重复记录

### G0-2 时间一致性

检查：

- 三源 timestamp 差；
- 排序；
- 时间大 gap；
- 重复时间戳。

### G0-3 数值异常

前期检查发现部分数据存在数量级异常的大数值，且疑似与部分谐波故障及故障恢复阶段相关。

V0 不允许未经验证直接删除。

固定两套协议。

#### Protocol-Raw

保留全部 58,153 条同步样本。

归一化参数只由 Train 估计。

采用：

- channel-wise median / IQR robust scaling；
- scaling 后统一 clip 到 `[-10, 10]`。

#### Protocol-Sensitivity

定义候选数值饱和样本：

[
max(|x|)ge10^8
]

从三源任一分支出现该现象的 triplet 中打 `saturation_candidate=1`。

执行一个敏感性实验：

- Raw 全量；
- Exclude saturation candidate。

若两种协议下方法结论一致，正文以 Raw 为主，敏感性结果作为数据质量说明。

**禁止依据类别表现人工选择清洗阈值。**

---

## 8. 训练集归一化

禁止使用全数据统计量。

### 8.1 Current branch

输入 `[3,20]`。

每个 phase 单独计算 Train：

- median；
- IQR。

### 8.2 DC V/I branch

输入 `[2,20]`。

电压与电流通道分别计算 Train statistics。

### 8.3 RMS branch

三个 RMS feature 分别计算 Train statistics。

Val/Test 只使用 Train statistics 变换。

---

## 9. V0 模型

### 9.1 设计原则

- 结构简单；
- 参数量小；
- 不加入 Transformer；
- 不加入 attention；
- 不加入 BiLSTM；
- 不做复杂手工特征；
- 先验证多源融合本身是否有效。

### 9.2 Current branch

输入：

```text
[B,3,20]
```

结构：

```text
Conv1d(3, 32, kernel=3, padding=1)
BatchNorm
ReLU
Conv1d(32, 64, kernel=3, padding=1)
BatchNorm
ReLU
AdaptiveAvgPool1d(1)
Flatten
```

输出：

[
z_Iinmathbb{R}^{64}
]

### 9.3 DC V/I branch

输入：

```text
[B,2,20]
```

结构：

```text
Conv1d(2, 32, kernel=3, padding=1)
BatchNorm
ReLU
Conv1d(32, 64, kernel=3, padding=1)
BatchNorm
ReLU
AdaptiveAvgPool1d(1)
Flatten
```

输出：

[
z_{DC}inmathbb{R}^{64}
]

### 9.4 RMS branch

输入：

```text
[B,3]
```

结构：

```text
Linear(3,32)
ReLU
Linear(32,32)
ReLU
```

输出：

[
z_{RMS}inmathbb{R}^{32}
]

### 9.5 Fusion

直接 concat：

[
z=[z_I;z_{DC};z_{RMS}]inmathbb{R}^{160}
]

分类头：

```text
Linear(160,128)
ReLU
Dropout(0.2)
Linear(128,8)
```

V0 不使用额外 fusion module。V0 的研究变量是“多源 vs 单源”，而不是“哪种复杂注意力 fusion 最先进”。

---

## 10. 损失函数

类别严重不平衡。

主损失：

[
mathcal{L}=-sum_c w_cy_clog p_c
]

其中 class weight 只由 Train 统计：

[
w_c=rac{N_{train}}{Kcdot n_c}
]

并归一化使平均权重约为 1。

V0 不使用：

- focal loss；
- contrastive loss；
- auxiliary loss。

如果 Weighted CE 明显训练不稳定，再进入 V1。

---

## 11. 训练超参数

固定初始配置：

```yaml
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
batch_size: 256
max_epochs: 100
early_stopping_patience: 15
monitor: val_macro_f1
dropout: 0.2
train_seeds: [42, 123, 2026]
```

学习率调度：

```text
ReduceLROnPlateau
factor = 0.5
patience = 5
```

checkpoint：

```text
best val Macro-F1
```

最终 Test 只在训练结束后评估。禁止反复观察 Test 后调参。

---

## 12. Baseline

### B0. Random Forest

输入：

```text
flatten(id2) + flatten(id3) + id10
= 60 + 40 + 3
= 103 dims
```

用于连接原论文传统 ML 思路。

### B1. MLP

输入 103-D flattened vector。

结构：

```text
103 -> 256 -> 128 -> 8
```

用于判断性能提升是否仅来自“使用神经网络”。

### B2. Current-only CNN

仅 `id_mea=2`。

### B3. DC-only CNN

仅 `id_mea=3`。

### B4. RMS-only MLP

仅 `id_mea=10`。

### B5. Proposed Multi-source CNN

`id2 + id3 + id10`，这是论文主方法。

---

## 13. 核心消融

固定输入组合：

| Experiment | Current | DC V/I | RMS |
|---|:---:|:---:|:---:|
| A1 | ✓ |  |  |
| A2 |  | ✓ |  |
| A3 |  |  | ✓ |
| A4 | ✓ | ✓ |  |
| A5 | ✓ |  | ✓ |
| A6 |  | ✓ | ✓ |
| A7 | ✓ | ✓ | ✓ |

核心问题：

1. 哪个单源最强？
2. 哪两个源互补最大？
3. 三源是否稳定优于最强单源？

---

## 14. 辅助实验

### 14.1 48 类诊断

48-class 只作为探索性辅助实验。

目的：

- 与原论文的 fine-grained setting 对照；
- 观察哪些细粒度故障仍无法区分；
- 判断多源信息是否减少原论文中的部分 overlap。

不要求 V0 在 48 类上取得很高性能。

### 14.2 Random-row split 对照

额外可运行一次普通 sample-level random split：

```text
80/20
```

只用于展示 Random-row split 与 episode-level split 的性能差异，该结果不能作为主论文的最好结果。

---

## 15. 评价指标

### 15.1 主指标

[
oxed{	ext{Macro-F1}}
]

原因：Normal 占约 71.6%，Accuracy 会显著受到多数类影响。

### 15.2 必报指标

- Macro-F1
- Macro-Precision
- Macro-Recall
- Balanced Accuracy
- Overall Accuracy
- Per-class F1
- Confusion Matrix

三次训练：

```text
mean ± std
```

### 15.3 复杂度

至少报告：

- Parameters
- Model size
- 单样本 inference latency（可选 GPU + CPU）
- FLOPs（若工具计算方便）

不要求必须做 Raspberry Pi 部署。

---

## 16. 核心结果表模板

### Table 1：Baseline

| Method | Input | Macro-F1 | Bal.Acc | Acc | Params |
|---|---|---:|---:|---:|---:|
| RF | All | | | | |
| MLP | All | | | | |
| CNN | Current | | | | |
| CNN | DC | | | | |
| MLP | RMS | | | | |
| **Ours** | **All** | | | | |

### Table 2：Modality Ablation

| Current | DC | RMS | Macro-F1 | Macro-R | Acc |
|:---:|:---:|:---:|---:|---:|---:|
| ✓ | | | | | |
| | ✓ | | | | |
| | | ✓ | | | |
| ✓ | ✓ | | | | |
| ✓ | | ✓ | | | |
| | ✓ | ✓ | | | |
| ✓ | ✓ | ✓ | | | |

### Table 3：Per-class F1

| Class | Best Single | Fusion |
|---|---:|---:|
| Normal | | |
| AC Short | | |
| Weak Grid | | |
| Phase Disconnect | | |
| Grid Abnormality | | |
| Harmonic | | |
| DC-side | | |
| IGBT | | |

### Table 4：Data Protocol Sensitivity

| Protocol | Macro-F1 | Macro-R | Acc |
|---|---:|---:|---:|
| Raw + robust scaling | | | |
| Exclude saturation candidates | | | |

---

## 17. V0 成功判据

V0 不预设必须达到某个绝对准确率。

### Gate 1：数据协议成立

必须满足：

- 三源同步样本稳定复现；
- 无 episode 跨 split；
- Train-only normalization；
- 8 类在 Train/Val/Test 均有样本。

不满足则禁止训练主模型。

### Gate 2：神经网络 baseline 可训练

要求：

- loss 正常下降；
- 无 NaN；
- 三 seed 无明显崩溃；
- Macro-F1 显著高于 trivial majority baseline。

### Gate 3：融合是否有实际价值

优先希望：

[
F1_{fusion}>F1_{best-single}
]

并且增益至少在多数 seed 中存在。

若三源融合相对最强单源提升不足约 1 个百分点：

> **不立即添加复杂 attention。**

先检查：

1. 哪些类别没有增益；
2. pair-wise fusion 是否已经达到上限；
3. RMS 是否贡献很弱；
4. 三源对齐误差是否影响融合；
5. 极端数值是否主导结果。

只有这些问题排查完成后才进入 V1。

---

## 18. 论文主线

### 18.1 问题

功率电子变换器具有多种可观测电气量，但不同故障对不同测量变量的响应不同。单一测量源可能对某些故障敏感、对另外一些故障缺乏可分性。

### 18.2 方法

```text
Three-phase current ─ CNN ─┐
                            │
DC voltage/current ─ CNN ──┼─ Concatenate ─ FC ─ Fault class
                            │
RMS voltage ─────── MLP ───┘
```

### 18.3 实验论证

通过：

- 单源；
- 双源；
- 三源；
- traditional ML；
- simple NN；

系统比较。

### 18.4 主结论允许写到什么程度

若实验支持，可以写：

> 多源电气测量的联合学习能够利用不同测量变量之间的互补故障信息，在 episode-level 独立划分下改善功率电子变换器粗粒度故障诊断性能。

不能直接宣称：

- 解决了所有功率电子设备故障诊断问题；
- 实现工业现场泛化；
- 真实设备性能已验证；
- 适用于航空电子设备而无需进一步验证。

因为当前主数据仍来自 HIL 仿真。

---

## 19. 论文可用题目

暂定英文：

**Lightweight Multi-Source Electrical Signal Fusion for Fault Diagnosis of Power Electronic Converters**

可选：

**Multi-Source Electrical Measurement Fusion with a Lightweight Neural Network for Power Converter Fault Diagnosis**

中文工作题目：

**基于轻量级多源电气信号融合的功率电子变换器故障诊断方法**

---

## 20. 计划中的仓库结构

```text
power-converter-fault-diagnosis/
├── README.md
├── docs/
│   ├── V0_EXPERIMENT_PLAN.md
│   ├── DATA_AUDIT.md
│   └── EXPERIMENT_LOG.md
├── data/
│   └── README.md
├── configs/
│   ├── v0.yaml
│   └── baselines/
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_aligned_dataset.py
│   └── build_episode_split.py
├── src/
│   ├── data/
│   ├── models/
│   ├── train.py
│   └── evaluate.py
├── outputs/
│   └── .gitkeep
└── tests/
```

原始 Zenodo 数据不进入 Git。

---

## 21. 执行顺序

```text
Step 0  数据完整性检查
   ↓
Step 1  三源时间对齐
   ↓
Step 2  8类标签映射
   ↓
Step 3  episode 重建
   ↓
Step 4  固定 train/val/test manifest
   ↓
Step 5  RF / MLP baseline
   ↓
Step 6  三个 single-source model
   ↓
Step 7  三源 Fusion V0
   ↓
Step 8  两两融合 + 三源消融
   ↓
Step 9  数值异常敏感性实验
   ↓
Step 10  48类辅助实验（可选）
   ↓
Step 11  汇总结果并决定是否进入 V1
```

---

## 22. V0 冻结项

从现在开始，除非数据检查发现明确错误，以下内容不边训练边修改：

- 主任务：8 类；
- 标签映射；
- 三源对齐阈值：0.2 s；
- episode-level split；
- 主网络拓扑；
- weighted cross entropy；
- Macro-F1 主指标；
- baseline 列表；
- modality ablation；
- 3 个训练 seed。

原则：

> **先完整跑完 V0，再根据实验结果决定 V1，不在 V0 阶段一边训练一边堆模块。**

---

## 23. V0 最小论文贡献表述

如果结果成立，贡献控制在三点：

1. **建立三源同步故障诊断任务**：将公开的三相电流、DC 侧电压/电流和 RMS 电压按时间戳对齐，用于联合故障识别。
2. **提出轻量级多分支融合网络**：分别对波形和低维状态量进行特征学习，并通过简单特征级融合完成粗粒度故障诊断。
3. **采用 episode-level 独立评估协议**：避免同一次连续故障注入事件同时进入训练与测试，并系统评估不同测量源组合的诊断贡献。

这里的“提出”仅指本项目具体网络实现，不夸大为全新通用架构。

---

## 24. 当前最终决策

V0 最终固定为：

[
oxed{
58,153 synchronized samples
+
8 coarse fault classes
+
episode	ext{-}level split
+
multi	ext{-}branch lightweight CNN
+
modality ablation
}
]

这就是第一版会议论文实验主线。

在 V0 完整跑通之前，不扩展到 LLM、复杂 Transformer、对比学习或跨域泛化。
