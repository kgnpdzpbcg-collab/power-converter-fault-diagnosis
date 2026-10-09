# RF-Current、RF-All与CNN-Mean的固定八分类对照

## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: run/validate
- Origin Date: 2026-10-09
- Verification Status: VERIFIED（当前协议的拟合、重载及指标）；UNVERIFIED（原论文具体RF超参数）
- Version Label: rf_baselines_v1

## 已完成与主要结论

按 [RF_BASELINE_PLAN.md](RF_BASELINE_PLAN.md) 完成RF-Current/RF-All各三个种子，共6次新RF拟合，
另复用并重新核验3份CNN-Mean基线checkpoint。31项测试、6个RF模型及3份CNN重载核验均通过。

**RF-All平均验证Macro-F1为89.02%，CNN-Mean为88.47%，RF-All高0.55个百分点。**
当前数据协议下，没有证据支持“必须使用CNN才能达到约88%的性能”。
RF-All与RF-Current的差值为3.45个百分点，支持三源测量的联合输入具有价值。
RF-All的Normal/DC混淆仍明显，不能将小幅总分优势解释为这个问题已经解决。
差值来自同一固定开发验证集，三种子不是三个独立划分；本轮没有显著性或跨批次泛化结论。

## 原论文与本轮的关系

数据来源与原论文已核实：[数据发布页](https://zenodo.org/records/20484338)，
[García-Campos等，Electronics 2026，15(14)，3029](https://doi.org/10.3390/electronics15143029)。
原文使用DT/RF并研究UMAP与聚类后的标签重组，原文80/20协议和目标类别与当前任务不同。
本轮只使用RF算法，不复刻原文标签重组，不拿原文分数与当前八分类直接比较。
原文RF树数、深度等具体超参数未能核实；以下参数在看到本轮结果前固定。
因此应称为**原论文所用RF算法在当前协议下的对照**，不能称原文超参数的精确复现。

## 完全一致的数据与评价协议

- 原三源匹配样本池、原八类映射、原事件block划分保持不变。
- Train39,687、Val9,298；RF-Current也仅使用同一匹配池，不使用额外单源记录。
- 输入沿用Raw协议的训练集median/IQR及clip10；不增加统计特征、FFT、UMAP或新标准化。
- RF-Current取current的60维；RF-All依次拼接current60、DC40、RMS3，共103维。
- 标签、细标签、时间、block、样本编号不进入输入；它们只用于目标和核验。
- RF逐项使用CNN的训练集逆频数类别权重；RF/CNN目标函数仍不同，差值不能单独归因于表示学习。
- 固定八类Macro-F1，未预测类别按0计入；并用sklearn独立复算确认指标一致。
- 原Test只存在于已有数据池中，本轮未创建其Dataset、特征或模型预测。

冻结RF设置：scikit-learn1.9.1，100树，Gini，max_features=sqrt，max_depth=None，
min_samples_split2，min_samples_leaf1，bootstrap=True，n_jobs1。
每种RF使用seed42、123、2026；无验证调参或OOB选模。
CNN使用此前按Val选出的最佳checkpoint，未重训练或重新选择模型。
六次RF累计拟合44.45秒，不包含全部预测、核验、准备及报告耗时。

## 三个模型的实际结果

均为**验证集**，单位为%；均值±三个训练种子的样本标准差。

| 模型 | Macro-F1 | Accuracy | 正常误报率 | DC精确率 | DC召回率 | DC F1 | 电网异常F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF-Current | 85.57±0.17 | 86.83±0.11 | 16.59±0.07 | 41.91±0.77 | 87.73±2.08 | 56.72±1.14 | 79.40±0.54 |
| RF-All | **89.02±0.18** | **90.68±0.08** | **11.89±0.10** | 42.50±0.07 | 91.38±1.26 | **58.01±0.28** | **79.66±0.32** |
| CNN-Mean | 88.47±0.87 | 88.40±3.05 | 14.58±6.10 | 55.46±36.84 | 77.69±34.54 | 51.95±3.13 | 72.16±3.02 |

| seed | RF-Current Macro-F1 | RF-All Macro-F1 | CNN-Mean Macro-F1 | RF-All−CNN（百分点） |
|---|---:|---:|---:|---:|
| 42 | 85.37 | 89.17 | 88.88 | +0.30 |
| 123 | 85.69 | 88.83 | 87.47 | +1.36 |
| 2026 | 85.64 | 89.06 | 89.06 | +0.005 |

RF-All原始未舍入值三个种子均略高，但seed2026仅高0.005个百分点，四舍五入后持平。
不能把它描述为三个种子均取得明显提升，或据此声称统计显著。
训练集重代入Macro-F1均值：RF-Current89.80%、RF-All92.26%；这些不是泛化指标。

## 哪些类别改善，哪些类别下降

| 类别 | RF-Current F1 | RF-All F1 | CNN-Mean F1 |
|---|---:|---:|---:|
| 正常 | 90.07 | **93.11** | 91.27 |
| 交流短路 | 75.98 | **100.00** | 99.83 |
| 弱电网 | 100.00 | 100.00 | 100.00 |
| 相断开 | 100.00 | 100.00 | 100.00 |
| 电网电气异常 | 79.40 | **79.66** | 72.16 |
| 谐波 | 97.99 | 97.99 | **98.48** |
| DC故障 | 56.72 | **58.01** | 51.95 |
| IGBT闭锁 | 84.37 | 83.39 | **94.06** |

RF-All相对CNN主要改善正常、DC和电网异常，但IGBT F1下降10.67个百分点。
两者对IGBT的召回接近，RF额外把正常记录误报为IGBT：三个种子合计79次，CNN为0。
因此RF不是各类别全面优于CNN；0.55个百分点的总分收益包含类别间的得失。
RF-Current交流短路F1只有75.98%，加入DC/RMS后RF-All达到100%，支持联合测量提供互补信息。
这两个输入同时增加，不能分别归因DC和RMS，也不能把高分解释为跨工况可靠性。

## Normal/DC是否解决

验证集每个模型均包含6,648条正常和375条DC。

| 模型/seed | 正常→DC | 正常→电网异常 | DC→正常 |
|---|---:|---:|---:|
| RF-All/42 | 470 | 273 | 27 |
| RF-All/123 | 458 | 289 | 36 |
| RF-All/2026 | 463 | 295 | 34 |
| CNN-Mean/42 | 628 | 392 | 16 |
| CNN-Mean/123 | 790 | 534 | 2 |
| CNN-Mean/2026 | 3 | 530 | 233 |

RF-All比CNN的种子间波动小，但每个种子仍将约460条正常记录判为DC。
DC精确率仅42.50%、F1仅58.01%，说明正常误报仍是实质问题。
电网异常的误报有所减少，Normal/DC尚未得到根本解决。

这支持把下一步研究重点放在正常/DC及恢复段定义、观测差异和分类目标，
不支持继续把增加CNN层数当作默认解决办法。
**但两个模型均有混淆，不等于已经证明类别不可分、标签错误或数据达到性能上限。**
具体原因仍须受控验证；本轮只确定了问题可跨模型家族出现。

## 对研究定位的影响

当前不能用“CNN平均性能优于传统方法”作为论文主张。
三源信息的价值有对照支持；神经网络对个别类别的优势也存在，但总体优势尚未建立。
若继续提出神经模型，需说明它针对哪个已确认问题，并证明相对RF-All的收益。
也不能因RF分数稍高就称其部署更轻量，本轮没有进行设备上的算力/存储对照。

## 核验、版本与产物

31项测试全部通过，新增测试核对60/103维列顺序、共同记录行顺序及标签/元数据隔离。
六个RF保存后重新加载，训练和验证指标复算一致，验证概率逐元素完全一致。
三份CNN checkpoint重载也精确复现历史验证概率；全部模型的样本编号与标签相同。
源码、原数据、划分、归一化及CNN参照文件指纹保持不变，未新增Test评估。

- 实验冻结标签：`rf-baselines-frozen`，本地提交 `4c180b300c42e1b648ea5d990d6cf41272f32cc5`。
- 新代码：`baselines/rf.py`；只新增scikit-learn依赖及锁文件条目，旧converter/candidate源码不变。
- 完整本地模型、预测、快照：`artifacts/experiments/rf_baselines/`。
- 汇总、类别指标、模型详情和核验：`reports/rf_baselines/`。
- 报告与源码归档分支：[`codex/rf-baselines`](https://github.com/kgnpdzpbcg-collab/power-converter-fault-diagnosis/tree/codex/rf-baselines)。
- 完整Git版本核验回执留在 `artifacts/checks/rf_baselines_archive_receipt.json`。

```powershell
.venv\Scripts\python.exe -m baselines.rf validate
```

