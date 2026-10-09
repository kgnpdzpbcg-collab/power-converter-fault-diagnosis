# 当前协议下的Random Forest与CNN对照

## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-09
- Verification Status: VERIFIED（论文身份和当前协议）；UNVERIFIED（原论文RF具体超参数）
- Version Label: rf_baselines_v1

## 目标与范围

用户指定RF-Current、RF-All、CNN-Mean三个模型，用相同八类标签、事件划分和Macro-F1，
检查复杂神经网络是否必要，以及Normal/DC混淆是否也出现在传统模型中。
不新增标签重组、UMAP、其他模型、超参数搜索或Test评估。

原论文：García-Campos等，
[Hybrid Fault-Space Restructuring for Machine Learning-Based Fault Diagnosis in Power Electronic Converters](https://doi.org/10.3390/electronics15143029)，
Electronics 2026，15(14)，3029。
关联数据：[Zenodo 20484338](https://zenodo.org/records/20484338)。
已核实原文使用RF，其标签重组和80/20协议与当前任务不同，原文分数不能直接比较。
出版方页面访问受限，尚未核实RF树数、深度等超参数。本轮是RF算法在现有协议下的对照，
不是原文全部流程或超参数的精确复现。下面参数在看到本轮结果前固定。

## 固定数据与模型

沿用 `artifacts/prepared/v0/` Raw协议，Train39,687、Val9,298；原Test隔离。
所有模型使用共同三源匹配样本池，RF-Current也不额外使用未匹配的电流记录。
复用CNN的训练集median/IQR和clip10，不重拟合统计量；仅使用current、dc、rms，不使用元数据。

| 模型 | 输入与操作 | 实际训练 |
|---|---|---|
| RF-Current | current[3,20]按原顺序展平为60维 | seed42/123/2026，各100树 |
| RF-All | 展平电流60＋DC40＋RMS3，按该顺序拼接103维 | seed42/123/2026，各100树 |
| CNN-Mean | 原三分支编码＋64维等权平均＋八类分类头 | 复用已完成的三份基线checkpoint，重新核验验证预测 |

RF明确使用scikit-learn 1.9.1：n_estimators100、criterion=gini、max_features=sqrt、
max_depth=None、min_samples_split2、min_samples_leaf1、bootstrap=True、n_jobs1。
不在Val上选参数，不用OOB选模型。按用户要求只有两种RF模型，共6次新拟合。

RF的class_weight使用原CNN训练集的逆频数权重字典，按八类逐项复用。
这控制类别成本设置，不代表RF的Gini目标与CNN交叉熵完全相同。
RF-All与CNN-Mean比较的是整个分类方案，不能只把差值归因于CNN表示学习。
所有种子共享同一划分，标准差描述模型随机性，不是独立划分置信区间。

## 执行与验证

执行前冻结代码、参数、数据指纹和CNN参照预测/checkpoint指纹，并保存源码快照。
数据审计必须保持passed、事件/来源行号跨集合重复为0；输入列只从共享Dataset的测量输入字典构造。
新增依赖仅scikit-learn及其必要依赖，uv锁定版本；旧converter/candidate源码不修改。

正式入口：`.venv\Scripts\python.exe -m baselines.rf run`。
完整结果：`artifacts/experiments/rf_baselines/`，目录存在时拒绝覆盖。
运行失败先报告，不自动改变参数或追加实验。进程运行与输出进度需监控，硬超时30分钟。

核验入口：`.venv\Scripts\python.exe -m baselines.rf validate`。
6个RF模型保存后重载，预测概率和类别必须精确一致；用sklearn独立重算固定八类指标。
CNN三份checkpoint重载预测必须精确复现已保存结果，CNN本轮不重新训练或选择checkpoint。
核对全部样本编号、标签、特征指纹、原划分和数据文件SHA256。

报告均值±三个种子样本标准差、逐种子结果、accuracy、正常误报、DC精确率/召回率/F1、
电网异常指标和混淆矩阵。主要结论看Macro-F1，不能只看多数正常类驱动的accuracy。
本轮仍是同批次事件开发验证，不能证明跨批次泛化，也不能宣称某类数据绝对不可分。

代码与汇总归档分支 `codex/rf-baselines`；原始数据、权重和逐样本预测仅保存在本地。
最终分析 `RF_BASELINE_FINDINGS.md`，可归档汇总 `reports/rf_baselines/`。

