# 简化执行：检查原划分，验证多尺度残差候选

## Material Passport

- Origin Skill: academic-research-suite/experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-09
- Verification Status: VERIFIED（当前泄露审计）；UNVERIFIED（候选收益，执行前）
- Version Label: multiscale_candidate_v1

## 本轮范围

用户要求先检查现有划分及标签泄露，无问题则搭建修订候选并跑实验。
因此简化此前 `NEXT_EXPERIMENT_PLAN.md`：不执行新R/F划分和四损失矩阵。
原始数据、当前Train/Val/Test划分、训练归一化、逆频数加权CE和训练配置均沿用。
本轮只改电流/DC波形编码，RMS、等权平均和八类分类头保持原样。

## 数据检查结论

`artifacts/checks/current_split_20261009/leakage_audit.json` 已通过：

- 从三份原始JSON完整重建，prepared全部数组、标签/事件、清单、两套训练统计完全一致。
- 事件及三个来源行号跨集合重复均为0，原始/归一化完整三源向量跨集合精确重复均为0。
- 模型特征仅current、dc、rms，label是训练目标；label/fine_label/timestamp/block/sample_index不进入forward。
- median/IQR和类别权重仅取Train；原prepared文件SHA256未变。
- 对齐只按时间选最近邻，之后剔除2组标签不一致记录，属于建库一致性筛选；标签不生成输入特征。

未发现直接标签或记录泄露。当前随机事件划分共享注入批次，泛化解释限于该数据条件。
历史V1已观察Test；本轮只训练/验证，不新增Test模型预测。Test仅在排重/重建审计中被检查。

## 固定矩阵

| 变体 | 结构 | 参数量 |
|---|---|---:|
| baseline | 原两层k3 CNN＋RMS MLP＋等权平均 | 24,936 |
| wide | 两层k3，波形首层32→38，其余不变 | 27,366 |
| multiscale | 并行k3/k5＋输入残差，尾部k3 | 27,432 |

各用seed42、123、2026，共9次正式训练；不根据中间结果追加变体。
wide与候选参数差66（约0.24%），用于检查容量增加的影响。

多尺度波形编码：C→16的k3/BN/ReLU和k5/BN/ReLU并行；拼接32通道；
1×1投影32→32，加上C→32输入投影，ReLU；再经32→64的k3/BN/ReLU和全局池化。
各波形仍输出64维，与RMS64维等权平均，接原64→128→8分类头。
这个实验验证多尺度＋残差的整体组合，不分别归因两个机制，也不宣称其为全新通用算法。

先创建同种子原模型，再隔离随机数创建替换编码器。RMS和分类头初值严格一致，
候选兼容的尾部卷积/BN也沿用基线初值；训练Dropout随机数起点相同。

## 训练、核验与判断

复用原训练器：AdamW1e-3、weight_decay1e-4、batch256、CPU2线程；最多100轮，
验证Macro-F1早停15轮，ReduceLROnPlateau factor0.5/patience5。相同分数保留较早checkpoint。
数据划分不随训练seed改变；无新损失、无标签重标、无验证统计拟合。

正式训练前冻结Git提交、配置、候选/旧源码快照和数据指纹。
完成后恢复每份checkpoint，重算验证预测、最佳轮次、类别指标和原始日志一致性。
三次baseline应精确复现历史V2 mean_plain验证概率；若不一致，先定位，不能继续把它当等价基线。
训练/评估只创建Train/Val Dataset，所有结果明确为开发验证。

报告Macro-F1及三个种子标准差、逐种子候选相对baseline/wide差值、accuracy、
正常误报与DC precision/recall/F1、八类混淆、参数量/耗时。
优先判断候选是否同时优于旧基线与容量对照；只有提高的个别种子不能说明稳定改善。
不为了正结论选择种子或追加模型。

## 输出与复现

新代码放 `candidate/`，旧 `converter/` 文件不修改；所有代码包含中文说明。
完整结果：`artifacts/experiments/multiscale_candidate/`。
汇总与泄露审计存档：`reports/candidate/`；最终分析：`CANDIDATE_FINDINGS.md`。

```powershell
# 审计输出目录已存在，不重复覆盖；其完整结果可直接读取。
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m candidate.run
.venv\Scripts\python.exe -m candidate.validate
```

已有结果目录拒绝覆盖，执行失败先报告原因；训练和选模规则不根据结果修改。
