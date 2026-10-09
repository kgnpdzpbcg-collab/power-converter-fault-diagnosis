# 项目当前状态与续接入口

更新：2026-10-09；当前完成阶段：现有划分泄露审计与9次多尺度候选训练，全部核验通过。

## 当前最新完成阶段

用户最新要求：简化问题，先检查当前划分及标签泄露，无问题则实现候选并跑实验。
实际执行 `CANDIDATE_RUN_PLAN.md`，结果先读 `CANDIDATE_FINDINGS.md`。
原始JSON完整重建一致；事件、三源行号及完整联合输入跨集合精确重复为0；
标签/编号等不进入forward，归一化与类别权重仅由Train计算，未发现直接泄露。
随机事件划分仍共享注入批次，不能证明跨批次或跨工况泛化。

沿用Raw原划分及加权CE，baseline/wide/multiscale各3种子，共9次，227轮。
验证Macro-F1：88.47±0.87% / 88.40±0.89% / 87.09±0.48%。
多尺度残差候选三个种子均落后基线，平均低1.38个百分点；保留原等权平均作为最好基线。
候选DC召回100%伴随DC精确率29.23%、正常召回78.30%，整体未改善。
29项测试、9份checkpoint重载/指标重算核验通过；baseline三种子概率精确复现历史V2。

当前分支 `codex/multiscale-candidate`；冻结标签 `candidate-multiscale-frozen`，
结果标签 `candidate-multiscale-results`。源码新增在 `candidate/`，历史 `converter/`、
原始/预处理数据、划分、归一化和旧快照均未修改。
完整产物 `artifacts/experiments/multiscale_candidate/`，可存档汇总 `reports/candidate/`。
本轮未新增Test预测；Test只在重建和排重审计时被检查。
本轮已结束，不自动追加损失、模块或新划分以寻找正结果。

### 此前下一轮提案（未执行）

用户最新要求为“据新讨论稿细化下一轮实验方案”。已完成 `NEXT_EXPERIMENT_PLAN.md`，未开始新训练。
方案修改优先级：先协议/八类时间对照和四损失比较，再有条件研究多尺度残差编码。
时间覆盖核查确认：IGBT最后block1768，交流短路最早block1770，不存在训练/未来评估均覆盖八类的全局切点。
拟议R/F为批内随机/前向60/20/20，使用原Train/Val池，旧Test仍隔离；应新建manifest并重拟合训练归一化。
R/F三角色覆盖八类，事件数匹配，原始三源行号无跨角色重复；见 `reports/planning/next_protocol_audit.json`。
该方案保留在 `codex/next-experiment-plan`；原诊断分支保留。
最新用户要求简化后，本轮未执行上述R/F及四损失矩阵。

### 已完成的上一轮诊断

用户授权本轮客观评估讨论稿并执行两项检查，不是实现新层次监督或时频神经网络。
执行协议见 `FEASIBILITY_CHECK_PLAN.md`，最终结果先读 `FEASIBILITY_FINDINGS.md`。
132个简单诊断探针拟合、参数重拟合与预测重放、24组事件bootstrap和24项测试全部通过。
数据、划分、归一化、历史checkpoint与converter源码均未改变；只生成Train/Val特征。

全48类Raw冻结均值/事件原型Macro-F1为61.55±0.68%，原始输入原型65.62%。
细类可辨识性不均匀：交流短路/IGBT强，谐波弱，相断开混淆，DC早晚子集下降。
仅时间的已知故障47类对照Macro-F1为97.38%，说明标签/注入批次高度绑定，工况混杂尚未排除。
粗类谱形有判别信息，但三源强近邻添加谱形仅Raw小增益，非候选/早晚不稳定。
上一轮报告曾建议优先做轻量频域增强，尚未实现；根据新讨论稿与用户要求，
最新方案调整为先验证批内时间敏感性与损失，再研究轻量多尺度编码；统一48类辅助监督继续暂缓。

完整诊断在 `artifacts/diagnostics/feasibility_v1/`；汇总在 `reports/feasibility/`。
诊断存档分支 `codex/fine-frequency-diagnostics`；本地冻结标签 `diagnostics-feasibility-frozen`。
`V4_DESIGN_PROPOSAL.md`仍只是此前残差候选讨论，不代表已执行或本轮最终路线。

## 已完成

1. uv/Python3.12 独立环境；数据流式读取、三源同步、八类映射、恢复段事件 block 和固定划分。
2. V1：48 次正式实验，包括融合对照、测量组合和极大值敏感性；见 `FINDINGS.md`。
3. V2：6 变体×3种子，训练/验证的分支归一化与初始门控对照；见 `V2_FINDINGS.md`。
4. V3：B0–B3×3种子，292轮，9.39分钟累计训练；见 `V3_PLAN.md` 和 `V3_FINDINGS.md`。
5. V3 的17项测试、两次两轮确定性检查、真实原始文件/训练统计审计，以及12次 checkpoint 预测重载核验均通过。

## V3 结论

验证 Macro-F1：B0原始拼接87.73±0.61%；B1普通MLP跳跃拼接85.57±2.69%；
B2仅注意力85.72±2.06%；B3注意力跳跃拼接84.47±1.46%。
B3相对B0低3.26个百分点，三个种子均落后；相对参数匹配B1低1.10个百分点。
不能宣称这轮注意力主模型优于简单融合。

本轮未评估 Test。此前 V1 已看过同一 Test，因此整个研究不能将该集合宣称为完全未接触的终评数据。
V2等权平均验证88.47±0.87%是历史参照，本轮未重跑，不属于参数匹配对照。

## 继续工作时先读

- `CANDIDATE_FINDINGS.md`：最新划分审计、9次训练结果及解释边界。
- `CANDIDATE_RUN_PLAN.md`：本轮冻结架构、对照和执行范围。
- `reports/candidate/validation_report.json`：本轮9份checkpoint核验状态。
- `V3_PLAN.md`：架构、矩阵、冻结配置、执行次序与解释边界。
- `V3_FINDINGS.md`：实际结果、逐种子差值、类别指标、注意力统计和建议。
- `docs/VERSION_ARCHIVE.md`：本地提交与 GitHub API 归档提交的对应关系。
- `artifacts/experiments/v3_validation/validation_report.json`：全部正式核验状态。
- `artifacts/experiments/v3_validation/source_snapshot/`：实际运行的源码和字节指纹。

## 后续建议与禁止混用

先与用户评审恢复段标签、加权损失和训练/验证脱节，再制定下一轮方案。
B3三个种子最佳均在第一轮；除了正常/DC，电网异常与IGBT表现也下降，原因尚未确定。
不自动增加注意力层，不根据已看到的验证结果修改并冒充本轮冻结实验。

数据、V1/V2结果和旧源码快照保留在本地，不覆盖、不重新划分。
GitHub仅归档代码、方案和汇总，完整模型和逐样本预测在 `artifacts/`。
修改 `converter/` 后，旧实验 runner 的源码一致性检查可能拒绝续跑；应使用其冻结快照审计，
新方案用新的目录，不能为通过检查而改旧快照或指纹。

用户授权的最新9次候选训练已完成；冻结源码已归档GitHub，结果已提交本地Git。
用户追加要求将Markdown结果报告同步至 `codex/multiscale-candidate`，同步内容包含
`CANDIDATE_FINDINGS.md`、`reports/candidate/`及状态文档。
前次接口网络错误作为历史记录保留；同步版本及完整文件树比对回执在本地审计目录。
尚未要求继续新实验或评估测试集。
