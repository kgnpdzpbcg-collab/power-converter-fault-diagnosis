# Git 版本与 GitHub 存档

本项目以远端已有 `main` 历史为起点，工作分支为 `codex/v3-attention-skip`。
原有 `docs/V0_EXPERIMENT_PLAN.md` 被完整保留。旧模型、训练器和数据代码未为 V3 重构。

本机 Git Credential Manager 没有可用于该仓库的命令行凭据，因此普通 `git push` 失败。
已连接的 GitHub 接口具有该仓库写入权限，使用 Git 数据 API 按阶段建立归档提交。
提交者和时间来自接口，故远端提交 SHA 与原始本地提交不同；每个阶段的 Git **文件树 SHA 完全一致**。
这表示文件名、内容和模式相同，而非只比较文件数量或摘要文本。

| 阶段 | 原始本地提交 | GitHub归档提交 | 相同文件树 |
|---|---|---|---|
| 远端早期 V0 | 901c4bfe657e4a3be3e2212c587efd6a7366c7dd | 同左 | 远端原历史 |
| V1/V2代码归档 | ffae1b7df4e973ad146308255cc1c707baf714ef | 00a5572907a48dffc464dd7f6d37c61fcaebd114 | a766160236b5c046725928e82ed896bfd5e7353e |
| V3冻结方案与代码 | 6a761163d7db2b624408567391e97b400fc225c2 | 44af3e16f0206aef3268c931e1fd2aa0ea93d142 | 667e8dc0166b6aa45f6b7d18e995fa7ad75201ff |

正式训练 `version.json` 记录原始本地 V3 冻结提交，`source_snapshot/manifest.json` 记录实际训练源码的字节 SHA256。
本地标签 `archive/v2-validation`、`experiments/v3-frozen` 保留原始阶段，结果阶段另保存 `experiments/v3-results`。
GitHub 按相同顺序保留代码、方案和结果历史；最终阶段通过拉取远端并比较完整文件树进行核验。
分支在文件树一致、工作区干净的条件下对齐远端归档历史，以便后续正常跟踪；原始本地提交仍可通过阶段标签恢复。

远端不包含原始数据、预处理样本、模型权重、逐样本预测或虚拟环境。
`reports/v3/` 仅含汇总、实验矩阵、核验摘要与注意力描述统计。
本地完整结果保留在 `artifacts/`，其 Git 忽略规则并不意味着这些实验没有完成。

V3 不直接覆盖 `main`，不强制更新任何远端分支，也没有自动合并。
若以后配置命令行凭据，可在对齐后的分支上正常提交和推送。

## 细标签/频域诊断阶段

在 `codex/fine-frequency-diagnostics` 独立存档，原V3/V4分支继续保留。
本轮不修改旧converter代码、原数据或实验快照，不上传逐样本特征和预测。

| 阶段 | 原始本地提交 | GitHub归档提交 | 相同文件树 |
|---|---|---|---|
| 诊断冻结方案与代码 | f97081c | f9e1d80eec2b411947d41087cdc5cb65d2a8cf43 | 6e7efab168cf0b1eeec93455ea635a2f3bf86fcd |

结果阶段保存 `diagnostics-feasibility-results` 本地标签，再由同一GitHub接口归档报告、汇总及导出脚本。
完整运行证据留在 `artifacts/diagnostics/feasibility_v1/`；其 `plan.json` 指向执行前冻结提交。

## 当前划分审计与简化多尺度候选

2026-10-09，独立分支 `codex/multiscale-candidate`。旧划分、数据及converter源码不变。
依最新要求简化为原等权平均、普通宽卷积、多尺度残差各三个种子，共9次训练。
此前 `NEXT_EXPERIMENT_PLAN.md` 的R/F及四损失矩阵仍未执行。

| 阶段 | 原始本地提交 | GitHub归档提交 | 相同文件树 |
|---|---|---|---|
| 候选冻结方案与代码 | fb1e3cfd836e18142d3835b94affd264fc7bb24b | d9ffc754c99dc2957bee4949032e9e36fe33eced | 8cc6e1aef5448fd7c0f535c7bc6d9853bcbe62ea |

训练 `version.json` 指向上述冻结提交；冻结标签 `candidate-multiscale-frozen`。
结果阶段标签 `candidate-multiscale-results` 保存审计、报告及 `reports/candidate/`。
本地回执保存在 `artifacts/checks/current_split_20261009/archive_receipt.json`，
记录冻结阶段相同文件树SHA、结果本地提交及上传状态；回执不进入自身提交，避免循环引用。
源码冻结已通过GitHub接口归档，未覆盖main、未自动合并。
结果阶段前次两次创建远端Git文件树均遇到接口网络传输错误，未更新远端分支，
结果和报告完整保留在本地Git。用户追加同步要求后，按同一分支恢复结果归档，
包含 `CANDIDATE_FINDINGS.md`、`reports/candidate/`和状态文档。
本次本地同步提交保留在标签 `candidate-multiscale-results-sync`；具体远端提交和相同文件树SHA见本地回执。
本地完整checkpoint和逐样本预测位于 `artifacts/experiments/multiscale_candidate/`，不上传。

## 固定八类RF/CNN对照阶段

独立分支 `codex/rf-baselines`，以多尺度结果分支归档提交cbecb55为起点。
旧converter/candidate源码、数据、事件划分和训练统计不变；新增scikit-learn1.9.1与必要依赖并锁定版本。
原论文具体RF超参数尚未核实，本轮为预先固定100树设置的协议内RF对照。

| 阶段 | 原始本地提交 | GitHub归档提交 | 相同文件树 |
|---|---|---|---|
| RF冻结方案与代码 | 4c180b300c42e1b648ea5d990d6cf41272f32cc5 | cc4b7bda7bcb13584ff58db912e5cb4e8bb8d6b3 | fad7519231582820d901a2be41203915f3fb20dc |

冻结标签 `rf-baselines-frozen`；结果标签 `rf-baselines-results`。
结果报告为 `RF_BASELINE_FINDINGS.md`，汇总和核验在 `reports/rf_baselines/`。
结果归档使用同一分支，完整文件树核验回执保存到 `artifacts/checks/rf_baselines_archive_receipt.json`。
本轮6次RF新拟合、3份CNN参照重载，完整模型和逐样本预测只保存在本地，不上传数据。
