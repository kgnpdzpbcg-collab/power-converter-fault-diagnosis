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
