"""从通过独立核验的 V3 产物生成中文报告和可上传的汇总文件。"""

import json
from pathlib import Path
import shutil

import numpy as np


def main():
    """不训练、不选模型，只读取正式验证汇总；保留负结果与解释限制。"""
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/experiments/v3_validation"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    qa = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert qa["status"] == "passed" and summary["completed_runs"] == 12
    rows = summary["rows"]
    by_name = {row["variant"]: row for row in rows}
    comparisons = {row["reference"]: row for row in summary["paired_comparisons"]}
    main_model = by_name["b3_attention_skip"]
    best = max(rows, key=lambda row: row["macro_f1_mean"])
    previous_summary = json.loads((root / "artifacts/experiments/v2_validation/summary.json").read_text(encoding="utf-8"))
    previous_mean = next(row for row in previous_summary["rows"] if row["variant"] == "mean_plain")
    seed_list = (42, 123, 2026)
    lines = ["# V3 实验结果：三源特征自注意力与跳跃拼接", "",
             "日期：2026-10-09；协议：Raw；本轮只训练/验证，未评估测试集。", "",
             f"已完成 4 个模型 × 3 个种子，共 12 次完整训练、{summary['total_epochs']} 个 epoch。"
             f"累计训练时间 {summary['total_train_seconds'] / 60:.2f} 分钟（不含短流程、单测和核验）。",
             "结构和旧代码合计 17 项测试通过；两次同种子两轮真实训练的日志、权重、验证概率完全相同。",
             "12 个最优 checkpoint 的验证概率逐样本重载复现，独立混淆矩阵复算指标全部一致。"
             "三个 B0 均完全复现 V1 concat 的训练轨迹和验证预测（耗时除外）。", "",
             "## 1. 主要结果", "",
             "均值±样本标准差（ddof=1），指标单位 %；参数量为可训练参数。", "",
             "| 模型 | Macro-F1 | Macro-Recall | Accuracy | 参数量 | CPU中位延迟/ms |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['variant'].split('_')[0].upper()} {row['description']} | "
                     f"{row['macro_f1_mean']*100:.2f}±{row['macro_f1_std']*100:.2f} | "
                     f"{row['macro_recall_mean']*100:.2f}±{row['macro_recall_std']*100:.2f} | "
                     f"{row['accuracy_mean']*100:.2f}±{row['accuracy_std']*100:.2f} | "
                     f"{row['parameters']:,} | {row['latency_median_ms_mean']:.3f} |")
    lines += ["", "CPU 延迟来自同机、两线程、单样本前向，各 checkpoint 预热10次/测量100次；"
              "不是工业部署或跨硬件性能结论。B1/B3 参数相近，但计算结构不同。", "",
              "| 模型 | seed42 | seed123 | seed2026 | 最优epoch（对应三种子） |", "|---|---:|---:|---:|---|"]
    for row in rows:
        values = " | ".join(f"{row['macro_f1_by_seed'][str(seed)]*100:.2f}" for seed in seed_list)
        lines.append(f"| {row['variant']} | {values} | {row['best_epochs']} |")
    lines += ["", "## 2. 对照结论", "",
              f"本轮验证均值最高的是 **{best['description']}，Macro-F1 {best['macro_f1_mean']*100:.2f}%**。", "",
              "| 比较 | seed42差值/百分点 | seed123 | seed2026 | 平均差值 | B3胜出次数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary["paired_comparisons"]:
        values = " | ".join(f"{row['differences_by_seed'][str(seed)]*100:+.2f}" for seed in seed_list)
        lines.append(f"| B3 − {row['reference']} | {values} | {row['mean_difference']*100:+.2f} | {row['wins']}/3 |")
    mlp_compare = comparisons["b1_mlp_skip"]
    base_compare = comparisons["b0_concat"]
    if mlp_compare["mean_difference"] > 0 and base_compare["mean_difference"] > 0:
        judgement = "B3 的平均得分高于原始拼接和参数匹配 MLP，支持它作为后续候选。"
    else:
        judgement = "本轮未同时证明 B3 优于原始拼接和参数匹配 MLP，不能据此宣称注意力融合更优。"
    lines += ["", judgement,
              f"B3 相对 B1 平均差值 {mlp_compare['mean_difference']*100:+.2f} 个百分点，"
              f"在 {mlp_compare['wins']}/3 个种子中胜出。三个重复来自同一划分，不能据此声称统计显著。",
              "B3/B0 同时改变模型大小；B3/B2 同时改变跳跃路径和分类头大小。"
              "这两个比较支持整体结构评估，不足以单独证明某个机制的因果收益。", "",
              "B3 三个种子的最优 checkpoint 均出现在第 1 轮，后续训练没有超过初轮验证 Macro-F1。"
              "这提示需检查优化过程、类别加权与泛化行为；目前不足以确定具体原因，"
              "也不能从这一配置推广为所有注意力方案都无效。", "",
              f"历史参照：V2 等权平均在同一验证集上的 Macro-F1 为 "
              f"{previous_mean['macro_f1_mean']*100:.2f}±{previous_mean['macro_f1_std']*100:.2f}%。"
              f"B3 相对该均值为 {(main_model['macro_f1_mean']-previous_mean['macro_f1_mean'])*100:+.2f} 个百分点。"
              "这是已完成的 V2 验证结果，本轮没有重新训练等权模型，也不属于参数匹配对照。", "",
              "## 3. 正常与直流故障", "",
              "各项均为三个种子的均值±标准差，单位 %。", "",
              "| 模型 | 正常Recall | DC F1 | DC Precision | DC Recall |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        values = " | ".join(f"{row[key+'_mean']*100:.2f}±{row[key+'_std']*100:.2f}"
                            for key in ("normal_recall", "dc_f1", "dc_precision", "dc_recall"))
        lines.append(f"| {row['variant']} | {values} |")
    labels = ("正常", "AC短路", "弱电网", "断相", "电网异常", "谐波", "DC故障", "IGBT")
    lines += ["", "| 类别F1均值 | B0 | B1 | B2 | B3 |", "|---|---:|---:|---:|---:|"]
    for label, name in enumerate(labels):
        values = " | ".join(f"{row['per_class_f1_mean'][label]*100:.2f}" for row in rows)
        lines.append(f"| {name} | {values} |")
    lines += ["", "相对 B0，B3 的退化还包括电网异常和 IGBT 类，不能把总得分下降全部归因于正常/DC 混淆。"]
    lines += ["", "## 4. 注意力诊断", "",
              "每个样本的矩阵为 [4 heads,3 queries,3 keys]，query/key 顺序均为 current、dc、rms。"
              "对角注意力保留；每行非负且和为1，已独立重载复现。", "",
              "| 模型 | 平均注意力行熵 | 样本间最大权重标准差 |", "|---|---:|---:|"]
    attention_reports = {}
    for variant in ("b2_attention", "b3_attention_skip"):
        group = [json.loads((output / f"{variant}_seed{seed}" / "attention_summary.json").read_text(encoding="utf-8"))
                 for seed in seed_list]
        entropy = float(np.mean([r["mean_entropy_by_head_query"] for r in group]))
        deviation = float(np.max([r["std_across_samples_by_head"] for r in group]))
        lines.append(f"| {variant} | {entropy:.4f} | {deviation:.4f} |")
        attention_reports[variant] = {str(seed): r for seed, r in zip(seed_list, group)}
    lines += ["", "三个 key 的均匀注意力行熵为 ln(3)=1.0986。行熵和标准差只描述网络行为，"
              "不是故障机理、物理因果或传感器可靠性的证据；不能单独证明注意力是性能变化原因。", "",
              "## 5. 下一步与限制", ""]
    if mlp_compare["mean_difference"] > 0 and base_compare["mean_difference"] > 0:
        lines.append("暂保留 B3 为候选，先结合类别混淆及种子差异评审是否值得进入后续验证；"
                     "本轮不追加模型层数、改损失或调参，也不自动评估 Test。")
    else:
        lines.append("保留全部负结果；先评审正常/直流恢复段标签和类别加权损失，再决定后续方案。"
                     "不根据这轮结果继续堆注意力层。")
    lines += ["同一事件划分上的多个训练种子只反映优化随机性。此前 V1 已查看过同一 Test 的结果，"
              "后续即使再评估它，也不能声称整个研究拥有完全未接触的终评集。"
              "更强的泛化结论需要独立事件划分或额外数据。当前仅有单设备 HIL 数据。", "",
              "## 6. 文件、复现与存档", "",
              "方案：`V3_PLAN.md`。冻结代码提交见正式目录 `version.json`；"
              "逐运行数据、最优模型、日志和代码 SHA256 见 `experiment.json` 与 `source_snapshot/`。",
              "本地完整产物：`artifacts/experiments/v3_validation/`；GitHub 汇总：`reports/v3/`。"
              "原始数据、样本、权重和逐样本预测均未上传。", "",
              "```powershell", "# 已完成的正式训练会跳过；源码或计划变化时拒绝混用。",
              "& $uvPath run python -m converter.experiments_v3",
              "& $uvPath run python -m converter.validate_v3",
              "& $uvPath run python scripts/report_v3.py", "```", ""]
    (root / "V3_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    reports = root / "reports/v3"
    reports.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "summary.csv", "validation_report.json", "plan.json", "version.json"):
        shutil.copy2(output / name, reports / name)
    shutil.copy2(root / "artifacts/checks/v3_determinism/validation_report.json", reports / "determinism_report.json")
    shutil.copy2(root / "artifacts/checks/v3_data_audit.json", reports / "data_audit.json")
    (reports / "attention_summary.json").write_text(json.dumps(attention_reports, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(root / "V3_FINDINGS.md"), "completed_runs": 12,
                      "best_variant": best["variant"], "best_macro_f1": best["macro_f1_mean"],
                      "b3_macro_f1": main_model["macro_f1_mean"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
