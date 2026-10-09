"""将已核验诊断导出为适合Git存档的汇总，不重拟合模型或读取Test波形。"""

import csv
import json
from pathlib import Path
import shutil

import numpy as np


def write_csv(path, rows):
    """保留实际精度与中文列值；UTF-8 BOM便于Windows表格软件打开。"""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export(root):
    """来源仅为本轮Train/Val缓存与已保存结果，汇总种子不视为独立数据重复。"""
    source = root / "artifacts/diagnostics/feasibility_v1"
    output = root / "reports/feasibility"
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((source / "label_audit.json").read_text(encoding="utf-8"))
    verification = json.loads((source / "validation_report.json").read_text(encoding="utf-8"))
    assert verification["status"] == "passed"
    for name in ("summary.csv", "plan.json", "validation_report.json"):
        shutil.copy2(source / name, output / name)
    write_csv(output / "fine_support.csv", audit["rows"])
    write_csv(output / "injection_batches.csv", audit["injection_batches"])

    # 相同表示的三个冻结编码器种子分组；原始输入没有编码器种子。
    groups = {}
    for row in summary["rows"]:
        if row["target"] == "fine48":
            representation = row["representation"].split("_seed")[0]
            groups.setdefault((row["view"], representation, row["method"]), []).append(row)
    overall, families, per_class, errors = [], [], [], []
    for (view, representation, method), rows in groups.items():
        sample_f1 = np.array([row["metrics"]["sample"]["macro_f1"] for row in rows])
        overall.append(dict(view=view, representation=representation, method=method, seeds=len(rows),
                            macro_f1_mean=float(sample_f1.mean()),
                            macro_f1_sample_std=float(sample_f1.std(ddof=1)) if len(rows) > 1 else "",
                            event_macro_f1_mean=float(np.mean([row["metrics"]["event_balanced"]["macro_f1"] for row in rows]))))
        for index, family in enumerate(rows[0]["within_family"]):
            values = np.array([row["within_family"][index]["metrics"]["sample"]["macro_f1"] for row in rows])
            families.append(dict(view=view, representation=representation, method=method, seeds=len(rows),
                                 coarse=family["coarse"], name=family["name"], fine_labels="/".join(map(str, family["fine_labels"])),
                                 oracle_coarse_label=True, macro_f1_mean=float(values.mean()),
                                 macro_f1_sample_std=float(values.std(ddof=1)) if len(rows) > 1 else ""))
            if method == "prototype":
                # 平均错误数并非新增验证样本；各种子使用同一份验证标签。
                matrices = np.array([row["within_family"][index]["metrics"]["sample"]["confusion_matrix"] for row in rows])
                cm = matrices.mean(0)
                candidates = [(cm[i, j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
                for count, i, j in sorted(candidates, reverse=True)[:5]:
                    errors.append(dict(view=view, representation=representation, method=method, coarse=family["coarse"],
                                       oracle_coarse_label=True, true_fine=family["fine_labels"][i], predicted_fine=family["fine_labels"][j],
                                       mean_error_samples_per_seed=float(count), support_per_seed=float(cm[i].sum()),
                                       mean_error_rate=float(count / cm[i].sum()), seeds=len(rows)))
        for label in range(48):
            per_class.append(dict(view=view, representation=representation, method=method, fine_label=label, seeds=len(rows),
                                  f1_mean=float(np.mean([row["metrics"]["sample"]["per_class_f1"][label] for row in rows])),
                                  recall_mean=float(np.mean([row["metrics"]["sample"]["per_class_recall"][label] for row in rows])),
                                  val_support_per_seed=rows[0]["metrics"]["sample"]["support"][label]))
    write_csv(output / "fine_overall.csv", overall)
    write_csv(output / "within_family.csv", families)
    write_csv(output / "fine_class_scores.csv", per_class)
    write_csv(output / "fine_error_pairs.csv", errors)
    comparisons = []
    for row in summary["paired_frequency_comparisons"]:
        comparisons.append({key: row[key] for key in ("target", "view", "reference", "variant", "method", "repeats", "mean_difference")}
                           | dict(ci95_low=row["percentile_95_interval"][0], ci95_high=row["percentile_95_interval"][1]))
    write_csv(output / "frequency_comparisons.csv", comparisons)
    distributions = json.loads((source / "spectral_distributions.json").read_text(encoding="utf-8"))
    spectral_rows = []
    for row in distributions["rows"]:
        mean = np.asarray(row["mean_power_share"])
        for channel in range(3):
            spectral_rows.append(dict(split=row["split"], fine_label=row["fine_label"], candidate=row["candidate"],
                                      samples=row["samples"], current_channel=channel,
                                      mean_energy_bins_2_to_10=float(1 - mean[channel, 0]),
                                      median_bin1_share=row["median_power_share"][channel][0],
                                      bin1_q25=row["q25_power_share"][channel][0], bin1_q75=row["q75_power_share"][channel][0]))
    write_csv(output / "spectral_energy.csv", spectral_rows)

    # 检查正常标签在事件中的位置：初始基线与故障后恢复均保留fine=0。
    with np.load(source / "cache/meta.npz", allow_pickle=False) as meta:
        counts = dict(initial_normal_samples=0, recovery_normal_samples=0, normal_inside_fault_span=0)
        for block in np.unique(meta["block"]):
            positions = np.flatnonzero(meta["block"] == block)
            fault = positions[meta["fine_label"][positions] > 0]
            normal = positions[meta["fine_label"][positions] == 0]
            assert len(np.unique(meta["fine_label"][fault])) == 1
            t = meta["timestamp"][normal]
            counts["initial_normal_samples"] += int((t < meta["timestamp"][fault].min()).sum())
            counts["recovery_normal_samples"] += int((t > meta["timestamp"][fault].max()).sum())
            counts["normal_inside_fault_span"] += int(((t >= meta["timestamp"][fault].min()) & (t <= meta["timestamp"][fault].max())).sum())
        assert sum(counts.values()) == int((meta["fine_label"] == 0).sum())
        assert counts["normal_inside_fault_span"] == 0
    scope = {"probe_fits": summary["probe_runs"], "views": summary["views"],
             "time_only_fault47": {"scope": audit["time_only_fault47"]["scope"],
                                    "macro_f1": audit["time_only_fault47"]["metrics"]["sample"]["macro_f1"],
                                    "accuracy": audit["time_only_fault47"]["metrics"]["sample"]["accuracy"]},
             "normal_label_position_counts": counts, "fine48_missing_train_or_val_classes": [],
             "new_encoder_training": False, "test_evaluation": False}
    assert all(not row["missing_train_classes"] and not row["missing_val_classes"] for row in summary["rows"])
    (output / "scope_audit.json").write_text(json.dumps(scope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "files": len(list(output.iterdir())), "normal_label_audit": counts}, ensure_ascii=False))


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[1])
