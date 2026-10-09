"""固定实验矩阵：先 V0，再融合对照、测量组合消融和异常敏感性。"""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .data import load_samples
from .metrics import classification_metrics
from .prepare import CLASS_NAMES, write_json
from .train import TrainConfig, train


ALL = ("current", "dc", "rms")
# 保留 V0 的 32 维 RMS；concat64 与后面三种融合共享 64 维编码器。
FUSION_SPECS = [("v0", "v0", ALL), ("concat64", "concat", ALL), ("mean", "mean", ALL),
                ("static", "static", ALL), ("dynamic", "dynamic", ALL)]
ABLATION_SPECS = [("current", "dynamic", ("current",)), ("dc", "dynamic", ("dc",)),
                  ("rms", "dynamic", ("rms",)), ("current_dc", "dynamic", ("current", "dc")),
                  ("current_rms", "dynamic", ("current", "rms")), ("dc_rms", "dynamic", ("dc", "rms"))]


def summarize(output_dir, plan, samples):
    """只汇总已完成的正式实验，按同一 seed 配对比较，不挑选最好种子。

    std 使用样本标准差（ddof=1）。同时报告原始模型在未标记测试子集的
    性能，避免 Raw/过滤协议的测试样本构成差异被误读为训练收益。
    """
    output_dir = Path(output_dir)
    runs = []
    missing = []
    for spec in plan["runs"]:
        path = output_dir / spec["id"] / "result.json"
        if not path.exists():
            missing.append(spec["id"])
        else:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") != "complete" or "test" not in result:
                raise ValueError(f"不完整的正式结果: {path}")
            runs.append({**result, "id": spec["id"], "model": spec["model"]})
    groups = {}
    for run in runs:
        groups.setdefault((run["protocol"], run["model"]), []).append(run)
    rows, per_class = [], []
    for (protocol, model), group in groups.items():
        row = {"protocol": protocol, "model": model, "seeds_completed": len(group),
               "parameters": group[0]["parameters"]}
        for key in ("macro_f1", "balanced_accuracy", "accuracy"):
            values = np.array([run["test"][key] for run in group])
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        row["latency_median_ms"] = float(np.mean([run["latency_cpu"]["median_ms"] for run in group]))
        rows.append(row)
        for index, name in enumerate(CLASS_NAMES):
            values = np.array([run["test"]["per_class_f1"][index] for run in group])
            per_class.append({"protocol": protocol, "model": model, "class": index, "class_name": name,
                              "f1_mean": float(values.mean()), "f1_std": float(values.std(ddof=1)) if len(values) > 1 else None})
    for filename, data in (("summary.csv", rows), ("per_class.csv", per_class)):
        if data:
            with (output_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    comparisons = {}
    for protocol in ("raw", "exclude_saturation"):
        group_map = {model: {run["seed"]: run for run in group} for (p, model), group in groups.items() if p == protocol}
        if "dynamic" in group_map and "concat64" in group_map:
            common_seeds = sorted(set(group_map["dynamic"]) & set(group_map["concat64"]))
            differences = [group_map["dynamic"][seed]["test"]["macro_f1"] - group_map["concat64"][seed]["test"]["macro_f1"] for seed in common_seeds]
            comparisons[protocol] = {"paired_seeds": common_seeds, "dynamic_minus_concat64_pp": [value * 100 for value in differences],
                                     "mean_difference_pp": float(np.mean(differences) * 100), "dynamic_wins": sum(value > 0 for value in differences)}
    common_pool = []
    for run in runs:
        if run["protocol"] != "raw" or run["model"] not in {spec[0] for spec in FUSION_SPECS}:
            continue
        with np.load(output_dir / run["id"] / "test_predictions.npz", allow_pickle=False) as pred:
            eligible = ~samples["saturation_candidate"][pred["sample_index"]]
            labels, predictions = pred["label"][eligible], pred["prediction"][eligible]
        metrics = classification_metrics(np.bincount(labels * 8 + predictions, minlength=64).reshape(8, 8))
        common_pool.append({"model": run["model"], "seed": run["seed"], "samples": int(eligible.sum()), "metrics": metrics})
    baselines = {}
    for protocol in ("raw", "exclude_saturation"):
        eligible = np.ones(len(samples["label"]), dtype=bool) if protocol == "raw" else ~samples["saturation_candidate"]
        training = samples["label"][(samples["split"] == 0) & eligible]
        majority = int(np.bincount(training, minlength=8).argmax())
        test = samples["label"][(samples["split"] == 2) & eligible]
        confusion = np.zeros((8, 8), dtype=np.int64)
        confusion[:, majority] = np.bincount(test, minlength=8)
        baselines[protocol] = classification_metrics(confusion)
    summary = {"planned_runs": len(plan["runs"]), "completed_runs": len(runs), "missing_runs": missing,
               "rows": rows, "paired_fusion_comparisons": comparisons, "majority_baseline": baselines,
               "raw_models_on_common_non_candidate_test": common_pool,
               "limits": ["单设备单实验的事件划分，不能验证跨设备/跨工况泛化", "三个训练 seed 不代表三个独立数据集",
                          "极大值原因未确认；过滤改变谐波类构成，阈值不能视为完整物理清洗",
                          "门控权重是模型行为统计，不代表测量的因果贡献"]}
    write_json(output_dir / "summary.json", summary)
    lines = ["# 首轮实验结果", "", f"已完成 {len(runs)}/{len(plan['runs'])} 个正式训练。划分 seed=42；训练 seed={plan['seeds']}。",
             "所有模型用验证集 Macro-F1 选 checkpoint，训练结束后评估测试集；表中为三种子均值 ± 样本标准差。", "",
             "| 数据协议 | 模型 | Macro-F1 (%) | Balanced Acc (%) | 参数量 |", "|---|---|---:|---:|---:|"]
    for row in rows:
        f1_std = f"{row['macro_f1_std'] * 100:.2f}" if row["macro_f1_std"] is not None else "待补"
        lines.append(f"| {row['protocol']} | {row['model']} | {row['macro_f1_mean'] * 100:.2f} ± {f1_std} | {row['balanced_accuracy_mean'] * 100:.2f} | {row['parameters']:,} |")
    lines += ["", "V0 使用 RMS 32 维；concat64、mean、static、dynamic 使用完全相同的 64 维编码器。单/双源消融使用 dynamic 结构；单源时不设无效门控。", ""]
    for protocol, comparison in comparisons.items():
        lines.append(f"{protocol}：动态门控相对匹配维度的拼接，Macro-F1 平均差 {comparison['mean_difference_pp']:+.2f} 个百分点，{comparison['dynamic_wins']}/{len(comparison['paired_seeds'])} 个种子更高。")
    lines += ["", "## 解释边界", ""] + [f"- {limit}" for limit in summary["limits"]]
    lines += ["", "各类 F1 见 per_class.csv；完整混淆矩阵、权重分布和 checkpoint 在各实验子目录。summary.json 还包含多数类基线，以及原始模型在相同未标记测试子集上的结果。", ""]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = TrainConfig(max_epochs=args.max_epochs, threads=args.threads)
    runs = []
    for protocol, specs in (("raw", FUSION_SPECS + ABLATION_SPECS), ("exclude_saturation", FUSION_SPECS)):
        for name, fusion, branches in specs:
            for seed in args.seeds:
                runs.append({"id": f"{protocol}_{name}_s{seed}", "model": name, "fusion": fusion,
                             "branches": list(branches), "protocol": protocol, "seed": seed})
    plan = {"seeds": args.seeds, "training": asdict(config), "runs": runs,
            "selection": "validation Macro-F1; early stopping 15; test once after best checkpoint reload",
            "order": "V0→matched fusion comparisons→measurement combinations→sensitivity"}
    write_json(args.output_dir / "plan.json", plan)
    samples = load_samples(args.prepared_dir)
    start = time.perf_counter()
    for index, run in enumerate(runs, start=1):
        print(f"启动正式实验 {index}/{len(runs)}: {run['id']}", flush=True)
        config = TrainConfig(seed=run["seed"], max_epochs=args.max_epochs, threads=args.threads)
        train(args.prepared_dir, args.output_dir / run["id"], run["fusion"], tuple(run["branches"]), run["protocol"], config, samples)
        write_json(args.output_dir / "progress.json", {"completed": index, "planned": len(runs),
                                                      "last_completed": run["id"], "elapsed_seconds": time.perf_counter() - start})
        summarize(args.output_dir, plan, samples)
    print(f"正式实验全部完成：{len(runs)} 个，汇总在 {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
