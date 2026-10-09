"""预先固定的第二轮实验：六个变体、三个种子，仅训练和验证。"""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np

from .data import load_samples
from .diagnose_gate import diagnose
from .models_v2 import FaultModelV2
from .prepare import write_json
from .train import TrainConfig, fingerprint, train


VARIANTS = (
    ("mean_plain", "等权平均", "mean", False, False),
    ("mean_norm", "等权平均＋分支归一化", "mean", True, False),
    ("gate_plain", "动态门控对照", "dynamic", False, False),
    ("gate_uniform", "动态门控＋等权初始权重", "dynamic", False, True),
    ("gate_norm", "动态门控＋分支归一化", "dynamic", True, False),
    ("gate_norm_uniform", "动态门控＋两项改进", "dynamic", True, True),
)
SEEDS = (42, 123, 2026)


def make_plan(config):
    """在训练前列出全部实验；各变体的公共参数、数据划分和损失完全相同。"""
    return {"stage": "validation_only", "protocol": "raw", "seeds": list(SEEDS),
            "training": asdict(config), "initialization": "paired_encoders_classifier_and_dropout_rng",
            "selection_metric": "validation_macro_f1", "test_evaluation_requested": False,
            "runs": [{"id": f"{name}_seed{seed}", "variant": name, "description": description,
                      "seed": seed, "fusion": fusion, "branches": ["current", "dc", "rms"],
                      "model_options": {"branch_norm": norm, "uniform_gate_init": uniform}}
                     for name, description, fusion, norm, uniform in VARIANTS for seed in SEEDS]}


def aggregate(output_dir, plan):
    """只从本轮验证结果汇总；分种子保存成对差值，三次重复不宣称统计显著。"""
    output_dir = Path(output_dir)
    results = {run["id"]: json.loads((output_dir / run["id"] / "result.json").read_text(encoding="utf-8"))
               for run in plan["runs"] if (output_dir / run["id"] / "result.json").exists()}
    rows = []
    for name, description, _, _, _ in VARIANTS:
        group = [results[f"{name}_seed{seed}"] for seed in SEEDS if f"{name}_seed{seed}" in results]
        if len(group) != len(SEEDS):
            continue
        values = {"macro_f1": [r["validation"]["macro_f1"] for r in group],
                  "normal_recall": [r["validation"]["per_class_recall"][0] for r in group],
                  "dc_f1": [r["validation"]["per_class_f1"][6] for r in group],
                  "dc_precision": [r["validation"]["per_class_precision"][6] for r in group],
                  "dc_recall": [r["validation"]["per_class_recall"][6] for r in group]}
        row = {"variant": name, "description": description, "parameters": group[0]["parameters"],
               "best_epochs": [r["best_epoch"] for r in group]}
        for metric, scores in values.items():
            row[f"{metric}_mean"] = float(np.mean(scores))
            row[f"{metric}_std"] = float(np.std(scores, ddof=1))
        row["macro_f1_by_seed"] = {str(seed): value for seed, value in zip(SEEDS, values["macro_f1"])}
        rows.append(row)
    comparisons = []
    for reference in ("mean_plain", "gate_plain"):
        for name, description, _, _, _ in VARIANTS:
            if name == reference or not all(f"{name}_seed{s}" in results and f"{reference}_seed{s}" in results for s in SEEDS):
                continue
            differences = [results[f"{name}_seed{s}"]["validation"]["macro_f1"]
                           - results[f"{reference}_seed{s}"]["validation"]["macro_f1"] for s in SEEDS]
            comparisons.append({"variant": name, "reference": reference, "differences_by_seed": differences,
                                "mean_difference": float(np.mean(differences)),
                                "wins": sum(value > 0 for value in differences)})
    summary = {"dataset": "validation", "completed_runs": len(results), "planned_runs": len(plan["runs"]),
               "rows": rows, "paired_comparisons": comparisons,
               "total_epochs": sum(r["epochs_run"] for r in results.values()),
               "total_train_seconds": sum(r["train_seconds"] for r in results.values()),
               "interpretation": "验证集用于开发；三种子结果不等同于独立事件的置信区间或最终测试结论"}
    write_json(output_dir / "summary.json", summary)
    # CSV 保留容易比较的数值，逐种子详情在 JSON 内。
    flat_rows = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
    if flat_rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    return summary


def run(prepared_dir, output_dir):
    """冻结计划和源码后依次训练；允许恢复中断，但禁止覆盖或混用不同源码。"""
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    config = TrainConfig()
    plan = make_plan(config)
    code_dir = Path(__file__).parent
    code_hash = {path.name: fingerprint(path) for path in sorted(code_dir.glob("*.py"))}
    if output_dir.exists():
        if json.loads((output_dir / "plan.json").read_text(encoding="utf-8")) != plan:
            raise ValueError("已有实验计划与当前计划不同，禁止覆盖")
        archived = json.loads((output_dir / "source_snapshot" / "manifest.json").read_text(encoding="utf-8"))
        if any(archived.get(name) != digest for name, digest in code_hash.items()):
            raise ValueError("实验期间源码发生变化，不能在同一轮继续训练")
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "plan.json", plan)
        snapshot = output_dir / "source_snapshot"
        snapshot.mkdir()
        for filename in code_hash:
            shutil.copy2(code_dir / filename, snapshot / filename)
        for filename in ("uv.lock", "pyproject.toml", "V2_PLAN.md"):
            shutil.copy2(code_dir.parent / filename, snapshot / filename)
        write_json(snapshot / "manifest.json", {path.name: fingerprint(path) for path in snapshot.iterdir() if path.is_file()})
    samples = load_samples(prepared_dir)
    completed = []
    for spec in plan["runs"]:
        folder = output_dir / spec["id"]
        if (folder / "result.json").exists():
            completed.append(spec["id"])
            continue
        if folder.exists():
            raise ValueError(f"存在未完成目录 {folder}，需先检查中断原因，不自动删除")
        write_json(output_dir / "progress.json", {"completed": completed, "running": spec["id"], "total": len(plan["runs"])})
        train(prepared_dir, folder, spec["fusion"], tuple(spec["branches"]), "raw",
              replace(config, seed=spec["seed"]), samples, evaluate_test=False,
              model_factory=FaultModelV2, model_options=spec["model_options"], trace=True)
        diagnose(prepared_dir, folder, samples)
        completed.append(spec["id"])
        write_json(output_dir / "progress.json", {"completed": completed, "running": None, "total": len(plan["runs"])})
        aggregate(output_dir, plan)
    return aggregate(output_dir, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/v2_validation"))
    args = parser.parse_args()
    result = run(args.prepared_dir, args.output_dir)
    print(json.dumps({"completed_runs": result["completed_runs"], "dataset": result["dataset"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
