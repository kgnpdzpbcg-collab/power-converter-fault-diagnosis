"""固定的 V3 实验：B0–B3 各三个种子，只使用训练与验证集。"""

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from .data import ConverterDataset, load_samples
from .models_v3 import FaultModelV3, VARIANTS
from .prepare import write_json
from .train import TrainConfig, fingerprint, train


SEEDS = (42, 123, 2026)
DESCRIPTIONS = ("原始拼接", "普通MLP＋跳跃拼接", "仅注意力交互", "注意力＋跳跃拼接")


def make_plan(config):
    """训练前固定全部 12 次实验；不根据中间结果修改矩阵或超参数。"""
    return {"stage": "validation_only", "protocol": "raw", "seeds": list(SEEDS),
            "training": asdict(config), "test_evaluation_requested": False,
            "selection_metric": "validation_macro_f1",
            "initialization": "paired_encoders; head_seed+1; interaction_seed+2; restored_global_rng",
            "runs": [{"id": f"{variant}_seed{seed}", "variant": variant, "description": description,
                      "seed": seed, "fusion": variant, "branches": ["current", "dc", "rms"]}
                     for variant, description in zip(VARIANTS, DESCRIPTIONS) for seed in SEEDS]}


def diagnose_attention(prepared_dir, folder, samples):
    """在最优模型的验证集上保存逐头矩阵及描述统计，不参与训练或选模。

    矩阵行=query，列=key，次序都是 current/dc/rms。没有跨样本注意力。
    按类别均值只用于观察行为，不能解释为因果贡献或传感器可靠性。
    """
    folder = Path(folder)
    metadata = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
    if metadata["fusion"] not in ("b2_attention", "b3_attention_skip"):
        return
    dataset = ConverterDataset(prepared_dir, "val", "raw", samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=metadata["training"]["batch_size"], shuffle=False)
    model = FaultModelV3(metadata["fusion"])
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval()
    matrices = []
    with torch.inference_mode():
        for batch in loader:
            _, weights = model.fused_features(batch, collect_attention=True)
            matrices.append(weights.numpy())
    weights = np.concatenate(matrices)
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(-1)
    labels = dataset.labels.numpy()
    np.savez_compressed(folder / "validation_attention.npz", sample_index=dataset.indices,
                        label=labels, attention=weights)
    write_json(folder / "attention_summary.json", {
        "dataset": "validation", "branches": list(model.branches), "heads": 4,
        "shape": list(weights.shape), "row_query_column_key": True,
        "mean_by_head": weights.mean(0).tolist(), "std_across_samples_by_head": weights.std(0).tolist(),
        "mean_entropy_by_head_query": entropy.mean(0).tolist(),
        "uniform_entropy": float(np.log(3)),
        "by_class_mean": {str(label): weights[labels == label].mean(0).tolist() for label in range(8)},
        "interpretation": "描述网络交互行为，不是物理因果或传感器可信度"})


def aggregate(output_dir, plan):
    """汇总验证得分、复杂度及逐种子差值；标准差使用 ddof=1。"""
    output_dir = Path(output_dir)
    results = {spec["id"]: json.loads((output_dir / spec["id"] / "result.json").read_text(encoding="utf-8"))
               for spec in plan["runs"] if (output_dir / spec["id"] / "result.json").exists()}
    rows = []
    for variant, description in zip(VARIANTS, DESCRIPTIONS):
        group = [results[f"{variant}_seed{seed}"] for seed in SEEDS if f"{variant}_seed{seed}" in results]
        if len(group) != len(SEEDS):
            continue
        row = {"variant": variant, "description": description, "parameters": group[0]["parameters"],
               "macro_f1_by_seed": {str(seed): r["validation"]["macro_f1"] for seed, r in zip(SEEDS, group)},
               "best_epochs": [r["best_epoch"] for r in group],
               "per_class_f1_mean": np.mean([r["validation"]["per_class_f1"] for r in group], axis=0).tolist()}
        measurements = {key: [r["validation"][key] for r in group]
                        for key in ("macro_f1", "macro_precision", "macro_recall", "accuracy")}
        measurements.update({"normal_recall": [r["validation"]["per_class_recall"][0] for r in group],
                             "dc_f1": [r["validation"]["per_class_f1"][6] for r in group],
                             "dc_precision": [r["validation"]["per_class_precision"][6] for r in group],
                             "dc_recall": [r["validation"]["per_class_recall"][6] for r in group],
                             "latency_median_ms": [r["latency_cpu"]["median_ms"] for r in group],
                             "checkpoint_bytes": [r["checkpoint_bytes"] for r in group]})
        for name, values in measurements.items():
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    comparisons = []
    for reference in VARIANTS[:3]:
        if not all(f"b3_attention_skip_seed{s}" in results and f"{reference}_seed{s}" in results for s in SEEDS):
            continue
        differences = [results[f"b3_attention_skip_seed{s}"]["validation"]["macro_f1"]
                       - results[f"{reference}_seed{s}"]["validation"]["macro_f1"] for s in SEEDS]
        comparisons.append({"variant": "b3_attention_skip", "reference": reference,
                            "differences_by_seed": {str(s): d for s, d in zip(SEEDS, differences)},
                            "mean_difference": float(np.mean(differences)), "wins": sum(d > 0 for d in differences)})
    summary = {"dataset": "validation", "completed_runs": len(results), "planned_runs": len(plan["runs"]),
               "rows": rows, "paired_comparisons": comparisons,
               "total_epochs": sum(r["epochs_run"] for r in results.values()),
               "total_train_seconds": sum(r["train_seconds"] for r in results.values()),
               "interpretation": "开发验证；同一划分的三个训练种子，不作统计显著性或独立测试结论"}
    write_json(output_dir / "summary.json", summary)
    flat = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows]
    if flat:
        with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader(); writer.writerows(flat)
    return summary


def run(prepared_dir, output_dir):
    """先冻结计划、Git 提交和全部源码；可恢复已完成运行，不覆盖中断目录。"""
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    config = TrainConfig()
    plan = make_plan(config)
    code_dir = Path(__file__).parent
    hashes = {path.name: fingerprint(path) for path in sorted(code_dir.glob("*.py"))}
    if output_dir.exists():
        if json.loads((output_dir / "plan.json").read_text(encoding="utf-8")) != plan:
            raise ValueError("已有 V3 计划不同，禁止覆盖")
        archived = json.loads((output_dir / "source_snapshot" / "manifest.json").read_text(encoding="utf-8"))
        if any(archived.get(name) != digest for name, digest in hashes.items()):
            raise ValueError("V3 源码变化，禁止在同一轮混合训练")
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "plan.json", plan)
        snapshot = output_dir / "source_snapshot"
        snapshot.mkdir()
        for name in hashes:
            shutil.copy2(code_dir / name, snapshot / name)
        for name in ("uv.lock", "pyproject.toml", "V3_PLAN.md"):
            shutil.copy2(code_dir.parent / name, snapshot / name)
        write_json(snapshot / "manifest.json", {path.name: fingerprint(path) for path in snapshot.iterdir()})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_dir.parent, text=True).strip()
        write_json(output_dir / "version.json", {"local_commit": commit, "plan_sha256": fingerprint(code_dir.parent / "V3_PLAN.md")})
    samples = load_samples(prepared_dir)
    completed = []
    for spec in plan["runs"]:
        folder = output_dir / spec["id"]
        if (folder / "result.json").exists():
            completed.append(spec["id"])
            continue
        if folder.exists():
            raise ValueError(f"发现未完成目录 {folder}，先检查原因，不自动删除")
        write_json(output_dir / "progress.json", {"completed": completed, "running": spec["id"], "total": len(plan["runs"])})
        train(prepared_dir, folder, spec["fusion"], tuple(spec["branches"]), "raw",
              replace(config, seed=spec["seed"]), samples, evaluate_test=False, model_factory=FaultModelV3)
        diagnose_attention(prepared_dir, folder, samples)
        completed.append(spec["id"])
        write_json(output_dir / "progress.json", {"completed": completed, "running": None, "total": len(plan["runs"])})
        aggregate(output_dir, plan)
    return aggregate(output_dir, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/v3_validation"))
    args = parser.parse_args()
    summary = run(args.prepared_dir, args.output_dir)
    print(json.dumps({"completed_runs": summary["completed_runs"], "dataset": summary["dataset"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
