"""冻结九次实验后运行，复用原训练器与原Train/Val，不新增测试预测。"""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from converter.data import load_samples
from converter.prepare import write_json
from converter.train import TrainConfig, fingerprint, train
from .models import CandidateModel, VARIANTS, DESCRIPTIONS


SEEDS = (42, 123, 2026)


def source_hashes(root):
    """旧源码和新增候选源码都绑定运行版本，避免只记录旧训练器的指纹。"""
    return {str(path.relative_to(root)).replace("\\", "/"): fingerprint(path)
            for folder in ("converter", "candidate") for path in sorted((root / folder).glob("*.py"))}


def aggregate(output, plan):
    """汇总完整结果与逐种子差值，未完成的运行不混入平均。"""
    results = {row["id"]: json.loads((output / row["id"] / "result.json").read_text(encoding="utf-8"))
               for row in plan["runs"] if (output / row["id"] / "result.json").exists()}
    rows = []
    for variant, description in zip(VARIANTS, DESCRIPTIONS):
        if not all(f"{variant}_seed{seed}" in results for seed in SEEDS):
            continue
        group = [results[f"{variant}_seed{seed}"] for seed in SEEDS]
        row = {"variant": variant, "description": description, "parameters": group[0]["parameters"],
               "macro_f1_by_seed": {str(seed): value["validation"]["macro_f1"] for seed, value in zip(SEEDS, group)},
               "best_epochs": [value["best_epoch"] for value in group],
               "per_class_f1_mean": np.mean([value["validation"]["per_class_f1"] for value in group], axis=0).tolist()}
        measures = {"macro_f1": [v["validation"]["macro_f1"] for v in group],
                    "accuracy": [v["validation"]["accuracy"] for v in group],
                    "normal_recall": [v["validation"]["per_class_recall"][0] for v in group],
                    "dc_precision": [v["validation"]["per_class_precision"][6] for v in group],
                    "dc_recall": [v["validation"]["per_class_recall"][6] for v in group],
                    "dc_f1": [v["validation"]["per_class_f1"][6] for v in group]}
        for name, values in measures.items():
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    comparisons = []
    for reference in ("baseline", "wide"):
        if all(f"multiscale_seed{s}" in results and f"{reference}_seed{s}" in results for s in SEEDS):
            differences = [results[f"multiscale_seed{s}"]["validation"]["macro_f1"]
                           - results[f"{reference}_seed{s}"]["validation"]["macro_f1"] for s in SEEDS]
            comparisons.append({"reference": reference, "variant": "multiscale",
                                "differences_by_seed": dict(zip(map(str, SEEDS), differences)),
                                "mean_difference": float(np.mean(differences)), "wins": int(sum(d > 0 for d in differences))})
    summary = {"scope": "original Train/Val", "test_evaluated": False, "completed_runs": len(results),
               "planned_runs": len(plan["runs"]), "rows": rows, "paired_comparisons": comparisons,
               "total_epochs": sum(v["epochs_run"] for v in results.values()),
               "total_train_seconds": sum(v["train_seconds"] for v in results.values())}
    write_json(output / "summary.json", summary)
    if rows:
        flat = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in rows]
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    return summary


def run(root, output):
    """先核查本轮泄露审计与指纹，再冻结源码、配置并完成九次训练。

    目录拒绝覆盖；失败抛出原错误，不自动换配置或重试。
    选模始终使用旧Val Macro-F1，各种子不改变数据归属。
    """
    prepared = root / "artifacts/prepared/v0"
    audit_path = root / "artifacts/checks/current_split_20261009/leakage_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "passed" and not audit["direct_label_or_record_leakage_found"]
    for name, digest in audit["prepared_files_unchanged_sha256"].items():
        assert fingerprint(prepared / name) == digest
    config = TrainConfig()
    plan = {"training": asdict(config), "protocol": "raw", "fusion": "mean", "seeds": list(SEEDS),
            "loss": "original inverse-frequency weighted CE", "test_evaluation": False,
            "audit_sha256": fingerprint(audit_path), "source_sha256": source_hashes(root),
            "prepared_sha256": audit["prepared_files_unchanged_sha256"],
            "runs": [{"id": f"{variant}_seed{seed}", "variant": variant, "seed": seed}
                     for variant in VARIANTS for seed in SEEDS]}
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"; snapshot.mkdir()
    for name, digest in plan["source_sha256"].items():
        target = snapshot / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        assert fingerprint(target) == digest
    shutil.copy2(root / "CANDIDATE_RUN_PLAN.md", snapshot / "CANDIDATE_RUN_PLAN.md")
    shutil.copy2(root / "uv.lock", snapshot / "uv.lock")
    shutil.copy2(audit_path, output / "leakage_audit.json")
    write_json(output / "plan.json", plan)
    write_json(output / "version.json", {"local_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                                        "plan_sha256": fingerprint(root / "CANDIDATE_RUN_PLAN.md")})
    samples = load_samples(prepared)
    for index, spec in enumerate(plan["runs"], 1):
        assert source_hashes(root) == plan["source_sha256"]
        train(prepared, output / spec["id"], fusion="mean", config=TrainConfig(seed=spec["seed"]),
              samples=samples, evaluate_test=False, model_factory=CandidateModel,
              model_options={"variant": spec["variant"]})
        if spec["variant"] == "baseline":
            # 三个原基线先逐一重现历史预测，通过后才进入新编码器实验。
            old_path = root / f"artifacts/experiments/v2_validation/mean_plain_seed{spec['seed']}/validation_predictions.npz"
            with np.load(old_path, allow_pickle=False) as old, np.load(output / spec["id"] / "validation_predictions.npz", allow_pickle=False) as new:
                assert all(np.array_equal(old[key], new[key]) for key in ("sample_index", "label", "prediction", "probability")), "基线未精确复现"
        aggregate(output, plan)
        print(f"已完成 {index}/{len(plan['runs'])}: {spec['id']}", flush=True)
    return aggregate(output, plan)


if __name__ == "__main__":
    print(json.dumps(run(Path.cwd(), Path("artifacts/experiments/multiscale_candidate")), ensure_ascii=False, indent=2))
