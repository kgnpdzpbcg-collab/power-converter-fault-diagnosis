"""正式实验结束后的独立核验：清单、选模、预测指标和完整实验覆盖。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_samples
from .metrics import classification_metrics
from .models import FaultModel
from .prepare import protocol_mask, write_json
from .train import fingerprint


def validate(prepared_dir, output_dir):
    """逐实验重算已保存预测的指标；不重新训练或用测试分数选择模型。

    校验 checkpoint 的张量键、形状、有限性；核查最优验证轮次与早停。
    完整测试概率已经保存，核验它们不需要再次前向计算整个测试集。
    """
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    samples = load_samples(prepared_dir)
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    source_config = json.loads((prepared_dir / "config.json").read_text(encoding="utf-8"))
    manifest_hash = fingerprint(prepared_dir / "manifest.csv")
    verified = []
    # 老实验绑定其训练时的源码快照，允许当前目录继续开发而不误报旧代码变化。
    snapshot = output_dir / "source_snapshot"
    code_root = snapshot if snapshot.is_dir() else Path(__file__).parent
    for run in plan["runs"]:
        folder = output_dir / run["id"]
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        metadata = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
        assert result["status"] == "complete"
        assert metadata["manifest_sha256"] == manifest_hash
        assert metadata["normalization_sha256"] == fingerprint(prepared_dir / f"normalization_{run['protocol']}.npz")
        assert metadata["source_sha256"] == source_config["source_sha256"]
        assert metadata["lock_sha256"] == fingerprint(Path(__file__).parent.parent / "uv.lock")
        for key in ("fusion", "branches", "protocol"):
            assert metadata[key] == run[key] == result[key]
        assert metadata["training"]["seed"] == run["seed"] == result["seed"]
        for key, value in plan["training"].items():
            if key != "seed":
                assert metadata["training"][key] == value
        for filename, expected in metadata["code_sha256"].items():
            assert fingerprint(code_root / filename) == expected, f"实验源码与记录不一致: {filename}"
        with (folder / "history.csv").open(encoding="utf-8", newline="") as handle:
            history = list(csv.DictReader(handle))
        assert [int(row["epoch"]) for row in history] == list(range(1, len(history) + 1))
        assert all(np.isfinite(float(value)) for row in history for value in row.values())
        scores = np.array([float(row["val_macro_f1"]) for row in history])
        best_epoch = int(scores.argmax()) + 1
        assert best_epoch == result["best_epoch"]
        assert scores.max() == result["best_val_macro_f1"]
        assert len(history) == result["epochs_run"]
        if len(history) < metadata["training"]["max_epochs"]:
            assert len(history) - best_epoch == metadata["training"]["patience"]
        eligible = protocol_mask(samples, run["protocol"])
        for split_name, split_id, result_key in (("validation", 1, "validation"), ("test", 2, "test")):
            with np.load(folder / f"{split_name}_predictions.npz", allow_pickle=False) as predictions:
                expected_indices = np.flatnonzero((samples["split"] == split_id) & eligible)
                assert np.array_equal(predictions["sample_index"], expected_indices)
                assert np.array_equal(predictions["label"], samples["label"][expected_indices])
                probabilities = predictions["probability"]
                assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
                assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                assert np.array_equal(predictions["prediction"], probabilities.argmax(axis=1))
                confusion = np.bincount(predictions["label"] * 8 + predictions["prediction"], minlength=64).reshape(8, 8)
                recalculated = classification_metrics(confusion)
                for key, value in recalculated.items():
                    assert value == result[result_key][key], (run["id"], split_name, key)
                if "weights" in predictions:
                    weights = predictions["weights"]
                    assert weights.shape == (len(expected_indices), len(run["branches"]))
                    assert np.isfinite(weights).all() and np.all(weights >= 0)
                    assert np.allclose(weights.sum(axis=1), 1, atol=1e-6)
        model = FaultModel(run["fusion"], tuple(run["branches"]))
        state = torch.load(folder / "best.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(state, strict=True)
        assert all(torch.isfinite(tensor).all() for tensor in state.values())
        assert sum(parameter.numel() for parameter in model.parameters()) == result["parameters"]
        assert (folder / "best.pt").stat().st_size == result["checkpoint_bytes"]
        verified.append(run["id"])
    for name, digest in source_config["source_sha256"].items():
        audit = json.loads((prepared_dir / "audit.json").read_text(encoding="utf-8"))
        assert fingerprint(audit["sources"][name]["path"]) == digest
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_runs"] == len(plan["runs"]) and not summary["missing_runs"]
    for row in summary["rows"]:
        group = [json.loads((output_dir / run["id"] / "result.json").read_text(encoding="utf-8"))
                 for run in plan["runs"] if run["protocol"] == row["protocol"] and run["model"] == row["model"]]
        assert len(group) == len(plan["seeds"])
        values = np.array([result["test"]["macro_f1"] for result in group])
        assert values.mean() == row["macro_f1_mean"]
        if len(values) > 1:
            assert values.std(ddof=1) == row["macro_f1_std"]
    report = {"status": "passed", "verified_runs": verified, "completed_runs": len(verified),
              "fixed_manifest_and_train_configuration": True, "source_files_unchanged": True,
              "best_checkpoint_and_early_stopping_verified": True, "prediction_metrics_recalculated": True,
              "all_checkpoint_tensors_finite_and_loadable": True, "three_seed_aggregation_verified": True,
              "note": "正式训练未逐次重复；同 seed 短流程精确复现记录位于 artifacts/checks/reproducibility.json"}
    write_json(output_dir / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/v1"))
    args = parser.parse_args()
    result = validate(args.prepared_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "completed_runs": result["completed_runs"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
