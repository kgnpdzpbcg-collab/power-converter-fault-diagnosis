"""第二轮独立核验：只重算验证指标并恢复验证预测，不评估测试集。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import ConverterDataset, load_samples
from .experiments_v2 import aggregate
from .metrics import classification_metrics
from .models_v2 import FaultModelV2
from .prepare import write_json
from .train import evaluate, fingerprint


def validate(prepared_dir, output_dir, previous_dir=None):
    """检查固定数据、选模日志、训练观测、源码和实际 checkpoint 预测。

    previous_dir 可提供首轮实验目录，仅读取其 mean 验证预测及历史记录，
    确认本轮 mean_plain 完整复现首轮基线，不读取首轮 result.json 的测试指标。
    """
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    samples = load_samples(prepared_dir)
    dataset = ConverterDataset(prepared_dir, "val", "raw", samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=plan["training"]["batch_size"], shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(weight=dataset.class_weights, reduction="sum")
    torch.set_num_threads(plan["training"]["threads"])
    torch.use_deterministic_algorithms(True)
    verified, reproduced = [], []
    for spec in plan["runs"]:
        folder = output_dir / spec["id"]
        metadata = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert result["status"] == "complete"
        assert metadata["test_evaluation_requested"] is False
        assert metadata["dataset_splits"] == ["train", "val"]
        assert not (folder / "test_predictions.npz").exists()
        assert "test" not in result and "test_gate" not in result
        assert metadata["manifest_sha256"] == fingerprint(prepared_dir / "manifest.csv")
        assert metadata["normalization_sha256"] == fingerprint(prepared_dir / "normalization_raw.npz")
        assert metadata["model_options"] == spec["model_options"]
        assert metadata["fusion"] == spec["fusion"] == result["fusion"]
        assert metadata["branches"] == spec["branches"] == result["branches"]
        assert metadata["protocol"] == result["protocol"] == plan["protocol"]
        assert metadata["model_class"] == "converter.models_v2.FaultModelV2"
        assert metadata["training_trace"] is True
        assert metadata["training"]["seed"] == spec["seed"] == result["seed"]
        for name, value in plan["training"].items():
            if name != "seed":
                assert metadata["training"][name] == value
        for filename, digest in metadata["code_sha256"].items():
            assert fingerprint(output_dir / "source_snapshot" / filename) == digest
        with (folder / "history.csv").open(encoding="utf-8", newline="") as handle:
            history = list(csv.DictReader(handle))
        scores = np.array([float(row["val_macro_f1"]) for row in history])
        assert [int(row["epoch"]) for row in history] == list(range(1, len(history) + 1))
        assert all(np.isfinite(float(value)) for row in history for value in row.values())
        assert int(scores.argmax()) + 1 == result["best_epoch"]
        assert float(scores.max()) == result["best_val_macro_f1"]
        assert len(history) == result["epochs_run"]
        if len(history) < metadata["training"]["max_epochs"]:
            assert len(history) - result["best_epoch"] == metadata["training"]["patience"]
        with (folder / "gate_trace.csv").open(encoding="utf-8", newline="") as handle:
            trace = list(csv.DictReader(handle))
        assert len(trace) == len(history)
        assert all(np.isfinite(float(value)) for row in trace for value in row.values())
        assert all(int(row["samples"]) == int((samples["split"] == 0).sum()) for row in trace)
        assert all(abs(sum(float(row[f"{name}_weight_mean"]) for name in spec["branches"]) - 1) < 1e-6 for row in trace)
        with np.load(folder / "validation_predictions.npz", allow_pickle=False) as archive:
            predictions = {key: archive[key] for key in archive.files}
        assert np.array_equal(predictions["sample_index"], dataset.indices)
        assert np.array_equal(predictions["label"], dataset.labels.numpy())
        assert np.isfinite(predictions["probability"]).all()
        assert np.all(predictions["probability"] >= 0)
        assert np.allclose(predictions["probability"].sum(axis=1), 1, atol=1e-6)
        assert np.array_equal(predictions["prediction"], predictions["probability"].argmax(axis=1))
        assert predictions["weights"].shape == (len(dataset), len(spec["branches"]))
        assert np.isfinite(predictions["weights"]).all() and np.all(predictions["weights"] >= 0)
        assert np.allclose(predictions["weights"].sum(axis=1), 1, atol=1e-6)
        confusion = np.bincount(predictions["label"] * 8 + predictions["prediction"], minlength=64).reshape(8, 8)
        for name, value in classification_metrics(confusion).items():
            assert result["validation"][name] == value
        model = FaultModelV2(spec["fusion"], tuple(spec["branches"]), **spec["model_options"])
        state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        assert all(torch.isfinite(value).all() for value in state.values())
        model.load_state_dict(state, strict=True)
        assert sum(parameter.numel() for parameter in model.parameters()) == result["parameters"]
        assert (folder / "best.pt").stat().st_size == result["checkpoint_bytes"]
        metrics, restored_predictions = evaluate(model, loader, criterion, collect=True)
        assert metrics == result["validation"]
        assert all(np.array_equal(value, restored_predictions[name]) for name, value in predictions.items())
        if previous_dir is not None and spec["variant"] == "mean_plain":
            previous = Path(previous_dir) / f"raw_mean_s{spec['seed']}"
            with np.load(previous / "validation_predictions.npz", allow_pickle=False) as old:
                assert all(np.array_equal(value, old[name]) for name, value in predictions.items())
            with (previous / "history.csv").open(encoding="utf-8", newline="") as handle:
                old_history = list(csv.DictReader(handle))
            # 耗时受系统负载影响，其余训练轨迹应逐轮完全一致。
            assert [{k: v for k, v in row.items() if k != "seconds"} for row in history] == [
                {k: v for k, v in row.items() if k != "seconds"} for row in old_history]
            reproduced.append(spec["seed"])
        verified.append(spec["id"])
    saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert aggregate(output_dir, plan) == saved_summary
    assert len(verified) == len(plan["runs"])
    report = {"status": "passed", "dataset": "validation", "verified_runs": verified,
              "no_test_evaluation": True, "fixed_manifest_and_normalization": True,
              "prediction_metrics_recalculated": True, "best_checkpoint_predictions_exactly_reproduced": True,
              "mean_baseline_matches_v1_seeds": reproduced, "summary_recalculated": True}
    write_json(output_dir / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/v2_validation"))
    parser.add_argument("--previous-dir", type=Path, default=Path("artifacts/experiments/v1"))
    args = parser.parse_args()
    report = validate(args.prepared_dir, args.output_dir, args.previous_dir)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
