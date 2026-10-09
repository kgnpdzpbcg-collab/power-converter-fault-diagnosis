"""V3 独立核验：重算验证指标、恢复概率、复现旧 concat，不评估测试集。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import ConverterDataset, load_samples
from .experiments_v3 import aggregate
from .metrics import classification_metrics
from .models_v3 import FaultModelV3
from .prepare import write_json
from .train import evaluate, fingerprint


def read_history(folder):
    """读取完整日志；与旧实验比较时只排除不能确定复现的运行耗时。"""
    with (Path(folder) / "history.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate(prepared_dir, output_dir, previous_dir):
    """核验所有正式运行；任何不一致都中止，不把失败结果标为通过。"""
    prepared_dir, output_dir, previous_dir = Path(prepared_dir), Path(output_dir), Path(previous_dir)
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    samples = load_samples(prepared_dir)
    dataset = ConverterDataset(prepared_dir, "val", "raw", samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=plan["training"]["batch_size"], shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(weight=dataset.class_weights, reduction="sum")
    torch.set_num_threads(plan["training"]["threads"])
    torch.use_deterministic_algorithms(True)
    snapshot = output_dir / "source_snapshot"
    snapshot_hash = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in snapshot_hash.items():
        assert fingerprint(snapshot / name) == digest, name
        if name.endswith(".py"):
            assert fingerprint(Path(__file__).parent / name) == digest, name
    source = json.loads((prepared_dir / "config.json").read_text(encoding="utf-8"))["source_sha256"]
    verified, reproduced, attention_verified = [], [], []
    for spec in plan["runs"]:
        folder = output_dir / spec["id"]
        metadata = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert result["status"] == "complete"
        assert metadata["test_evaluation_requested"] is False and metadata["dataset_splits"] == ["train", "val"]
        assert not (folder / "test_predictions.npz").exists() and "test" not in result
        assert metadata["manifest_sha256"] == fingerprint(prepared_dir / "manifest.csv")
        assert metadata["normalization_sha256"] == fingerprint(prepared_dir / "normalization_raw.npz")
        assert metadata["source_sha256"] == source and metadata["lock_sha256"] == fingerprint(snapshot / "uv.lock")
        assert metadata["fusion"] == result["fusion"] == spec["fusion"]
        assert metadata["branches"] == result["branches"] == spec["branches"]
        assert metadata["protocol"] == result["protocol"] == plan["protocol"] == "raw"
        assert metadata["model_class"] == "converter.models_v3.FaultModelV3" and metadata["model_options"] == {}
        assert metadata["training_trace"] is False
        assert metadata["training"]["seed"] == result["seed"] == spec["seed"]
        assert all(metadata["training"][key] == value for key, value in plan["training"].items() if key != "seed")
        assert set(metadata["code_sha256"]) == {name for name in snapshot_hash if name.endswith(".py")}
        assert all(metadata["code_sha256"][name] == snapshot_hash[name] for name in metadata["code_sha256"])
        history = read_history(folder)
        assert all(np.isfinite(float(value)) for row in history for value in row.values())
        assert [int(row["epoch"]) for row in history] == list(range(1, len(history) + 1))
        scores = np.array([float(row["val_macro_f1"]) for row in history])
        assert int(scores.argmax()) + 1 == result["best_epoch"]
        assert float(scores.max()) == result["best_val_macro_f1"] and len(history) == result["epochs_run"]
        if len(history) < metadata["training"]["max_epochs"]:
            assert len(history) - result["best_epoch"] == metadata["training"]["patience"]
        with np.load(folder / "validation_predictions.npz", allow_pickle=False) as archive:
            predictions = {key: archive[key] for key in archive.files}
        assert "weights" not in predictions
        assert np.array_equal(predictions["sample_index"], dataset.indices)
        assert np.array_equal(predictions["label"], dataset.labels.numpy())
        probability = predictions["probability"]
        assert np.isfinite(probability).all() and np.all(probability >= 0)
        assert np.allclose(probability.sum(1), 1, atol=1e-6)
        assert np.array_equal(predictions["prediction"], probability.argmax(1))
        confusion = np.bincount(predictions["label"] * 8 + predictions["prediction"], minlength=64).reshape(8, 8)
        assert all(result["validation"][key] == value for key, value in classification_metrics(confusion).items())
        model = FaultModelV3(spec["fusion"])
        state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        assert all(torch.isfinite(value).all() for value in state.values())
        model.load_state_dict(state, strict=True)
        assert sum(parameter.numel() for parameter in model.parameters()) == result["parameters"] == metadata["parameters"]
        assert (folder / "best.pt").stat().st_size == result["checkpoint_bytes"]
        metrics, restored = evaluate(model, loader, criterion, collect=True)
        assert metrics == result["validation"]
        assert all(np.array_equal(value, restored[key]) for key, value in predictions.items())
        if spec["variant"] == "b0_concat":
            previous = previous_dir / f"raw_concat64_s{spec['seed']}"
            with np.load(previous / "validation_predictions.npz", allow_pickle=False) as old:
                assert all(np.array_equal(value, old[key]) for key, value in predictions.items())
            exclude_time = lambda rows: [{key: value for key, value in row.items() if key != "seconds"} for row in rows]
            assert exclude_time(history) == exclude_time(read_history(previous))
            reproduced.append(spec["seed"])
        if spec["variant"] in ("b2_attention", "b3_attention_skip"):
            with np.load(folder / "validation_attention.npz", allow_pickle=False) as archive:
                weights = archive["attention"]
                assert np.array_equal(archive["sample_index"], dataset.indices)
                assert np.array_equal(archive["label"], dataset.labels.numpy())
            assert weights.shape == (len(dataset), 4, 3, 3)
            assert np.isfinite(weights).all() and np.all(weights >= 0)
            assert np.allclose(weights.sum(-1), 1, atol=1e-6)
            assert np.all(weights.diagonal(axis1=-2, axis2=-1) > 0)
            replay = []
            with torch.inference_mode():
                for batch in loader:
                    fused, matrix = model.fused_features(batch, collect_attention=True)
                    assert torch.equal(model.classifier(fused), model(batch)[0])
                    replay.append(matrix.numpy())
            assert np.array_equal(weights, np.concatenate(replay))
            summary = json.loads((folder / "attention_summary.json").read_text(encoding="utf-8"))
            assert summary["mean_by_head"] == weights.mean(0).tolist()
            assert summary["std_across_samples_by_head"] == weights.std(0).tolist()
            attention_verified.append(spec["id"])
        verified.append(spec["id"])
    saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert aggregate(output_dir, plan) == saved_summary
    assert len(verified) == 12
    report = {"status": "passed", "dataset": "validation", "verified_runs": verified,
              "no_test_evaluation": True, "fixed_manifest_normalization_and_source_hashes": True,
              "frozen_code_snapshot_matches": True, "prediction_metrics_recalculated": True,
              "best_checkpoint_probabilities_exactly_reproduced": True, "attention_exactly_reproduced": attention_verified,
              "b0_matches_v1_concat_seeds": reproduced, "summary_recalculated": True}
    write_json(output_dir / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/v3_validation"))
    parser.add_argument("--previous-dir", type=Path, default=Path("artifacts/experiments/v1"))
    args = parser.parse_args()
    report = validate(args.prepared_dir, args.output_dir, args.previous_dir)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
