"""完整重放九个最优checkpoint，独立重算日志选模、预测、指标和版本边界。"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from converter.data import ConverterDataset, load_samples
from converter.metrics import classification_metrics
from converter.prepare import write_json
from converter.train import fingerprint
from .models import CandidateModel
from .run import source_hashes


def validate(root, output):
    """只构建Val Dataset，不重训练、不评估Test；任何不一致立即抛错。"""
    prepared = root / "artifacts/prepared/v0"
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert source_hashes(root) == plan["source_sha256"]
    for name, digest in plan["source_sha256"].items():
        assert fingerprint(output / "source_snapshot" / name) == digest
    for name, digest in plan["prepared_sha256"].items():
        assert fingerprint(prepared / name) == digest
    assert fingerprint(output / "leakage_audit.json") == plan["audit_sha256"]
    assert not list(output.glob("**/test_predictions.npz"))
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    samples = load_samples(prepared)
    dataset = ConverterDataset(prepared, "val", "raw", samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=plan["training"]["batch_size"], shuffle=False)
    counts = {}
    for spec in plan["runs"]:
        folder = output / spec["id"]
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        experiment = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
        assert result["status"] == "complete" and "test" not in result
        assert experiment["dataset_splits"] == ["train", "val"] and not experiment["test_evaluation_requested"]
        history = list(csv.DictReader((folder / "history.csv").open(encoding="utf-8")))
        score, best_epoch, stale = -1.0, 0, 0
        for row in history:
            value = float(row["val_macro_f1"])
            if value > score + 1e-12:
                score, best_epoch, stale = value, int(row["epoch"]), 0
            else:
                stale += 1
        assert len(history) == result["epochs_run"]
        assert result["best_epoch"] == best_epoch and abs(score-result["best_val_macro_f1"]) < 1e-12
        assert len(history) == plan["training"]["max_epochs"] or stale == plan["training"]["patience"]
        model = CandidateModel(variant=spec["variant"])
        model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True), strict=True)
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for batch in loader:
                probabilities.append(model(batch)[0].softmax(1).numpy())
        probabilities = np.concatenate(probabilities)
        predictions = probabilities.argmax(1)
        with np.load(folder / "validation_predictions.npz", allow_pickle=False) as saved:
            assert np.array_equal(saved["sample_index"], dataset.indices)
            assert np.array_equal(saved["label"], dataset.labels.numpy())
            assert np.array_equal(saved["probability"], probabilities) and np.array_equal(saved["prediction"], predictions)
        confusion = np.bincount(dataset.labels.numpy()*8 + predictions, minlength=64).reshape(8,8)
        metrics = classification_metrics(confusion)
        for name, value in metrics.items():
            assert np.allclose(value, result["validation"][name], atol=1e-12, rtol=0), (spec["id"], name)
        count = sum(p.numel() for p in model.parameters())
        assert count == experiment["parameters"] == result["parameters"]
        counts[spec["variant"]] = count
        if spec["variant"] == "baseline":
            with np.load(root / f"artifacts/experiments/v2_validation/mean_plain_seed{spec['seed']}/validation_predictions.npz", allow_pickle=False) as old:
                assert np.array_equal(old["probability"], probabilities)
        print(f"核验通过 {spec['id']}", flush=True)
    assert counts == {"baseline": 24936, "wide": 27366, "multiscale": 27432}
    report = {"status": "passed", "checkpoint_prediction_replays": len(plan["runs"]),
              "history_best_epoch_and_early_stop_checked": True, "all_metrics_recomputed": True,
              "historical_baseline_probabilities_exact_seeds": [42,123,2026],
              "parameters": counts, "source_and_data_hashes_unchanged": True,
              "dataset_splits": ["train","val"], "test_evaluated": False}
    write_json(output / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    validate(Path.cwd(), Path("artifacts/experiments/multiscale_candidate"))
