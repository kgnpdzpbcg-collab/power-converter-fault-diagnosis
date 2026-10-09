"""两次真实 Train/Val 短训练验证确定性；结果不进入正式 12 次实验。"""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.data import load_samples
from converter.models_v3 import FaultModelV3
from converter.prepare import write_json
from converter.train import TrainConfig, train


def main():
    """训练两轮不是性能结论，只核验整条数据、优化、保存和重载流程。"""
    root = Path(__file__).resolve().parents[1]
    prepared = root / "artifacts/prepared/v0"
    output = root / "artifacts/checks/v3_determinism"
    samples = load_samples(prepared)
    config = TrainConfig(seed=42, max_epochs=2)
    results = [train(prepared, output / f"repeat{i}", "b3_attention_skip", config=config,
                     samples=samples, evaluate_test=False, model_factory=FaultModelV3) for i in (1, 2)]
    histories, predictions, states = [], [], []
    for i in (1, 2):
        folder = output / f"repeat{i}"
        with (folder / "history.csv").open(encoding="utf-8", newline="") as handle:
            histories.append([{key: value for key, value in row.items() if key != "seconds"}
                              for row in csv.DictReader(handle)])
        with np.load(folder / "validation_predictions.npz", allow_pickle=False) as archive:
            predictions.append({key: archive[key] for key in archive.files})
        states.append(torch.load(folder / "best.pt", map_location="cpu", weights_only=True))
        metadata = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
        assert metadata["dataset_splits"] == ["train", "val"] and metadata["test_evaluation_requested"] is False
        assert "test" not in results[i - 1] and not (folder / "test_predictions.npz").exists()
    assert histories[0] == histories[1]
    assert all(np.array_equal(value, predictions[1][key]) for key, value in predictions[0].items())
    assert all(torch.equal(value, states[1][key]) for key, value in states[0].items())
    report = {"status": "passed", "epochs_each": 2, "repeats": 2,
              "history_except_timing_equal": True, "checkpoint_tensors_equal": True,
              "validation_probabilities_equal": True, "no_test_evaluation": True}
    write_json(output / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
