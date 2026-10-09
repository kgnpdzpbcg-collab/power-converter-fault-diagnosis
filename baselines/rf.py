"""两种固定RF与已训练CNN-Mean的八分类对照、重载核验和汇总。

只使用原Train/Val和相同归一化；原始标签、事件及时间元数据不作为输入。
RF参数执行前固定，不使用验证分数调参。所有正式目录拒绝覆盖。
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

from converter.data import ConverterDataset, load_samples
from converter.metrics import classification_metrics
from converter.models import FaultModel
from converter.prepare import CLASS_NAMES, write_json
from converter.train import fingerprint


SEEDS = (42, 123, 2026)
MODELS = ("RF-Current", "RF-All", "CNN-Mean")
OUTPUT = Path("artifacts/experiments/rf_baselines")
CNN_OUTPUT = Path("artifacts/experiments/multiscale_candidate")
SOURCE_FILES = ("baselines/__init__.py", "baselines/rf.py", "converter/data.py",
                "converter/models.py", "converter/metrics.py", "converter/prepare.py", "converter/train.py")


def feature_matrix(inputs, model_name):
    """从测量字典生成float32二维输入；不读取label或任何审计元数据。

    RF-Current只取电流，RF-All按电流/DC/RMS顺序拼接；每个分支展平时保持原列顺序。
    输入既可为Dataset的Torch张量，也可为测试用NumPy数组，输出[B,60]或[B,103]。
    """
    if model_name not in MODELS[:2]:
        raise ValueError(f"未知RF模型：{model_name}")
    names = ("current",) if model_name == "RF-Current" else ("current", "dc", "rms")
    arrays = [np.asarray(inputs[name]).reshape(len(inputs[name]), -1) for name in names]
    features = np.ascontiguousarray(np.concatenate(arrays, axis=1), dtype=np.float32)
    expected = 60 if model_name == "RF-Current" else 103
    if features.shape[1] != expected or not np.isfinite(features).all():
        raise ValueError("RF输入维数或有限性不符合冻结协议")
    return features


def rf_parameters(seed, weights):
    """明确固定RF参数；class_weight逐项沿用CNN的训练集权重，不额外拟合。

    n_jobs1保证预测概率按照固定树顺序累加，便于保存后逐元素精确重放。
    未宣称这些参数来自原论文，原文RF具体超参数尚未核实。
    """
    return {"n_estimators": 100, "criterion": "gini", "max_features": "sqrt",
            "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1,
            "bootstrap": True, "n_jobs": 1, "random_state": seed,
            "class_weight": {i: float(w) for i, w in enumerate(weights)}}


def score_predictions(labels, predictions):
    """固定八类计算混淆和指标，并以sklearn独立复核零分母处理与Macro-F1。"""
    confusion = np.bincount(labels * 8 + predictions, minlength=64).reshape(8, 8)
    metrics = classification_metrics(confusion)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(8), zero_division=0)
    for key, expected in (("per_class_precision", precision), ("per_class_recall", recall),
                          ("per_class_f1", f1), ("support", support)):
        np.testing.assert_allclose(metrics[key], expected, rtol=0, atol=1e-12)
    assert abs(metrics["macro_f1"] - float(f1.mean())) < 1e-12
    assert abs(metrics["accuracy"] - accuracy_score(labels, predictions)) < 1e-12
    return metrics


def array_digest(array):
    """绑定输入的形状、dtype和连续字节，核查训练和重载使用完全相同特征。"""
    import hashlib
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_context(root):
    """重用已审计的固定样本池；只创建Train/Val视图并检查历史数据/CNN证据。

    来源样本NPZ包含全体记录，但本轮不创建Test Dataset或计算Test特征/预测。
    prepared指纹绑定全体原划分，避免在模型比较时偷偷改变记录归属。
    """
    audit = json.loads((root / "reports/candidate/leakage_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed" and not audit["direct_label_or_record_leakage_found"]
    assert audit["cross_split_blocks_and_source_rows"] == 0
    prepared = root / "artifacts/prepared/v0"
    for name, value in audit["prepared_files_unchanged_sha256"].items():
        assert fingerprint(prepared / name) == value, name
    samples = load_samples(prepared)
    datasets = {s: ConverterDataset(prepared, s, "raw", samples) for s in ("train", "val")}
    assert len(datasets["train"]) == 39687 and len(datasets["val"]) == 9298
    assert not np.intersect1d(datasets["train"].indices, datasets["val"].indices).size
    cnn_plan = json.loads((root / CNN_OUTPUT / "plan.json").read_text(encoding="utf-8"))
    for name, value in cnn_plan["source_sha256"].items():
        assert fingerprint(root / name) == value, name
    cnn_check = json.loads((root / CNN_OUTPUT / "validation_report.json").read_text(encoding="utf-8"))
    assert cnn_check["status"] == "passed" and not cnn_check["test_evaluated"]
    return datasets, audit


def summarize(output):
    """汇总三个模型各三个种子；种子只改变模型随机性，不改变数据划分。"""
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(output.glob("*_seed*/result.json"))]
    rows, per_class, runs = [], [], []
    for model_name in MODELS:
        group = [r for r in results if r["model"] == model_name]
        if not group:
            continue
        row = {"model": model_name, "completed_seeds": len(group)}
        for name, values in {
            "macro_f1": [r["validation"]["macro_f1"] for r in group],
            "accuracy": [r["validation"]["accuracy"] for r in group],
            "normal_recall": [r["validation"]["per_class_recall"][0] for r in group],
            "normal_false_alarm": [1-r["validation"]["per_class_recall"][0] for r in group],
            "dc_precision": [r["validation"]["per_class_precision"][6] for r in group],
            "dc_recall": [r["validation"]["per_class_recall"][6] for r in group],
            "dc_f1": [r["validation"]["per_class_f1"][6] for r in group],
            "grid_f1": [r["validation"]["per_class_f1"][4] for r in group],
        }.items():
            row[name+"_mean"] = float(np.mean(values))
            row[name+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
        for r in group:
            metrics = r["validation"]
            cm = np.asarray(metrics["confusion_matrix"])
            runs.append({"model": model_name, "seed": r["seed"], "macro_f1": metrics["macro_f1"],
                         "accuracy": metrics["accuracy"], "normal_to_dc": int(cm[0,6]),
                         "normal_to_grid": int(cm[0,4]), "dc_to_normal": int(cm[6,0]),
                         "normal_recall": metrics["per_class_recall"][0],
                         "dc_precision": metrics["per_class_precision"][6],
                         "dc_recall": metrics["per_class_recall"][6],
                         "dc_f1": metrics["per_class_f1"][6]})
            for label, name in enumerate(CLASS_NAMES):
                per_class.append({"model": model_name, "seed": r["seed"], "label": label, "name": name,
                                  "precision": metrics["per_class_precision"][label],
                                  "recall": metrics["per_class_recall"][label],
                                  "f1": metrics["per_class_f1"][label], "samples": metrics["support"][label]})
    report = {"scope": "original Raw Train/Val", "test_evaluated": False, "rows": rows,
              "completed_results": len(results), "new_rf_fits": sum(r["model"] != "CNN-Mean" for r in results)}
    write_json(output / "summary.json", report)
    for name, values in (("summary.csv", rows), ("runs.csv", runs), ("per_class.csv", per_class)):
        if values:
            with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(values[0]))
                writer.writeheader(); writer.writerows(values)
    return report


def run(root, output):
    """冻结版本/指纹后执行六次RF拟合，并引入三份原CNN结果；不覆盖已有实验。"""
    datasets, audit = load_context(root)
    source = {p: fingerprint(root / p) for p in SOURCE_FILES}
    references = {str(CNN_OUTPUT / f"baseline_seed{s}" / name): fingerprint(root / CNN_OUTPUT / f"baseline_seed{s}" / name)
                  for s in SEEDS for name in ("best.pt", "validation_predictions.npz", "result.json")}
    plan = {"seeds": list(SEEDS), "models": list(MODELS), "sklearn": sklearn.__version__,
            "runtime": {"numpy": np.__version__, "torch": torch.__version__, "joblib": joblib.__version__},
            "source_sha256": source, "reference_sha256": references,
            "prepared_sha256": audit["prepared_files_unchanged_sha256"],
            "rf_parameters": rf_parameters(42, datasets["train"].class_weights.numpy()),
            "protocol": "raw", "dataset_splits": ["train", "val"], "test_evaluated": False,
            "paper_url": "https://doi.org/10.3390/electronics15143029", "paper_exact_rf_parameters_verified": False,
            "uv_lock_sha256": fingerprint(root / "uv.lock"),
            "local_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    for p in (*SOURCE_FILES, "RF_BASELINE_PLAN.md", "pyproject.toml", "uv.lock"):
        target = snapshot / p; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / p, target)
    write_json(output / "plan.json", plan)
    for model_name in MODELS[:2]:
        x_train = feature_matrix(datasets["train"].inputs, model_name)
        x_val = feature_matrix(datasets["val"].inputs, model_name)
        y_train, y_val = (datasets[s].labels.numpy() for s in ("train", "val"))
        for seed in SEEDS:
            folder = output / f"{model_name}_seed{seed}"; folder.mkdir()
            params = rf_parameters(seed, datasets["train"].class_weights.numpy())
            forest = RandomForestClassifier(**params)
            print(f"开始拟合 {model_name} seed={seed}, 输入={x_train.shape}, 100树", flush=True)
            start = time.perf_counter(); forest.fit(x_train, y_train)
            seconds = time.perf_counter() - start
            assert np.array_equal(forest.classes_, np.arange(8))
            probability = forest.predict_proba(x_val); predictions = probability.argmax(1)
            train_metrics = score_predictions(y_train, forest.predict(x_train))
            metrics = score_predictions(y_val, predictions)
            joblib.dump(forest, folder / "model.joblib")
            np.savez_compressed(folder / "validation_predictions.npz", sample_index=datasets["val"].indices,
                                label=y_val, prediction=predictions, probability=probability)
            result = {"status": "complete", "model": model_name, "seed": seed, "input_features": x_train.shape[1],
                      "parameters": forest.get_params(), "validation": metrics, "training": train_metrics,
                      "fit_seconds": seconds, "new_fit": True, "test_evaluated": False,
                      "feature_sha256": {"train": array_digest(x_train), "val": array_digest(x_val)},
                      "model_sha256": fingerprint(folder / "model.joblib")}
            write_json(folder / "result.json", result); summarize(output)
            print(f"完成 {model_name} seed={seed}: val_Macro-F1={metrics['macro_f1']:.6f}, 耗时={seconds:.1f}s", flush=True)
    for seed in SEEDS:
        original = root / CNN_OUTPUT / f"baseline_seed{seed}"
        folder = output / f"CNN-Mean_seed{seed}"; folder.mkdir()
        old = json.loads((original / "result.json").read_text(encoding="utf-8"))
        shutil.copy2(original / "validation_predictions.npz", folder / "validation_predictions.npz")
        write_json(folder / "result.json", {"status": "complete", "model": "CNN-Mean", "seed": seed,
                    "validation": old["validation"], "new_fit": False, "test_evaluated": False,
                    "reference_dir": str(original.relative_to(root)), "parameters": old["parameters"]})
    summary = summarize(output)
    assert summary["completed_results"] == 9 and summary["new_rf_fits"] == 6
    return summary


def validate(root, output):
    """六个RF反序列化重放、三份CNN重载，独立重算指标并核对所有输入/来源指纹。"""
    datasets, audit = load_context(root)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert sklearn.__version__ == plan["sklearn"] and fingerprint(root / "uv.lock") == plan["uv_lock_sha256"]
    assert plan["runtime"] == {"numpy": np.__version__, "torch": torch.__version__, "joblib": joblib.__version__}
    for p, digest in plan["source_sha256"].items():
        assert fingerprint(root / p) == fingerprint(output / "source_snapshot" / p) == digest
    for p, digest in plan["reference_sha256"].items():
        assert fingerprint(root / p) == digest
    assert audit["prepared_files_unchanged_sha256"] == plan["prepared_sha256"]
    assert not list(output.glob("**/test_predictions.npz"))
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    val = datasets["val"]
    rf_count, cnn_count = 0, 0
    for model_name in MODELS:
        for seed in SEEDS:
            folder = output / f"{model_name}_seed{seed}"
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            assert result["status"] == "complete" and not result["test_evaluated"]
            if model_name == "CNN-Mean":
                model = FaultModel("mean")
                model.load_state_dict(torch.load(root / CNN_OUTPUT / f"baseline_seed{seed}/best.pt", map_location="cpu", weights_only=True))
                model.eval(); batches = []
                with torch.inference_mode():
                    for batch in torch.utils.data.DataLoader(val, batch_size=256, shuffle=False):
                        batches.append(model(batch)[0].softmax(1).numpy())
                probability = np.concatenate(batches); cnn_count += 1
            else:
                model = joblib.load(folder / "model.joblib")
                assert fingerprint(folder / "model.joblib") == result["model_sha256"]
                params = rf_parameters(seed, datasets["train"].class_weights.numpy())
                for name, value in params.items():
                    assert model.get_params()[name] == value, name
                for split in ("train", "val"):
                    features = feature_matrix(datasets[split].inputs, model_name)
                    assert array_digest(features) == result["feature_sha256"][split]
                    if split == "train":
                        train_metrics = score_predictions(datasets[split].labels.numpy(), model.predict(features))
                        for name, value in train_metrics.items():
                            np.testing.assert_allclose(value, result["training"][name], rtol=0, atol=1e-12)
                probability = model.predict_proba(feature_matrix(val.inputs, model_name)); rf_count += 1
            predictions = probability.argmax(1)
            with np.load(folder / "validation_predictions.npz", allow_pickle=False) as saved:
                assert np.array_equal(saved["sample_index"], val.indices)
                assert np.array_equal(saved["label"], val.labels.numpy())
                assert np.array_equal(saved["probability"], probability)
                assert np.array_equal(saved["prediction"], predictions)
            metrics = score_predictions(val.labels.numpy(), predictions)
            for name, value in metrics.items():
                np.testing.assert_allclose(value, result["validation"][name], rtol=0, atol=1e-12)
            print(f"重载及指标核验通过 {model_name} seed={seed}", flush=True)
    report = {"status": "passed", "rf_model_replays": rf_count, "cnn_checkpoint_replays": cnn_count,
              "all_predictions_exact": True, "sklearn_metrics_independently_checked": True,
              "same_validation_sample_indices_and_labels": True, "source_and_data_unchanged": True,
              "dataset_splits": ["train", "val"], "test_evaluated": False}
    write_json(output / "validation_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "validate"))
    args = parser.parse_args()
    root = Path.cwd()
    function = run if args.mode == "run" else validate
    print(json.dumps(function(root, root / OUTPUT), ensure_ascii=False, indent=2))
