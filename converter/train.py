"""统一训练和评估：仅验证集选模型，训练结束后加载最优权重评估测试集。"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .data import load_samples, make_loaders
from .metrics import classification_metrics
from .models import FaultModel
from .prepare import write_json
from .training_trace import EpochTrace


@dataclass(frozen=True)
class TrainConfig:
    """正式实验初始配置；每次写入独立目录，不从测试表现修改超参数。"""
    seed: int = 42
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    threads: int = 2


def set_seed(seed, threads):
    """固定 Python、NumPy、Torch 与 CPU 运算；数据划分不在此函数内变动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fingerprint(path):
    """文件 SHA-256 用于绑定代码、固定 manifest、归一化和依赖版本。"""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evaluate(model, loader, criterion, collect=False):
    """评估不更新参数；损失按真实类别权重之和聚合，避免末批样本偏差。

    collect 为真时保存联合样本编号、类别概率与门控权重，便于离线核验。
    训练期间只对验证集调用；测试集在最优 checkpoint 重载后调用一次。
    """
    model.eval()
    confusion = np.zeros((8, 8), dtype=np.int64)
    total_loss, total_weight = 0.0, 0.0
    collected = {key: [] for key in ("sample_index", "label", "prediction", "probability", "weights")}
    with torch.inference_mode():
        for batch in loader:
            labels = batch["label"]
            logits, weights = model(batch)
            if not torch.isfinite(logits).all():
                raise ValueError("评估输出包含 NaN/Inf")
            prediction = logits.argmax(dim=1)
            total_loss += float(criterion(logits, labels))
            total_weight += float(criterion.weight[labels].sum())
            confusion += np.bincount((labels * 8 + prediction).numpy(), minlength=64).reshape(8, 8)
            if collect:
                for key, value in (("sample_index", batch["sample_index"]), ("label", labels),
                                   ("prediction", prediction), ("probability", logits.softmax(dim=1))):
                    collected[key].append(value.numpy())
                if weights is not None:
                    collected["weights"].append(weights.numpy())
    metrics = classification_metrics(confusion)
    metrics["loss"] = total_loss / total_weight
    arrays = {key: np.concatenate(value) for key, value in collected.items() if value} if collect else {}
    return metrics, arrays


def gate_summary(predictions, samples, branches):
    """汇总实际权重及按类别均值，只描述模型行为，不宣称因果或专门化成立。"""
    if "weights" not in predictions:
        return None
    weights = predictions["weights"].astype(np.float64)
    indices = predictions["sample_index"]
    entropy = -(weights * np.log(np.clip(weights, 1e-12, 1))).sum(axis=1)
    return {"branches": list(branches), "mean": weights.mean(axis=0).tolist(),
            "std_across_samples": weights.std(axis=0).tolist(), "mean_entropy": float(entropy.mean()),
            "fraction_max_weight_over_0_95": float((weights.max(axis=1) > 0.95).mean()),
            "by_coarse_class": {str(label): weights[predictions["label"] == label].mean(axis=0).tolist() for label in range(8)},
            "by_saturation_candidate": {str(flag): weights[samples["saturation_candidate"][indices] == flag].mean(axis=0).tolist()
                                        for flag in (False, True) if np.any(samples["saturation_candidate"][indices] == flag)}}


def latency(model, dataset):
    """单样本 CPU 前向耗时：10 次预热、100 次测量，含编码和分类、不含加载。

    同一机器、同一线程数测量；非部署指标，不跨硬件比较绝对数值。
    """
    batch = {name: dataset.inputs[name][:1] for name in model.branches}
    durations = []
    model.eval()
    with torch.inference_mode():
        for _ in range(10):
            model(batch)
        for _ in range(100):
            start = time.perf_counter()
            model(batch)
            durations.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(durations)), "p95_ms": float(np.quantile(durations, 0.95)), "repeats": 100}


def train(prepared_dir, output_dir, fusion="v0", branches=("current", "dc", "rms"),
          protocol="raw", config=TrainConfig(), samples=None, evaluate_test=True,
          model_factory=FaultModel, model_options=None, trace=False):
    """训练一个不可覆盖的实验，保存日志、最优权重和完整配置。

    evaluate_test=False 时只创建训练、验证视图，不评估测试集。
    model_factory/model_options 用于受控模型变体，训练和恢复使用同一构造器。
    trace=True 时额外保存逐轮分支权重及梯度，不改变损失或选模规则。
    最优 checkpoint 按验证 Macro-F1 选择；相同分数保留较早模型。
    """
    prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
    gate_path = prepared_dir / "validation_report.json"
    validation = json.loads(gate_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or not validation.get("full_raw_rebuild_exact_match"):
        raise ValueError("数据质量门未通过完整原始数据重建验证，禁止训练")
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(config.seed, config.threads)
    samples = load_samples(prepared_dir) if samples is None else samples
    selected_splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    loaders = make_loaders(prepared_dir, protocol, config.seed, config.batch_size, samples, selected_splits)
    model_options = {} if model_options is None else dict(model_options)
    model = model_factory(fusion, branches, **model_options)
    weights = loaders["train"].dataset.class_weights
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    metadata = {"fusion": fusion, "branches": list(branches), "protocol": protocol, "training": asdict(config),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "python": __import__("sys").version, "numpy": np.__version__, "torch": torch.__version__, "device": "cpu",
                "manifest_sha256": fingerprint(prepared_dir / "manifest.csv"),
                "normalization_sha256": fingerprint(prepared_dir / f"normalization_{protocol}.npz"),
                "source_sha256": json.loads((prepared_dir / "config.json").read_text(encoding="utf-8"))["source_sha256"],
                "code_sha256": {path.name: fingerprint(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
                "lock_sha256": fingerprint(Path(__file__).parent.parent / "uv.lock"),
                "test_evaluation_requested": evaluate_test, "dataset_splits": list(loaders),
                "model_class": f"{model_factory.__module__}.{model_factory.__name__}",
                "model_options": model_options, "training_trace": trace}
    write_json(output_dir / "experiment.json", metadata)
    best_score, best_epoch, stale = -1.0, 0, 0
    start = time.perf_counter()
    history = []
    trace_history = []
    fields = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_accuracy", "learning_rate", "seconds"]
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config.max_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            total_loss, total_weight = 0.0, 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_trace = EpochTrace(branches) if trace else None
            for batch in loaders["train"]:
                labels = batch["label"]
                optimizer.zero_grad(set_to_none=True)
                logits, batch_weights = model(batch)
                numerator = criterion(logits, labels)
                denominator = weights[labels].sum()
                loss = numerator / denominator
                if not torch.isfinite(loss):
                    raise ValueError(f"第 {epoch} 轮损失出现 NaN/Inf")
                loss.backward()
                if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                    raise ValueError(f"第 {epoch} 轮梯度出现 NaN/Inf")
                if epoch_trace is not None:
                    epoch_trace.observe(model, batch_weights)
                optimizer.step()
                total_loss += float(numerator.detach())
                total_weight += float(denominator)
            val_metrics, _ = evaluate(model, loaders["val"], criterion)
            score = val_metrics["macro_f1"]
            record = {"epoch": epoch, "train_loss": total_loss / total_weight, "val_loss": val_metrics["loss"],
                      "val_macro_f1": score, "val_accuracy": val_metrics["accuracy"],
                      "learning_rate": current_lr, "seconds": time.perf_counter() - epoch_start}
            writer.writerow(record)
            handle.flush()
            history.append(record)
            if epoch_trace is not None:
                trace_history.append(epoch_trace.record(epoch))
            if score > best_score + 1e-12:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save(model.state_dict(), output_dir / "best.pt")
            else:
                stale += 1
            scheduler.step(score)
            print(f"{output_dir.name} epoch={epoch:03d} loss={record['train_loss']:.4f} val_F1={score:.4f} best={best_score:.4f} time={record['seconds']:.2f}s", flush=True)
            if stale >= config.patience:
                break
    # 新建实例重载，顺带验证 checkpoint 足以独立恢复预测。
    if trace_history:
        with (output_dir / "gate_trace.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace_history[0]))
            writer.writeheader()
            writer.writerows(trace_history)
    restored = model_factory(fusion, branches, **model_options)
    restored.load_state_dict(torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True), strict=True)
    val_metrics, val_predictions = evaluate(restored, loaders["val"], criterion, collect=True)
    if abs(val_metrics["macro_f1"] - best_score) > 1e-12:
        raise ValueError("重载 checkpoint 后验证指标与最优日志不一致")
    result = {"status": "complete", "fusion": fusion, "branches": list(branches), "protocol": protocol,
              "seed": config.seed, "parameters": metadata["parameters"], "best_epoch": best_epoch,
              "epochs_run": len(history), "best_val_macro_f1": best_score, "validation": val_metrics,
              "train_seconds": time.perf_counter() - start,
              "train_loss_first": history[0]["train_loss"], "train_loss_last": history[-1]["train_loss"],
              "checkpoint_bytes": (output_dir / "best.pt").stat().st_size,
              "latency_cpu": latency(restored, loaders["val"].dataset)}
    np.savez_compressed(output_dir / "validation_predictions.npz", **val_predictions)
    result["validation_gate"] = gate_summary(val_predictions, samples, branches)
    if evaluate_test:
        test_metrics, test_predictions = evaluate(restored, loaders["test"], criterion, collect=True)
        result["test"] = test_metrics
        result["test_gate"] = gate_summary(test_predictions, samples, branches)
        np.savez_compressed(output_dir / "test_predictions.npz", **test_predictions)
    write_json(output_dir / "result.json", result)
    print(f"完成 {output_dir}: best_epoch={best_epoch}, val_F1={best_score:.4f}" + (f", test_F1={result['test']['macro_f1']:.4f}" if evaluate_test else ", 未评估测试集"), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fusion", choices=("v0", "concat", "mean", "static", "dynamic"), default="v0")
    parser.add_argument("--branches", nargs="+", default=["current", "dc", "rms"])
    parser.add_argument("--protocol", choices=("raw", "exclude_saturation"), default="raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--validation-only", action="store_true", help="用于短流程验证，不评估测试集")
    args = parser.parse_args()
    train(args.prepared_dir, args.output_dir, args.fusion, tuple(args.branches), args.protocol,
          TrainConfig(seed=args.seed, max_epochs=args.max_epochs, threads=args.threads), evaluate_test=not args.validation_only)


if __name__ == "__main__":
    main()
