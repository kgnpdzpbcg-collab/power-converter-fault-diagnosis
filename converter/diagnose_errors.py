"""仅使用训练/验证数据诊断正常/直流混淆、裁剪信息损失和类别权重影响。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import load_samples, normalize
from .metrics import classification_metrics
from .prepare import BRANCHES, FAMILY_MAP, episode_blocks, read_source, write_json


def recovery_context(samples, anchor):
    """从完整原始 RMS 事件得到正常样本距上次故障结束的秒数。

    元数据只用于离线分组，绝不能作为模型输入。初始正常段的 elapsed
    使用 NaN 标记，因为它之前没有故障；不把这一段错误归入长期恢复。
    """
    block, starts = episode_blocks(anchor.label)
    limits = np.r_[starts[1:], len(anchor.label)]
    ends = np.array([start + np.flatnonzero(anchor.label[start:end] != 0)[-1]
                     for start, end in zip(starts, limits)])
    sorted_position = np.empty(len(anchor.row), dtype=np.int64)
    sorted_position[anchor.row] = np.arange(len(anchor.row))
    positions = sorted_position[samples["row_rms"]]
    if not np.array_equal(block[positions], samples["block"]):
        raise ValueError("原始事件与固定 manifest 不一致")
    elapsed = samples["timestamp"] - anchor.time[ends[samples["block"]]]
    elapsed = np.where(positions > ends[samples["block"]], elapsed, np.nan)
    preceding_fine = anchor.label[starts[samples["block"]]]
    return elapsed, preceding_fine


def overlap_counts(x, labels):
    """严格逐值相等的训练向量是否同时对应正常和直流标签；不做四舍五入。

    若出现冲突，任何仅使用该向量的确定性分类器都无法同时正确识别这些
    记录。但单分支冲突不能证明三源联合输入同样冲突。
    """
    _, inverse = np.unique(np.ascontiguousarray(x), axis=0, return_inverse=True)
    low = np.full(inverse.max() + 1, 8, dtype=np.int64)
    high = np.full(inverse.max() + 1, -1, dtype=np.int64)
    np.minimum.at(low, inverse, labels)
    np.maximum.at(high, inverse, labels)
    conflicting = low != high
    return {"conflicting_vectors": int(conflicting.sum()),
            "normal_records_in_conflicts": int(((labels == 0) & conflicting[inverse]).sum()),
            "dc_records_in_conflicts": int(((labels == 6) & conflicting[inverse]).sum())}


def save_csv(path, records):
    """带 UTF-8 BOM 的表格，方便本机 Excel 直接查看中文分组。"""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(prepared_dir, previous_runs, raw_dir, output_dir):
    """保存诊断证据；不读取旧测试预测或 result.json 中的测试指标。"""
    prepared_dir, previous_runs, output_dir = map(Path, (prepared_dir, previous_runs, output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    samples = load_samples(prepared_dir)
    audit = json.loads((prepared_dir / "audit.json").read_text(encoding="utf-8"))
    anchor = read_source(Path(raw_dir) / "id_mea_10.json", 10, (3,), 1e8)
    if anchor.audit["sha256"] != audit["sources"]["rms"]["sha256"]:
        raise ValueError("原始 RMS 指纹变化，不能沿用旧事件")
    elapsed, preceding_fine = recovery_context(samples, anchor)
    val_indices = np.flatnonzero(samples["split"] == 1)
    train_mask = samples["split"] == 0
    with np.load(prepared_dir / "normalization_raw.npz", allow_pickle=False) as stats:
        weights = stats["class_weights"].copy()
        normalized = {name: normalize(samples[name][train_mask], stats[f"{name}_median"], stats[f"{name}_iqr"], float(stats["clip"]))
                      for name in BRANCHES}
        clipping = []
        for split_id, split_name in ((0, "train"), (1, "val")):
            for fine in (0, *range(39, 46)):
                mask = (samples["split"] == split_id) & (samples["fine_label"] == fine)
                for name in BRANCHES:
                    x = samples[name][mask]
                    shape = (1, -1, 1) if x.ndim == 3 else (1, -1)
                    scaled = (x - stats[f"{name}_median"].reshape(shape)) / stats[f"{name}_iqr"].reshape(shape)
                    clipping.append({"split": split_name, "fine_label": fine, "branch": name,
                                     "records": int(mask.sum()), "feature_fraction_clipped": float((np.abs(scaled) > 10).mean()),
                                     "record_fraction_any_clip": float(np.any(np.abs(scaled.reshape(len(x), -1)) > 10, axis=1).mean())})
    overlap = {}
    binary = np.isin(samples["label"][train_mask], [0, 6])
    binary_labels = samples["label"][train_mask][binary]
    for name in BRANCHES:
        raw = samples[name][train_mask][binary].reshape(int(binary.sum()), -1)
        clipped = normalized[name][binary].reshape(int(binary.sum()), -1)
        overlap[name] = {"raw": overlap_counts(raw, binary_labels), "normalized_clipped": overlap_counts(clipped, binary_labels)}
    joined = np.concatenate([normalized[name][binary].reshape(int(binary.sum()), -1) for name in BRANCHES], axis=1)
    overlap["all_normalized_clipped"] = overlap_counts(joined, binary_labels)
    normal_rows, fault_rows, corrections = [], [], []
    age_bins = [("故障后0至2秒", 0, 2), ("故障后2至5秒", 2, 5), ("故障后5至10秒", 5, 10),
                ("故障后10至20秒", 10, 20), ("故障后20秒以上", 20, np.inf)]
    for model in ("mean", "concat64", "dynamic"):
        for seed in (42, 123, 2026):
            folder = previous_runs / f"raw_{model}_s{seed}"
            with np.load(folder / "validation_predictions.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["sample_index"], val_indices):
                    raise ValueError("旧验证预测与当前固定验证索引不一致")
                prediction, labels, probabilities = saved["prediction"], saved["label"], saved["probability"]
                corrected = (probabilities.astype(np.float64) / weights).argmax(axis=1)
                for name, pred in (("原始argmax", prediction), ("除以训练类别权重的固定代数对照", corrected)):
                    metrics = classification_metrics(np.bincount(labels * 8 + pred, minlength=64).reshape(8, 8))
                    corrections.append({"model": model, "seed": seed, "decision": name,
                                        "macro_f1": metrics["macro_f1"], "normal_recall": metrics["per_class_recall"][0],
                                        "dc_f1": metrics["per_class_f1"][6], "dc_precision": metrics["per_class_precision"][6],
                                        "dc_recall": metrics["per_class_recall"][6]})
                for name, low, high in age_bins:
                    mask = (labels == 0) & (elapsed[val_indices] > low) & (elapsed[val_indices] <= high)
                    if mask.any():
                        normal_rows.append({"model": model, "seed": seed, "group": name, "records": int(mask.sum()),
                                            "any_fault_rate": float((prediction[mask] != 0).mean()),
                                            "predicted_dc_rate": float((prediction[mask] == 6).mean()),
                                            "predicted_grid_rate": float((prediction[mask] == 4).mean())})
                for coarse in range(1, 8):
                    mask = (labels == 0) & (FAMILY_MAP[preceding_fine[val_indices]] == coarse) & np.isfinite(elapsed[val_indices])
                    if mask.any():
                        normal_rows.append({"model": model, "seed": seed, "group": f"正常且前次故障属于粗类{coarse}", "records": int(mask.sum()),
                                            "any_fault_rate": float((prediction[mask] != 0).mean()),
                                            "predicted_dc_rate": float((prediction[mask] == 6).mean()),
                                            "predicted_grid_rate": float((prediction[mask] == 4).mean())})
                for fine in range(39, 46):
                    mask = samples["fine_label"][val_indices] == fine
                    if mask.any():
                        fault_rows.append({"model": model, "seed": seed, "fine_label": fine, "records": int(mask.sum()),
                                           "dc_recall": float((prediction[mask] == 6).mean()),
                                           "predicted_normal_rate": float((prediction[mask] == 0).mean()),
                                           "predicted_grid_rate": float((prediction[mask] == 4).mean())})
    save_csv(output_dir / "normal_recovery.csv", normal_rows)
    save_csv(output_dir / "fine_dc.csv", fault_rows)
    save_csv(output_dir / "clipping.csv", clipping)
    save_csv(output_dir / "class_weight_correction.csv", corrections)
    result = {"scope": "训练/验证集；测试预测与测试指标未读取", "exact_training_vector_conflicts": overlap,
              "class_weights": weights.tolist(), "label_changes": False, "raw_data_changes": False,
              "limits": ["恢复时间仅为原始标签事件的时间位置，不证明物理恢复已完成", "代数权重对照不保证网络概率已校准",
                         "单分支相同向量的冲突不代表三源联合观测不可分"]}
    write_json(output_dir / "diagnostics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--previous-runs", type=Path, default=Path("artifacts/experiments/v1"))
    parser.add_argument("--raw-dir", type=Path, default=Path("../数据"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/diagnostics/v2"))
    args = parser.parse_args()
    print(json.dumps(run(args.prepared_dir, args.previous_runs, args.raw_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
