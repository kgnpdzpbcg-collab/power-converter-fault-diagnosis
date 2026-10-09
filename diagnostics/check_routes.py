"""执行细标签与频域可行性检查；模型只提取冻结特征，不训练新网络。"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from converter.data import ConverterDataset, load_samples
from converter.models import FaultModel
from converter.prepare import CLASS_NAMES, FAMILY_MAP, write_json
from converter.train import fingerprint
from .probes import (fit_probe, predict_scores, score_predictions,
                     paired_event_bootstrap, waveform_features)


SEEDS = (42, 123, 2026)


def build_views(meta, block_rows):
    """返回只含原Train/Val的三套索引，不改变原manifest。

    早晚视图在每个连续注入批次内切分；未读取Test批次标签。
    正常恢复记录继承其父block的注入批次，防止全局时间切分丢失故障类别。
    """
    active = sorted((row for row in block_rows if row["split"] in ("train", "val")), key=lambda r: int(r["block"]))
    runs = []
    for row in active:
        label = int(row["fine_label"])
        if not runs or runs[-1][0] != label:
            runs.append((label, []))
        runs[-1][1].append(int(row["block"]))
    early, late = [], []
    for _, blocks in runs:
        boundary = int(np.ceil(len(blocks) * 0.6))
        early.extend(blocks[:boundary]); late.extend(blocks[boundary:])
    train = meta["split"] == 0
    val = meta["split"] == 1
    clean = ~meta["saturation_candidate"]
    views = {"raw": (np.flatnonzero(train), np.flatnonzero(val)),
             "noncandidate": (np.flatnonzero(train & clean), np.flatnonzero(val & clean)),
             "early_late": (np.flatnonzero(train & clean & np.isin(meta["block"], early)),
                            np.flatnonzero(val & clean & np.isin(meta["block"], late)))}
    return views, [{"fine_label": label, "blocks": blocks} for label, blocks in runs]


def extract_features(prepared, samples, selected, checkpoint, output):
    """严格重现旧均值模型验证概率，并提取两个集合的冻结特征。

    μ采用旧forward完全相同的乘1/3后求和，不引入新的归一化或分类头。
    所有Dataset仅为train/val；Test记录没有被编码或生成新预测。
    """
    model = FaultModel("mean")
    model.load_state_dict(torch.load(checkpoint / "best.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval()
    mean = np.empty((len(selected), 64), dtype=np.float32)
    concat = np.empty((len(selected), 192), dtype=np.float32)
    raw = np.empty((len(selected), 103), dtype=np.float32)
    validation_probabilities = []
    for split in ("train", "val"):
        dataset = ConverterDataset(prepared, split, "raw", samples)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        with torch.inference_mode():
            for batch in loader:
                encoded = [model.encoders[name](batch[name]) for name in model.branches]
                stacked = torch.stack(encoded, 1)
                fused = (stacked * stacked.new_full((len(stacked), 3, 1), 1 / 3)).sum(1)
                positions = np.searchsorted(selected, batch["sample_index"].numpy())
                mean[positions] = fused.numpy()
                concat[positions] = torch.cat(encoded, 1).numpy()
                raw[positions] = torch.cat([batch[name].flatten(1) for name in model.branches], 1).numpy()
                if split == "val":
                    validation_probabilities.append(model.classifier(fused).softmax(1).numpy())
    with np.load(checkpoint / "validation_predictions.npz", allow_pickle=False) as old:
        assert np.array_equal(old["probability"], np.concatenate(validation_probabilities))
    np.savez_compressed(output, mean=mean, concat=concat, raw=raw)
    return {"mean": mean, "concat": concat, "raw": raw}


def frequency_features(samples, selected, prepared):
    """生成逐样本表示；频谱只作用于Train/Val原始波形，避免对Test计算FFT。"""
    current = waveform_features(samples["current"][selected])
    dc = waveform_features(samples["dc"][selected])
    rms = samples["rms"][selected].astype(np.float64)
    rms = np.sign(rms) * np.log1p(np.abs(rms))
    join = lambda *arrays: np.concatenate(arrays, 1)
    time_all = join(current["stats"], current["time_shape"], dc["stats"], dc["time_shape"], rms)
    # clip后的谱形仅为敏感性对照，主谱形不受这项非线性预处理影响。
    with np.load(prepared / "normalization_raw.npz", allow_pickle=False) as norm:
        x = samples["current"][selected].astype(np.float64)
        clipped = np.clip((x - norm["current_median"][None, :, None]) / norm["current_iqr"][None, :, None], -10, 10)
    result = {
        "current_stats": current["stats"], "current_time_shape": current["time_shape"],
        "current_spectrum": current["spectral_shape"],
        "current_stats_time": join(current["stats"], current["time_shape"]),
        "current_stats_spectrum": join(current["stats"], current["spectral_shape"]),
        "current_stats_time_spectrum": join(current["stats"], current["time_shape"], current["spectral_shape"]),
        "dc_spectrum": dc["spectral_shape"], "multisource_time": time_all,
        "multisource_time_spectrum": join(time_all, current["spectral_shape"], dc["spectral_shape"]),
        "current_spectrum_hann": waveform_features(samples["current"][selected], "hann")["spectral_shape"],
        "current_spectrum_clipped": waveform_features(clipped)["spectral_shape"],
    }
    # 与缓存重载保持同一输入精度，后续拟合内部再使用float64。
    return {name: x.astype(np.float32) for name, x in result.items()}, current["spectral_shape"].reshape(-1, 3, 10)


def evaluate_probe(output, meta, features, view, representation, method, target, train_indices, val_indices):
    """拟合并保存单个探针，预测阶段不读取真实标签；细类oracle诊断另行明确标记。"""
    labels = meta["fine_label"] if target in ("fine48", "harmonic7") else meta["label"]
    if target == "harmonic7":
        train_indices = train_indices[meta["label"][train_indices] == 5]
        val_indices = val_indices[meta["label"][val_indices] == 5]
        expected_classes = np.arange(32, 39)
    else:
        expected_classes = np.arange(48 if target == "fine48" else 8)
    classes = np.unique(labels[train_indices])
    identifier = f"{target}_{view}_{representation}_{method}"
    model = fit_probe(features[train_indices], labels[train_indices], meta["block"][train_indices], classes, method)
    scores = predict_scores(model, features[val_indices])
    predictions = classes[scores.argmax(1)]
    evaluation = score_predictions(labels[val_indices], predictions, meta["block"][val_indices], expected_classes)
    missing_val = [int(label) for label in sorted(set(expected_classes) - set(labels[val_indices]))]
    row = {"id": identifier, "target": target, "view": view, "representation": representation, "method": method,
           "dimensions": features.shape[1], "train_samples": len(train_indices), "val_samples": len(val_indices),
           "train_events": len(np.unique(meta["block"][train_indices])),
           "val_events": len(np.unique(meta["block"][val_indices])),
           "missing_train_classes": [int(label) for label in sorted(set(expected_classes) - set(classes))], "missing_val_classes": missing_val,
           "metrics": evaluation}
    if target == "fine48":
        family_rows = []
        for coarse in (1, 3, 4, 5, 6, 7):
            positions = np.flatnonzero(meta["label"][val_indices] == coarse)
            columns = np.flatnonzero(FAMILY_MAP[classes] == coarse)
            conditional_predictions = classes[columns][scores[positions][:, columns].argmax(1)]
            coarse_classes = np.flatnonzero(FAMILY_MAP == coarse)
            family_rows.append({"coarse": coarse, "name": CLASS_NAMES[coarse],
                                "fine_labels": coarse_classes.tolist(),
                                "oracle_coarse_label": True,
                                "metrics": score_predictions(labels[val_indices][positions], conditional_predictions,
                                                             meta["block"][val_indices][positions], coarse_classes)})
        row["within_family"] = family_rows
    folder = output / "probes"
    np.savez_compressed(folder / f"{identifier}_model.npz", **model)
    np.savez_compressed(folder / f"{identifier}_predictions.npz", positions=val_indices,
                        sample_index=meta["sample_index"][val_indices], label=labels[val_indices], prediction=predictions)
    write_json(folder / f"{identifier}.json", row)
    return row, predictions, val_indices


def label_audit(meta, block_rows, runs):
    """样本支持、时间批次和仅时间的故障细类对照；不用时间作为正式输入。"""
    rows = []
    for fine in range(48):
        row = {"fine_label": fine, "coarse": int(FAMILY_MAP[fine]), "class_name": CLASS_NAMES[FAMILY_MAP[fine]]}
        for split, index in (("train", 0), ("val", 1)):
            mask = (meta["split"] == index) & (meta["fine_label"] == fine)
            row.update({f"{split}_samples": int(mask.sum()), f"{split}_events": len(np.unique(meta["block"][mask])),
                        f"{split}_candidates": int(meta["saturation_candidate"][mask].sum())})
        row["time_batches"] = sum(run["fine_label"] == fine for run in runs) if fine else None
        rows.append(row)
    train = np.flatnonzero((meta["split"] == 0) & (meta["fine_label"] > 0))
    val = np.flatnonzero((meta["split"] == 1) & (meta["fine_label"] > 0))
    train_blocks = np.unique(meta["block"][train])
    centers = np.array([meta["timestamp"][train[meta["block"][train] == block]].mean() for block in train_blocks])
    labels = np.array([meta["fine_label"][train[meta["block"][train] == block]][0] for block in train_blocks])
    order = np.argsort(centers)
    centers, labels = centers[order], labels[order]
    right = np.clip(np.searchsorted(centers, meta["timestamp"][val]), 0, len(centers) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(centers[left] - meta["timestamp"][val]) <= np.abs(centers[right] - meta["timestamp"][val]), left, right)
    predictions = labels[nearest]
    return {"rows": rows, "injection_batches": [{"fine_label": run["fine_label"], "events": len(run["blocks"]),
                                                   "first_block": run["blocks"][0], "last_block": run["blocks"][-1]} for run in runs],
            "time_only_fault47": {"scope": "已知为故障的记录；仅训练事件时间中心点最近邻，不读取波形",
                                  "metrics": score_predictions(meta["fine_label"][val], predictions, meta["block"][val], np.arange(1, 48))}}


def run(root, output):
    """冻结计划和源码后执行，所有拟合保持相同固定配置；中断不自动覆盖。"""
    root, output = Path(root), Path(output)
    prepared = root / "artifacts/prepared/v0"
    output.mkdir(parents=True, exist_ok=False)
    (output / "probes").mkdir(); (output / "cache").mkdir(); (output / "source_snapshot").mkdir()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    samples = load_samples(prepared)
    selected = np.flatnonzero(samples["split"] != 2)
    meta = {key: samples[key][selected] for key in ("fine_label", "label", "block", "split", "timestamp", "saturation_candidate")}
    meta["sample_index"] = selected
    np.savez_compressed(output / "cache/meta.npz", **meta)
    block_rows = list(csv.DictReader((prepared / "blocks.csv").open(encoding="utf-8")))
    views, runs = build_views(meta, block_rows)
    write_json(output / "views.json", {name: {"train_positions": train.tolist(), "val_positions": val.tolist()}
                                      for name, (train, val) in views.items()})
    checkpoint_dirs = {seed: root / f"artifacts/experiments/v2_validation/mean_plain_seed{seed}" for seed in SEEDS}
    plan = {"scope": "Train/Val diagnostics only", "seeds": list(SEEDS), "ridge_regularization": 0.01,
            "window_length": 20, "frequency_axis": "bin1..10; cycles/sample=bin/20; fs_unverified",
            "bootstrap_repeats": 200, "threads": 2, "views": list(views), "test_evaluation": False,
            "local_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "manifest_sha256": fingerprint(prepared / "manifest.csv"),
            "samples_sha256": fingerprint(prepared / "samples.npz"),
            "normalization_sha256": fingerprint(prepared / "normalization_raw.npz"),
            "checkpoint_sha256": {str(seed): fingerprint(path / "best.pt") for seed, path in checkpoint_dirs.items()},
            "converter_sha256": {p.name: fingerprint(p) for p in (root / "converter").glob("*.py")}}
    for path in (root / "diagnostics").glob("*.py"):
        shutil.copy2(path, output / "source_snapshot" / path.name)
    shutil.copy2(root / "FEASIBILITY_CHECK_PLAN.md", output / "source_snapshot/FEASIBILITY_CHECK_PLAN.md")
    plan["diagnostic_sha256"] = {p.name: fingerprint(p) for p in (output / "source_snapshot").iterdir()}
    write_json(output / "plan.json", plan)
    audit = label_audit(meta, block_rows, runs)
    write_json(output / "label_audit.json", audit)
    rows = []
    for seed in SEEDS:
        representations = extract_features(prepared, samples, selected, checkpoint_dirs[seed], output / f"cache/embedding_seed{seed}.npz")
        for view, (train, val) in views.items():
            for name in ("mean", "concat"):
                for method in ("ridge", "prototype"):
                    row, _, _ = evaluate_probe(output, meta, representations[name], view, f"{name}_seed{seed}", method, "fine48", train, val)
                    rows.append(row)
                    print(f"fine {row['id']}: F1={row['metrics']['sample']['macro_f1']:.4f}", flush=True)
        if seed == 42:
            for view, (train, val) in views.items():
                for method in ("ridge", "prototype"):
                    row, _, _ = evaluate_probe(output, meta, representations["raw"], view, "raw103", method, "fine48", train, val)
                    rows.append(row)
    representations, spectra = frequency_features(samples, selected, prepared)
    np.savez_compressed(output / "cache/frequency_features.npz", **representations)
    spectrum_rows = []
    for split, index in (("train", 0), ("val", 1)):
        for fine in (0, *range(32, 39)):
            for flag in (False, True):
                mask = (meta["split"] == index) & (meta["fine_label"] == fine) & (meta["saturation_candidate"] == flag)
                if mask.any():
                    spectrum_rows.append({"split": split, "fine_label": fine, "candidate": flag, "samples": int(mask.sum()),
                                          "mean_power_share": spectra[mask].mean(0).tolist(),
                                          "median_power_share": np.median(spectra[mask], axis=0).tolist(),
                                          "q25_power_share": np.quantile(spectra[mask], 0.25, axis=0).tolist(),
                                          "q75_power_share": np.quantile(spectra[mask], 0.75, axis=0).tolist()})
    write_json(output / "spectral_distributions.json", {"axis": "FFT bin1..10", "rows": spectrum_rows})
    comparisons = []
    prototype_names = {"current_time_shape", "current_spectrum", "multisource_time", "multisource_time_spectrum"}
    for view, (train, val) in views.items():
        for target in ("coarse8", "harmonic7"):
            prediction_sets = {}
            for name, x in representations.items():
                for method in (("ridge", "prototype") if name in prototype_names else ("ridge",)):
                    row, predictions, positions = evaluate_probe(output, meta, x, view, name, method, target, train, val)
                    rows.append(row)
                    prediction_sets[(name, method)] = (predictions, positions)
                    print(f"frequency {row['id']}: F1={row['metrics']['sample']['macro_f1']:.4f}", flush=True)
            for a, b, method in (("current_stats", "current_stats_spectrum", "ridge"),
                                 ("current_stats_time", "current_stats_time_spectrum", "ridge"),
                                 ("multisource_time", "multisource_time_spectrum", "ridge"),
                                 ("multisource_time", "multisource_time_spectrum", "prototype")):
                pred_a, positions = prediction_sets[(a, method)]
                pred_b, other_positions = prediction_sets[(b, method)]
                assert np.array_equal(positions, other_positions)
                labels = meta["label"] if target == "coarse8" else meta["fine_label"]
                classes = np.arange(8) if target == "coarse8" else np.arange(32, 39)
                comparison = paired_event_bootstrap(labels[positions], pred_a, pred_b, meta["block"][positions], classes)
                comparison.update(view=view, target=target, reference=a, variant=b, method=method)
                comparisons.append(comparison)
    summary = {"status": "complete", "scope": "Train/Val only", "probe_runs": len(rows), "rows": rows,
               "paired_frequency_comparisons": comparisons, "original_validation_probabilities_reproduced_seeds": list(SEEDS),
               "views": {name: {"train_samples": len(train), "val_samples": len(val)} for name, (train, val) in views.items()},
               "no_new_encoder_training": True, "test_evaluated": False}
    write_json(output / "summary.json", summary)
    flat = [{"id": r["id"], "target": r["target"], "view": r["view"], "method": r["method"],
             "representation": r["representation"], "dimensions": r["dimensions"],
             "train_samples": r["train_samples"], "val_samples": r["val_samples"],
             "macro_f1": r["metrics"]["sample"]["macro_f1"], "accuracy": r["metrics"]["sample"]["accuracy"],
             "event_macro_f1": r["metrics"]["event_balanced"]["macro_f1"]} for r in rows]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    print(json.dumps({"completed_probe_fits": len(rows), "output": str(output)}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/diagnostics/feasibility_v1"))
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.output_dir)


if __name__ == "__main__":
    main()
