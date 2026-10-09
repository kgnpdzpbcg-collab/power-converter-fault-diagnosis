"""从保存的训练特征重新拟合并重放全部诊断，核查边界、指标和文件哈希。"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from converter.data import load_samples
from converter.prepare import FAMILY_MAP, write_json
from converter.train import fingerprint
from .check_routes import build_views, frequency_features, label_audit
from .probes import fit_probe, predict_scores, score_predictions, paired_event_bootstrap


def read_npz(path):
    """立即关闭文件句柄，返回不允许pickle的数值缓存。"""
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def same_numbers(actual, expected):
    """递归检查完整指标，避免只核对汇总F1而漏掉混淆矩阵或类别支持。"""
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            same_numbers(actual[key], expected[key])
    elif isinstance(expected, list) and expected and isinstance(expected[0], dict):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            same_numbers(left, right)
    elif isinstance(expected, (list, int, float)):
        assert np.allclose(actual, expected, atol=1e-11, rtol=1e-11)
    else:
        assert actual == expected


def validate(root, output):
    """所有拟合只用保存的Train索引；重新FFT也只取原Train/Val记录。

    检查对象包括132个探针、24个配对bootstrap及三套视图。
    这种重放验证计算与可复现性，不能证明独立工况泛化。
    """
    prepared = root / "artifacts/prepared/v0"
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    for field, name in (("manifest_sha256", "manifest.csv"), ("samples_sha256", "samples.npz"),
                        ("normalization_sha256", "normalization_raw.npz")):
        assert fingerprint(prepared / name) == plan[field]
    for seed, digest in plan["checkpoint_sha256"].items():
        assert fingerprint(root / f"artifacts/experiments/v2_validation/mean_plain_seed{seed}/best.pt") == digest
    for name, digest in plan["converter_sha256"].items():
        assert fingerprint(root / "converter" / name) == digest
    for name, digest in plan["diagnostic_sha256"].items():
        assert fingerprint(output / "source_snapshot" / name) == digest
        source = root / name if name.endswith(".md") else root / "diagnostics" / name
        assert fingerprint(source) == digest
    config = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    for name, measurement in (("current", 2), ("dc", 3), ("rms", 10)):
        assert fingerprint(root.parent / "数据" / f"id_mea_{measurement}.json") == config["source_sha256"][name]

    samples = load_samples(prepared)
    meta = read_npz(output / "cache/meta.npz")
    selected = np.flatnonzero(samples["split"] != 2)
    assert np.array_equal(selected, meta["sample_index"])
    for key in meta:
        if key != "sample_index":
            assert np.array_equal(meta[key], samples[key][selected])
    assert set(meta["split"]) == {0, 1}
    block_rows = list(csv.DictReader((prepared / "blocks.csv").open(encoding="utf-8")))
    views, runs = build_views(meta, block_rows)
    saved_views = json.loads((output / "views.json").read_text(encoding="utf-8"))
    for name, (train, val) in views.items():
        assert np.array_equal(train, saved_views[name]["train_positions"])
        assert np.array_equal(val, saved_views[name]["val_positions"])
        assert (meta["split"][train] == 0).all() and (meta["split"][val] == 1).all()
        assert not set(meta["block"][train]) & set(meta["block"][val])
        if name != "raw":
            assert not meta["saturation_candidate"][np.r_[train, val]].any()
    same_numbers(label_audit(meta, block_rows, runs), json.loads((output / "label_audit.json").read_text(encoding="utf-8")))

    cache = {seed: read_npz(output / f"cache/embedding_seed{seed}.npz") for seed in plan["seeds"]}
    frequencies = read_npz(output / "cache/frequency_features.npz")
    rebuilt, _ = frequency_features(samples, selected, prepared)
    assert set(rebuilt) == set(frequencies)
    for name in frequencies:
        assert np.array_equal(rebuilt[name], frequencies[name])
    del rebuilt
    predictions_by_id = {}
    for row in summary["rows"]:
        train, val = views[row["view"]]
        target = row["target"]
        labels = meta["label"] if target == "coarse8" else meta["fine_label"]
        classes = np.arange(8) if target == "coarse8" else (np.arange(32, 39) if target == "harmonic7" else np.arange(48))
        if target == "harmonic7":
            train = train[meta["label"][train] == 5]
            val = val[meta["label"][val] == 5]
        representation = row["representation"]
        if "_seed" in representation:
            name, seed = representation.split("_seed")
            x = cache[int(seed)][name]
        elif representation == "raw103":
            x = cache[42]["raw"]
        else:
            x = frequencies[representation]
        model = read_npz(output / "probes" / f"{row['id']}_model.npz")
        refitted = fit_probe(x[train], labels[train], meta["block"][train], np.unique(labels[train]), row["method"])
        assert set(model) == set(refitted)
        for key in model:
            assert np.array_equal(model[key], refitted[key]), (row["id"], key)
        scores = predict_scores(model, x[val])
        predictions = model["classes"][scores.argmax(1)]
        saved = read_npz(output / "probes" / f"{row['id']}_predictions.npz")
        for key, actual in (("prediction", predictions), ("positions", val),
                            ("sample_index", selected[val]), ("label", labels[val])):
            assert np.array_equal(actual, saved[key]), (row["id"], key)
        same_numbers(score_predictions(labels[val], predictions, meta["block"][val], classes), row["metrics"])
        for family in row.get("within_family", []):
            position = np.flatnonzero(meta["label"][val] == family["coarse"])
            column = np.flatnonzero(FAMILY_MAP[model["classes"]] == family["coarse"])
            conditional = model["classes"][column][scores[position][:, column].argmax(1)]
            same_numbers(score_predictions(labels[val][position], conditional, meta["block"][val][position],
                                           np.asarray(family["fine_labels"])), family["metrics"])
        predictions_by_id[row["id"]] = (predictions, val)
    for comparison in summary["paired_frequency_comparisons"]:
        prefix = f"{comparison['target']}_{comparison['view']}_"
        a, val = predictions_by_id[prefix + comparison["reference"] + "_" + comparison["method"]]
        b, other = predictions_by_id[prefix + comparison["variant"] + "_" + comparison["method"]]
        assert np.array_equal(val, other)
        labels = meta["label"] if comparison["target"] == "coarse8" else meta["fine_label"]
        classes = np.arange(8) if comparison["target"] == "coarse8" else np.arange(32, 39)
        result = paired_event_bootstrap(labels[val], a, b, meta["block"][val], classes)
        same_numbers(result, {key: comparison[key] for key in result})
    report = {"status": "passed", "probe_refits_and_prediction_replays": len(summary["rows"]),
              "paired_bootstrap_replays": len(summary["paired_frequency_comparisons"]),
              "frequency_cache_reproduced": True, "views_and_all_metrics_reproduced": True,
              "data_manifest_normalization_checkpoints_and_old_code_unchanged": True,
              "raw_source_hashes_unchanged": True, "train_val_samples": len(selected),
              "test_feature_or_prediction_computed": False}
    write_json(output / "validation_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/diagnostics/feasibility_v1")
    args = parser.parse_args()
    print(json.dumps(validate(Path.cwd(), Path(args.output)), ensure_ascii=False, indent=2))
