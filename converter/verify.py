"""训练前的数据质量门：独立重算统计、校验清单，并可完整重建确认复现。"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch

from .data import load_samples, make_loaders
from .prepare import BRANCHES, FAMILY_MAP, PROTOCOLS, SPLITS, assert_no_leakage, build, fit_normalization, protocol_mask, write_json


def verify(prepared_dir, raw_dir=None):
    """任何检查失败立即抛出异常，不输出成功报告；raw_dir 启用从原文件重建。"""
    prepared_dir = Path(prepared_dir)
    samples = load_samples(prepared_dir)
    config = json.loads((prepared_dir / "config.json").read_text(encoding="utf-8"))
    audit = json.loads((prepared_dir / "audit.json").read_text(encoding="utf-8"))
    count = len(samples["label"])
    assert count == audit["joint_samples"]
    assert all(len(values) == count for values in samples.values())
    assert np.array_equal(samples["label"], FAMILY_MAP[samples["fine_label"]])
    assert_no_leakage(samples)
    saturation = np.zeros(count, dtype=bool)
    for name, (_, shape) in BRANCHES.items():
        x = samples[name]
        assert x.shape == (count, *shape) and x.dtype == np.float32 and np.isfinite(x).all()
        saturation |= np.max(np.abs(x.reshape(count, -1)), axis=1) >= config["saturation_threshold"]
        assert len(np.unique(samples[f"row_{name}"])) == count
    assert np.array_equal(saturation, samples["saturation_candidate"])
    for name in ("current", "dc"):
        delta = samples[f"timestamp_{name}"] - samples["timestamp"]
        assert np.array_equal(delta, samples[f"delta_{name}_s"])
        assert np.all(np.abs(delta) < config["max_delta_s_strict"])
    with (prepared_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == count
    for index, record in enumerate(records):
        assert int(record["sample"]) == index and record["split"] == SPLITS[samples["split"][index]]
        for key in ("block", "fine_label", "label", "row_current", "row_dc", "row_rms"):
            assert int(record[key]) == samples[key][index]
        for key in ("timestamp", "delta_current_s", "delta_dc_s"):
            assert float(record[key]) == samples[key][index]
        assert (record["saturation_candidate"] == "True") == samples["saturation_candidate"][index]
    checks = {"samples": count, "no_cross_split_blocks_or_source_records": True,
              "unique_source_record_per_triplet": True, "manifest_matches_arrays": True,
              "normalization_train_only_recomputed": True, "protocols": {}}
    for protocol in PROTOCOLS:
        expected = fit_normalization(samples, protocol, config["clip"])
        with np.load(prepared_dir / f"normalization_{protocol}.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(expected)
            for key, values in expected.items():
                assert np.array_equal(values, saved[key]), (protocol, key)
        loaders = make_loaders(prepared_dir, protocol, samples=samples)
        checks["protocols"][protocol] = {}
        for split, loader in loaders.items():
            batch = next(iter(loader))
            for name, (_, shape) in BRANCHES.items():
                assert tuple(batch[name].shape[1:]) == shape
                assert batch[name].dtype == torch.float32 and torch.isfinite(batch[name]).all()
                assert torch.abs(loader.dataset.inputs[name]).max() <= config["clip"]
            assert batch["label"].dtype == torch.int64
            counts = np.bincount(loader.dataset.labels.numpy(), minlength=8)
            assert np.all(counts > 0)
            checks["protocols"][protocol][split] = {"samples": len(loader.dataset), "class_counts": counts.tolist()}
    if raw_dir is not None:
        rebuilt_samples, rebuilt_stats, rebuilt_blocks, rebuilt_audit, rebuilt_config = build(
            raw_dir, config["manifest_seed"], config["max_delta_s_strict"], config["saturation_threshold"], config["clip"])
        assert samples.keys() == rebuilt_samples.keys()
        for key in samples:
            assert np.array_equal(samples[key], rebuilt_samples[key]), f"完整重建不一致: {key}"
        assert config == rebuilt_config and audit == rebuilt_audit
        with (prepared_dir / "blocks.csv").open(encoding="utf-8", newline="") as handle:
            saved_blocks = list(csv.DictReader(handle))
        assert saved_blocks == [{key: str(value) for key, value in block.items()} for block in rebuilt_blocks]
        for protocol in PROTOCOLS:
            with np.load(prepared_dir / f"normalization_{protocol}.npz", allow_pickle=False) as saved:
                for key in saved.files:
                    assert np.array_equal(saved[key], rebuilt_stats[protocol][key])
        checks["full_raw_rebuild_exact_match"] = True
        checks["raw_file_fingerprints_unchanged"] = True
    checks["status"] = "passed"
    write_json(prepared_dir / "validation_report.json", checks)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--raw-dir", type=Path, help="指定时从原始数据完整重建，验证精确复现")
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared_dir, args.raw_dir), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
