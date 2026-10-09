"""V3 数据与旧源码审计；只重算训练统计，不构造测试 Dataset。"""

import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converter.data import load_samples
from converter.prepare import assert_no_leakage, fit_normalization, write_json
from converter.train import fingerprint


def main():
    """核验真实原始文件哈希、固定清单、训练归一化和未改动的旧代码。"""
    root = Path(__file__).resolve().parents[1]
    prepared = root / "artifacts/prepared/v0"
    config = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    sources = {"current": "id_mea_2.json", "dc": "id_mea_3.json", "rms": "id_mea_10.json"}
    assert all(fingerprint(root.parent / "数据" / filename) == config["source_sha256"][branch]
               for branch, filename in sources.items())
    samples = load_samples(prepared)
    assert len(samples["label"]) == 58153
    assert_no_leakage(samples)
    with (prepared / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(samples["label"])
    for index, row in enumerate(rows):
        assert int(row["sample"]) == index
        assert row["split"] == ("train", "val", "test")[samples["split"][index]]
        for key in ("block", "label", "fine_label", "row_current", "row_dc", "row_rms"):
            assert int(row[key]) == samples[key][index]
    expected = fit_normalization(samples, "raw", config["clip"])
    with np.load(prepared / "normalization_raw.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(expected)
        assert all(np.array_equal(archive[key], expected[key]) for key in archive.files)
    snapshot = root / "artifacts/experiments/v2_validation/source_snapshot"
    old_modules = list(snapshot.glob("*.py"))
    assert old_modules
    assert all(fingerprint(path) == fingerprint(root / "converter" / path.name) for path in old_modules)
    report = {"status": "passed", "samples": 58153,
              "raw_file_sha256_unchanged": True, "no_cross_split_events_or_source_rows": True,
              "manifest_matches_samples": True, "raw_train_only_statistics_recomputed_exactly": True,
              "old_v2_code_modules_unchanged": len(old_modules), "no_test_dataset_constructed": True}
    write_json(root / "artifacts/checks/v3_data_audit.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
