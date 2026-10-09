"""重新核查现有划分和直接泄露，不改变旧数据质量报告或创建测试预测。"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from converter.data import ConverterDataset, load_samples, normalize
from converter.models import FaultModel
from converter.prepare import BRANCHES, FAMILY_MAP, SPLITS, assert_no_leakage, build, write_json
from converter.train import fingerprint


def duplicate_vectors(samples, prepared, normalized):
    """检查完整三源输入是否跨集合完全相同；单通道相似不直接判为泄露。

    使用数值的字节表示做精确比较，统一正负零；包含Test仅用于排重，
    不产生任何Test模型预测，不以其分布选择预处理或候选架构。
    """
    arrays = []
    with np.load(prepared / "normalization_raw.npz", allow_pickle=False) as stats:
        for name in BRANCHES:
            x = samples[name]
            if normalized:
                x = normalize(x, stats[f"{name}_median"], stats[f"{name}_iqr"], float(stats["clip"]))
            arrays.append(x.reshape(len(x), -1))
    joined = np.ascontiguousarray(np.concatenate(arrays, axis=1))
    joined[joined == 0] = 0
    vectors = joined.view(np.dtype((np.void, joined.dtype.itemsize * joined.shape[1]))).ravel()
    _, inverse, counts = np.unique(vectors, return_inverse=True, return_counts=True)
    membership = np.zeros(len(counts), dtype=np.uint8)
    np.bitwise_or.at(membership, inverse, 1 << samples["split"])
    cross = (membership & (membership - 1)) != 0
    return {"cross_split_identical_vector_groups": int(cross.sum()),
            "records_in_cross_split_identical_groups": int(counts[cross].sum())}


def audit(root, output):
    """从原始JSON完整重建，核验标签映射、事件/记录隔离、统计来源和模型入口。

    报告无直接泄露不等于证明跨批次泛化；共享注入批次和历史Test观察单独说明。
    只有检查全部通过才返回passed，原prepare目录中的文件不被覆盖。
    """
    prepared = root / "artifacts/prepared/v0"
    output.mkdir(parents=True, exist_ok=False)
    before = {name: fingerprint(prepared / name) for name in
              ("samples.npz", "manifest.csv", "blocks.csv", "normalization_raw.npz", "normalization_exclude_saturation.npz")}
    samples = load_samples(prepared)
    config = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    saved_audit = json.loads((prepared / "audit.json").read_text(encoding="utf-8"))
    assert_no_leakage(samples)
    assert np.array_equal(samples["label"], FAMILY_MAP[samples["fine_label"]])
    with (prepared / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(samples["label"])
    for index, row in enumerate(rows):
        assert int(row["sample"]) == index and row["split"] == SPLITS[samples["split"][index]]
        for key in ("block", "fine_label", "label", "row_current", "row_dc", "row_rms"):
            assert int(row[key]) == samples[key][index]

    raw_duplicates = duplicate_vectors(samples, prepared, False)
    normalized_duplicates = duplicate_vectors(samples, prepared, True)
    assert raw_duplicates["cross_split_identical_vector_groups"] == 0, raw_duplicates
    assert normalized_duplicates["cross_split_identical_vector_groups"] == 0, normalized_duplicates
    print("事件/原始行号隔离、清单一致性、三源输入跨集合精确排重通过", flush=True)

    rebuilt, statistics, blocks, rebuilt_audit, rebuilt_config = build(
        root.parent / "数据", config["manifest_seed"], config["max_delta_s_strict"],
        config["saturation_threshold"], config["clip"])
    assert samples.keys() == rebuilt.keys()
    for name in samples:
        assert np.array_equal(samples[name], rebuilt[name]), name
    assert rebuilt_config == config and rebuilt_audit == saved_audit
    with (prepared / "blocks.csv").open(encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [{key: str(value) for key, value in row.items()} for row in blocks]
    for protocol, expected in statistics.items():
        with np.load(prepared / f"normalization_{protocol}.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(expected)
            for key, value in expected.items():
                assert np.array_equal(value, saved[key]), (protocol, key)
    del rebuilt, statistics

    # 真实Train样本验证label/sample_index及任意额外元数据不影响旧均值网络输出。
    torch.set_num_threads(2)
    dataset = ConverterDataset(prepared, "train", "raw", samples)
    assert set(dataset.inputs) == {"current", "dc", "rms"}
    batch = {key: value[:8] for key, value in dataset.inputs.items()}
    batch.update(label=dataset.labels[:8], sample_index=dataset.sample_indices[:8])
    model = FaultModel("mean").eval()
    changed = {**batch, "label": (batch["label"] + 1) % 8,
               "sample_index": batch["sample_index"] + 99999,
               "fine_label": torch.arange(8), "timestamp": torch.arange(8), "block": torch.arange(8)}
    with torch.inference_mode():
        assert torch.equal(model(batch)[0], model(changed)[0])
    assert all(fingerprint(prepared / name) == digest for name, digest in before.items())
    report = {"status": "passed", "direct_label_or_record_leakage_found": False,
              "full_raw_rebuild_exact_match": True, "normalization_and_class_weights_train_only_exact": True,
              "manifest_and_blocks_match_arrays": True, "cross_split_blocks_and_source_rows": 0,
              "raw_joint_vectors": raw_duplicates, "normalized_joint_vectors": normalized_duplicates,
              "feature_inputs": ["current", "dc", "rms"], "metadata_perturbation_changes_model_output": False,
              "prepared_files_unchanged_sha256": before,
              "source_sha256": config["source_sha256"],
              "split_counts": {name: int((samples["split"] == index).sum()) for index, name in enumerate(SPLITS)},
              "label_consistency_filter_after_time_matching": {
                  "time_matched": saved_audit["time_matched"],
                  "label_mismatch_excluded": saved_audit["label_mismatches_after_time_match"],
                  "description": "最近邻仅按时间匹配，随后剔除2组标签不一致的三源记录；标签不进入特征"},
              "limitations": ["随机事件划分共享注入批次；只能支持当前数据内评估，不能证明跨批次泛化",
                              "历史V1已观察Test；本轮训练只用Train/Val，不将Test称为完全未观察的终评集"],
              "test_model_prediction_computed": False}
    write_json(output / "leakage_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    audit(Path.cwd(), Path("artifacts/checks/current_split_20261009"))
