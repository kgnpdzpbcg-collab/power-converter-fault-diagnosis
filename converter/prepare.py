"""从三个原始 JSON 构造联合样本，先重建事件，再固定划分和训练集统计量。"""

import argparse
from array import array
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


BRANCHES = {"current": (2, (3, 20)), "dc": (3, (2, 20)), "rms": (10, (3,))}
CLASS_NAMES = ["正常", "交流短路", "弱电网", "相断开", "电网电气异常", "谐波", "直流侧故障", "IGBT闭锁"]
SPLITS = ("train", "val", "test")
PROTOCOLS = ("raw", "exclude_saturation")
# 在建模前固定映射，禁止依据训练性能合并类别。
FAMILY_MAP = np.array([0] + [1] * 11 + [2] + [3] * 7 + [4] * 12 + [5] * 7 + [6] * 7 + [7] * 2, dtype=np.int64)


def write_json(path, value):
    """以 UTF-8 保存审计/配置；禁止 NaN 被写成非标准 JSON。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def iter_records(path, chunk_size=1 << 20):
    """分块解析顶层 JSON 数组；逐条返回字典，截断或多余内容直接报错。

    大文件不整体转换成 Python 对象列表，降低内存占用。chunk_size 仅用于
    I/O，记录可以跨块；逗号、结束括号与尾部内容也必须符合 JSON 语法。
    """
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer, position, ended = "", 0, False

        def refill():
            nonlocal buffer, position, ended
            piece = handle.read(chunk_size)
            buffer = buffer[position:] + piece
            position = 0
            ended = not piece

        def peek():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if ended:
                    return ""
                refill()

        if peek() != "[":
            raise ValueError(f"{path}: 顶层必须是 JSON 数组")
        position += 1
        first = True
        while peek() != "]":
            if not first:
                if peek() != ",":
                    raise ValueError(f"{path}: 记录间缺少逗号或文件被截断")
                position += 1
                if peek() == "]":
                    raise ValueError(f"{path}: 不允许尾部逗号")
            peek()
            while True:
                try:
                    record, position = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as error:
                    if ended:
                        raise ValueError(f"{path}: 无效或截断的 JSON") from error
                    refill()
            if not isinstance(record, dict):
                raise ValueError(f"{path}: 数组元素必须是记录字典")
            yield record
            first = False
        position += 1
        if peek():
            raise ValueError(f"{path}: 数组结束后存在多余内容")


@dataclass
class Source:
    """保留数值和原文件行号；row 指顶层数组中的零起始记录位置。"""
    x: np.ndarray
    time: np.ndarray
    label: np.ndarray
    row: np.ndarray
    max_abs: np.ndarray
    context: tuple
    audit: dict


def read_source(path, measurement_id, shape, threshold):
    """验证原始字段、维度、数值和重复记录，并按采集时间排序。

    当前协议只覆盖一个设备、一次实验；如果数据变为多个上下文，明确报错，
    防止不同设备被全局时间匹配。store_timestamp 不参与对齐或模型输入。
    """
    values, times, labels, maxima = array("f"), array("d"), array("b"), array("d")
    contexts, record_ids = set(), set()
    dimension = int(np.prod(shape))
    for row, record in enumerate(iter_records(path)):
        if record["id_mea"] != measurement_id:
            raise ValueError(f"{path}, 第 {row} 条: id_mea 错误")
        label = record["id_error"]
        vector = record["mea_value"]
        timestamp = float(record["timestamp"])
        if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < 48:
            raise ValueError(f"{path}, 第 {row} 条: 标签超出整数 0..47")
        if len(vector) != dimension or not np.isfinite(vector).all() or not np.isfinite(timestamp):
            raise ValueError(f"{path}, 第 {row} 条: 维度错误或 NaN/Inf")
        record_id = record["_id"]["$oid"]
        if record_id in record_ids:
            raise ValueError(f"{path}: 重复记录 _id={record_id}")
        record_ids.add(record_id)
        contexts.add((record["device"], record["experiment_id"]))
        values.extend(vector)
        times.append(timestamp)
        labels.append(label)
        maxima.append(max(abs(value) for value in vector))
    if not times or len(contexts) != 1:
        raise ValueError(f"{path}: 空文件或多个设备/实验，当前协议不适用")
    time = np.asarray(times)
    order = np.argsort(time, kind="stable")
    time = time[order]
    if np.any(np.diff(time) == 0):
        raise ValueError(f"{path}: 重复采集时间戳，最近邻匹配不唯一")
    x = np.asarray(values).reshape((-1, *shape))[order]
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: 转为 float32 后出现 Inf")
    label = np.asarray(labels, dtype=np.int64)[order]
    max_abs = np.asarray(maxima)[order]
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    audit = {
        "path": str(Path(path).resolve()), "sha256": digest, "records": len(time),
        "shape": list(shape), "context": list(next(iter(contexts))),
        "reordered": bool(np.any(order != np.arange(len(time)))),
        "nan_inf": 0, "duplicate_ids": 0, "duplicate_timestamps": 0,
        "fine_class_counts": np.bincount(label, minlength=48).tolist(),
        "saturation_candidates": int((max_abs >= threshold).sum()),
        "timestamp_gap_quantiles_s": np.quantile(np.diff(time), [0.5, 0.95, 0.99, 1.0]).tolist(),
        "max_abs_quantiles": np.quantile(max_abs, [0.5, 0.95, 0.99, 1.0]).tolist(),
    }
    print(f"读取 {Path(path).name}: {len(time):,} 条，字段与重复检查通过", flush=True)
    return Source(x, time, label, order.astype(np.int64), max_abs, next(iter(contexts)), audit)


def nearest(anchor, source):
    """只用时间选择最近邻；距离相等时取较早记录，不使用标签筛选邻居。"""
    right = np.clip(np.searchsorted(source, anchor), 0, len(source) - 1)
    left = np.clip(right - 1, 0, len(source) - 1)
    return np.where(np.abs(source[left] - anchor) <= np.abs(source[right] - anchor), left, right)


def episode_blocks(fine_labels):
    """在未过滤、未合并的 RMS 标签序列上建立“故障＋后续正常段” block。

    连续不同的非零细标签属于不同事件，即使映射到同一粗类；初始正常段
    并入第一个事件。不能在三源过滤后重建，否则丢失中间记录可能合并事件。
    """
    starts = np.flatnonzero((fine_labels != 0) & np.r_[True, fine_labels[1:] != fine_labels[:-1]])
    if not len(starts):
        raise ValueError("没有故障事件，无法建立当前分层协议")
    block = np.maximum(np.searchsorted(starts, np.arange(len(fine_labels)), side="right") - 1, 0)
    return block.astype(np.int64), starts


def split_blocks(block_fine_labels, seed=42):
    """按故障细标签分层，约 70/15/15；seed 只控制固定 manifest。

    每个细类至少需三个事件以保证覆盖；使用确定的四舍五入规则，剩余事件
    进入训练集。后续模型的训练 seed 不能改变这里的划分。
    """
    rng = np.random.default_rng(seed)
    split = np.full(len(block_fine_labels), -1, dtype=np.int64)
    for label in np.unique(block_fine_labels):
        ids = np.flatnonzero(block_fine_labels == label)
        if len(ids) < 3:
            raise ValueError(f"细标签 {label} 只有 {len(ids)} 个事件，无法覆盖三个集合")
        ids = rng.permutation(ids)
        held = max(1, int(np.floor(len(ids) * 0.15 + 0.5)))
        split[ids[:held]] = 1
        split[ids[held:2 * held]] = 2
        split[ids[2 * held:]] = 0
    return split


def protocol_mask(samples, protocol):
    """敏感性协议只过滤固定样本池，不重新划分事件。"""
    if protocol not in PROTOCOLS:
        raise ValueError(f"未知数据协议: {protocol}")
    return np.ones(len(samples["label"]), dtype=bool) if protocol == "raw" else ~samples["saturation_candidate"]


def fit_normalization(samples, protocol, clip=10.0):
    """仅用对应协议的训练样本，按物理通道拟合 median/IQR 和类别权重。

    波形统计跨样本和 20 个点；RMS 按三个特征分别计算。零 IQR 明确报错，
    不添加任意 epsilon 或使用验证/测试统计量补足。
    """
    train = (samples["split"] == 0) & protocol_mask(samples, protocol)
    counts = np.bincount(samples["label"][train], minlength=8)
    if np.any(counts == 0):
        raise ValueError(f"{protocol}: 训练集缺少类别")
    weights = len(samples["label"][train]) / (8 * counts.astype(np.float64))
    result = {"class_counts": counts, "class_weights": (weights / weights.mean()).astype(np.float32),
              "clip": np.array(clip, dtype=np.float32), "train_indices": np.flatnonzero(train)}
    for branch in BRANCHES:
        x = samples[branch][train].astype(np.float64)
        axes = (0, 2) if x.ndim == 3 else 0
        q25, median, q75 = np.quantile(x, [0.25, 0.5, 0.75], axis=axes)
        iqr = q75 - q25
        if np.any(iqr <= 0) or not np.isfinite(iqr).all():
            raise ValueError(f"{protocol}/{branch}: 零或无效 IQR，需明确设计尺度处理")
        result[f"{branch}_median"] = median.astype(np.float32)
        result[f"{branch}_iqr"] = iqr.astype(np.float32)
    return result


def assert_no_leakage(samples):
    """事件与任一测量的原始记录均不跨集合；所有集合必须覆盖八类。"""
    for split_id, name in enumerate(SPLITS):
        if len(np.unique(samples["label"][samples["split"] == split_id])) != 8:
            raise ValueError(f"{name} 缺少粗类别")
    for key in ("block", "row_current", "row_dc", "row_rms"):
        memberships = [set(samples[key][samples["split"] == index].tolist()) for index in range(3)]
        if any(memberships[a] & memberships[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"{key} 跨集合重复")


def build(raw_dir, seed=42, max_delta=0.2, threshold=1e8, clip=10.0):
    """构造内存中样本、事件清单、统计与审计；不写入或改动原始数据。"""
    sources = {name: read_source(Path(raw_dir) / f"id_mea_{mid}.json", mid, shape, threshold)
               for name, (mid, shape) in BRANCHES.items()}
    if len({source.context for source in sources.values()}) != 1:
        raise ValueError("三个来源的设备/实验不一致")
    anchor = sources["rms"]
    block, starts = episode_blocks(anchor.label)
    block_split = split_blocks(anchor.label[starts], seed)
    matched = {name: nearest(anchor.time, sources[name].time) for name in ("current", "dc")}
    delta = {name: sources[name].time[index] - anchor.time for name, index in matched.items()}
    time_ok = (np.abs(delta["current"]) < max_delta) & (np.abs(delta["dc"]) < max_delta)
    label_ok = (sources["current"].label[matched["current"]] == anchor.label) & (sources["dc"].label[matched["dc"]] == anchor.label)
    keep = time_ok & label_ok
    indices = {**{name: index[keep] for name, index in matched.items()}, "rms": np.flatnonzero(keep)}
    for name, index in indices.items():
        if len(np.unique(index)) != len(index):
            raise ValueError(f"{name}: 一个原始记录被多个联合样本复用，需先明确匹配规则")
    saturation = np.zeros(int(keep.sum()), dtype=bool)
    samples = {"label": FAMILY_MAP[anchor.label[keep]], "fine_label": anchor.label[keep],
               "timestamp": anchor.time[keep], "block": block[keep], "split": block_split[block[keep]]}
    for name, index in indices.items():
        source = sources[name]
        samples[name] = source.x[index]
        samples[f"row_{name}"] = source.row[index]
        samples[f"timestamp_{name}"] = source.time[index]
        saturation |= source.max_abs[index] >= threshold
    samples["saturation_candidate"] = saturation
    for name in matched:
        samples[f"delta_{name}_s"] = delta[name][keep]
    assert_no_leakage(samples)
    stats = {protocol: fit_normalization(samples, protocol, clip) for protocol in PROTOCOLS}
    block_starts = np.r_[0, starts[1:]]
    block_ends = np.r_[starts[1:], len(anchor.time)]
    blocks = [{"block": index, "fine_label": int(anchor.label[starts[index]]),
               "split": SPLITS[int(block_split[index])], "anchor_start_row": int(anchor.row[start]),
               "anchor_end_row": int(anchor.row[end - 1]), "fault_start_row": int(anchor.row[starts[index]]),
               "start_timestamp": float(anchor.time[start]), "end_timestamp": float(anchor.time[end - 1])}
              for index, (start, end) in enumerate(zip(block_starts, block_ends))]
    audit = {"sources": {name: source.audit for name, source in sources.items()},
             "time_matched": int(time_ok.sum()), "label_mismatches_after_time_match": int((time_ok & ~label_ok).sum()),
             "joint_samples": int(keep.sum()), "episode_blocks": len(starts),
             "coarse_class_counts": np.bincount(samples["label"], minlength=8).tolist(),
             "saturation_candidates": int(saturation.sum()),
             "saturation_by_class": np.bincount(samples["label"][saturation], minlength=8).tolist(),
             "match_delta_abs_quantiles_ms": {name: (np.quantile(np.abs(delta[name][keep]), [0.5, 0.95, 0.99, 1.0]) * 1000).tolist() for name in matched},
             "protocols": {}}
    for protocol in PROTOCOLS:
        eligible = protocol_mask(samples, protocol)
        audit["protocols"][protocol] = {}
        for split_id, name in enumerate(SPLITS):
            mask = eligible & (samples["split"] == split_id)
            counts = np.bincount(samples["label"][mask], minlength=8)
            if np.any(counts == 0):
                raise ValueError(f"{protocol}/{name}: 缺少粗类，禁止训练")
            audit["protocols"][protocol][name] = {"samples": int(mask.sum()), "class_counts": counts.tolist(), "blocks": len(np.unique(samples["block"][mask]))}
    config = {"manifest_seed": seed, "max_delta_s_strict": max_delta, "saturation_threshold": threshold,
              "clip": clip, "family_map": FAMILY_MAP.tolist(), "class_names": CLASS_NAMES,
              "protocols": list(PROTOCOLS), "split_names": list(SPLITS),
              "normalization": "训练集按通道 median/IQR；波形跨样本和点；线性分位数；float32；裁剪",
              "source_sha256": {name: source.audit["sha256"] for name, source in sources.items()}}
    return samples, stats, blocks, audit, config


def save(output_dir, samples, stats, blocks, audit, config):
    """保存固定清单及数值数组；已有目录拒绝覆盖，避免悄悄改变实验划分。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "samples.npz", **samples)
    for protocol, parameters in stats.items():
        np.savez(output_dir / f"normalization_{protocol}.npz", **parameters)
    fields = ["sample", "split", "block", "fine_label", "label", "timestamp", "row_current", "row_dc", "row_rms", "delta_current_s", "delta_dc_s", "saturation_candidate"]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(samples["label"])):
            record = {key: samples[key][index].item() for key in fields if key not in ("sample", "split")}
            writer.writerow({"sample": index, "split": SPLITS[samples["split"][index]], **record})
    with (output_dir / "blocks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "config.json", config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("../数据"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/prepared/v0"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("输出目录已存在；请使用新目录，禁止覆盖固定 manifest")
    result = build(args.raw_dir)
    save(args.output_dir, *result)
    print(json.dumps({"output": str(args.output_dir.resolve()), "samples": result[3]["joint_samples"], "blocks": result[3]["episode_blocks"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
