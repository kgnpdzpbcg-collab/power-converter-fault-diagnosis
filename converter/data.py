"""共用固定样本池的 PyTorch 数据视图：训练/验证/测试与两种数值协议。"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .prepare import BRANCHES, SPLITS, protocol_mask


def load_samples(prepared_dir):
    """只读取数值 NPZ，不允许 pickle；三个 Dataset 可共享原始内存。"""
    with np.load(Path(prepared_dir) / "samples.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def normalize(x, median, iqr, clip):
    """广播通道统计并裁剪，返回独立 float32 数组，原始数组不修改。"""
    shape = (1, -1, 1) if x.ndim == 3 else (1, -1)
    return np.clip((x - median.reshape(shape)) / iqr.reshape(shape), -clip, clip).astype(np.float32)


class ConverterDataset(Dataset):
    """通过 manifest 索引取数据；元数据仅用于审计，不作为模型特征。

    每条返回 current[3,20]、dc[2,20]、rms[3]、整数 label、联合样本编号。
    所有分支始终使用共同的联合样本池，单源模型也不能扩大样本数量。
    """
    def __init__(self, prepared_dir, split, protocol="raw", samples=None):
        if split not in SPLITS:
            raise ValueError(f"未知集合: {split}")
        samples = load_samples(prepared_dir) if samples is None else samples
        mask = (samples["split"] == SPLITS.index(split)) & protocol_mask(samples, protocol)
        self.indices = np.flatnonzero(mask)
        with np.load(Path(prepared_dir) / f"normalization_{protocol}.npz", allow_pickle=False) as stats:
            self.inputs = {name: torch.from_numpy(normalize(samples[name][mask], stats[f"{name}_median"], stats[f"{name}_iqr"], float(stats["clip"]))) for name in BRANCHES}
            self.class_weights = torch.from_numpy(stats["class_weights"].copy())
        self.labels = torch.from_numpy(samples["label"][mask].astype(np.int64))
        self.sample_indices = torch.from_numpy(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {**{name: values[index] for name, values in self.inputs.items()},
                "label": self.labels[index], "sample_index": self.sample_indices[index]}


def make_loaders(prepared_dir, protocol="raw", seed=42, batch_size=256, samples=None, splits=SPLITS):
    """训练打乱、验证/测试顺序固定；Windows 下使用单进程加载避免派生开销。

    每次调用创建独立 Generator，seed 改变训练顺序，不能改变样本归属。
    最后一批不丢弃，保证训练损失、评估指标覆盖实际协议样本数。
    splits 指定需要的集合；验证阶段的改进实验只创建 train/val 数据视图。
    """
    if not splits or len(set(splits)) != len(splits) or any(split not in SPLITS for split in splits):
        raise ValueError("集合名称必须为不重复的 train/val/test")
    samples = load_samples(prepared_dir) if samples is None else samples
    datasets = {split: ConverterDataset(prepared_dir, split, protocol, samples) for split in splits}
    generator = torch.Generator().manual_seed(seed)
    return {split: DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=0,
                              generator=generator if split == "train" else None)
            for split, dataset in datasets.items()}
