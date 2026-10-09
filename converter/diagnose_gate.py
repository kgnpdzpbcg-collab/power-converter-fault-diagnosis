"""在验证集上描述分支表示幅值与门控行为，不用测试集调整模型。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import ConverterDataset, load_samples
from .models import FaultModel
from .models_v2 import FaultModelV2
from .prepare import write_json


def diagnose(prepared_dir, run_dir, samples=None):
    """统计最优 checkpoint 的特征 L2 范数及权重。

    这是训练后的相关性观察：特征幅值与权重会互相影响，不能据此确认
    分支受抑制的因果来源。恒定权重或恒定范数不计算 Pearson 相关。
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "experiment.json").read_text(encoding="utf-8"))
    torch.set_num_threads(config["training"]["threads"])
    samples = load_samples(prepared_dir) if samples is None else samples
    dataset = ConverterDataset(prepared_dir, "val", config["protocol"], samples)
    # 仅允许项目内已知模型类，不执行元数据中的任意模块或表达式。
    constructors = {"converter.models.FaultModel": FaultModel, "converter.models_v2.FaultModelV2": FaultModelV2}
    constructor = constructors[config.get("model_class", "converter.models.FaultModel")]
    model = constructor(config["fusion"], tuple(config["branches"]), **config.get("model_options", {}))
    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    features, weights = {name: [] for name in model.branches}, []
    with torch.inference_mode():
        for start in range(0, len(dataset), 256):
            batch = {name: value[start:start + 256] for name, value in dataset.inputs.items()}
            for name in model.branches:
                features[name].append(model.encoders[name](batch[name]).norm(dim=1).numpy())
            _, weight = model(batch)
            if weight is not None:
                weights.append(weight.numpy())
    feature_norms = {name: np.concatenate(values).astype(np.float64) for name, values in features.items()}
    weight_array = np.concatenate(weights).astype(np.float64) if weights else None
    branches = {}
    for index, name in enumerate(model.branches):
        norm = feature_norms[name]
        record = {"feature_l2_mean": float(norm.mean()), "feature_l2_std": float(norm.std()),
                  "feature_l2_q05_q50_q95": np.quantile(norm, [0.05, 0.5, 0.95]).tolist()}
        if weight_array is not None:
            branch_weight = weight_array[:, index]
            correlation = float(np.corrcoef(norm, branch_weight)[0, 1]) if norm.std() > 1e-10 and branch_weight.std() > 1e-10 else None
            record.update(weight_mean=float(branch_weight.mean()), weight_std=float(branch_weight.std()),
                          fraction_weight_below_0_01=float((branch_weight < 0.01).mean()),
                          feature_norm_weight_pearson=correlation)
        branches[name] = record
    result = {"dataset": "validation", "samples": len(dataset), "protocol": config["protocol"],
              "fusion": config["fusion"], "branches": branches,
              "interpretation": "训练后描述统计，不证明因果；不据此改变本轮实验配置"}
    write_json(run_dir / "gate_diagnostic_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("artifacts/prepared/v0"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.prepared_dir, args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
