"""训练独占统计的轻量分类探针、通用指标和20点波形表示。"""

import numpy as np


def balanced_weights(labels, blocks):
    """每类总权重相同，每类内的(block,label)单元也相同，避免长事件主导。

    返回归一化总和为1的权重，仅根据传入训练记录计算。
    正常恢复段与故障段即使同block，也按各自标签作为不同评估单元。
    """
    labels, blocks = np.asarray(labels), np.asarray(blocks)
    units, inverse, counts = np.unique(np.column_stack((labels, blocks)), axis=0,
                                        return_inverse=True, return_counts=True)
    classes, unit_class_counts = np.unique(units[:, 0], return_counts=True)
    counts_by_class = dict(zip(classes, unit_class_counts))
    weights = 1.0 / (counts[inverse] * np.array([counts_by_class[y] for y in labels]))
    return weights / weights.sum()


def fit_probe(x, labels, blocks, classes, method="ridge", regularization=0.01):
    """只读训练输入，拟合固定正则线性分类器或事件原型最近邻。

    训练标准化使用类别/事件平衡权重；不存在验证集输入或调参。
    最近邻原型是每个训练(block,label)的均值，标签只来自训练集。
    """
    x = np.asarray(x, dtype=np.float64)
    labels, classes = np.asarray(labels), np.asarray(classes)
    if not np.isfinite(x).all() or not set(classes).issubset(set(labels)):
        raise ValueError("探针训练特征非有限，或指定类别缺少训练记录")
    weights = balanced_weights(labels, blocks)
    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2))
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (x - mean) / scale
    model = {"mean": mean, "scale": scale, "classes": classes,
             "method": np.array(method), "regularization": np.array(regularization)}
    if method == "ridge":
        design = np.column_stack((normalized, np.ones(len(x))))
        target = (labels[:, None] == classes[None, :]).astype(np.float64)
        penalty = np.eye(design.shape[1]) * regularization
        penalty[-1, -1] = 0  # 截距不正则化，标准化后仍允许类别基准项。
        model["coefficients"] = np.linalg.solve(design.T @ (design * weights[:, None]) + penalty,
                                                design.T @ (target * weights[:, None]))
    elif method == "prototype":
        units, inverse, counts = np.unique(np.column_stack((labels, blocks)), axis=0,
                                            return_inverse=True, return_counts=True)
        centers = np.zeros((len(units), normalized.shape[1]))
        np.add.at(centers, inverse, normalized)
        model.update(centers=centers / counts[:, None], center_labels=units[:, 0])
    else:
        raise ValueError(f"未知探针 {method}")
    return model


def predict_scores(model, x):
    """返回每个指定类别的得分，不读取验证标签；分块避免距离矩阵占用过大。

    岭得分并非校准概率。近邻得分为该类别最近训练事件原型的负平方距离。
    """
    normalized = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    if str(model["method"]) == "ridge":
        return np.column_stack((normalized, np.ones(len(x)))) @ model["coefficients"]
    centers = model["centers"]
    center_norm = (centers ** 2).sum(1)
    result = []
    for start in range(0, len(x), 256):
        batch = normalized[start:start + 256]
        distances = np.maximum((batch ** 2).sum(1)[:, None] + center_norm - 2 * batch @ centers.T, 0)
        result.append(np.column_stack([-distances[:, model["center_labels"] == label].min(1)
                                       for label in model["classes"]]))
    return np.concatenate(result)


def confusion_matrix(labels, predictions, classes, weights=None):
    """支持任意固定类别集合和浮点事件权重；不删除模型没有预测的类别。"""
    classes = np.asarray(classes)
    rows, cols = np.searchsorted(classes, labels), np.searchsorted(classes, predictions)
    if np.any(rows >= len(classes)) or np.any(cols >= len(classes)):
        raise ValueError("预测或标签不在类别集合")
    if not np.array_equal(classes[rows], labels) or not np.array_equal(classes[cols], predictions):
        raise ValueError("类别集合不包含输入标签")
    return np.bincount(rows * len(classes) + cols, weights=weights,
                       minlength=len(classes)**2).reshape(len(classes), len(classes))


def metrics(confusion):
    """零分母计0；事件加权矩阵保持float64，不截断权重。"""
    cm = np.asarray(confusion, dtype=np.float64)
    support, predicted = cm.sum(1), cm.sum(0)
    correct = np.diag(cm)
    precision = np.divide(correct, predicted, out=np.zeros(len(cm)), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(len(cm)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(cm)), where=(precision + recall) > 0)
    return {"macro_f1": float(f1.mean()), "macro_recall": float(recall.mean()),
            "accuracy": float(correct.sum() / cm.sum()) if cm.sum() else 0.0,
            "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
            "per_class_precision": precision.tolist(), "support": support.tolist(),
            "confusion_matrix": cm.tolist()}


def score_predictions(labels, predictions, blocks, classes):
    """同时报告样本指标及每个(block,label)权重为1的事件平衡指标。"""
    _, inverse, counts = np.unique(np.column_stack((labels, blocks)), axis=0,
                                   return_inverse=True, return_counts=True)
    return {"sample": metrics(confusion_matrix(labels, predictions, classes)),
            "event_balanced": metrics(confusion_matrix(labels, predictions, classes, 1 / counts[inverse]))}


def waveform_features(x, window="rectangular"):
    """输入[N,C,20]原始波形，返回幅值统计、单位RMS时域形状及非DC谱形。

    FFT不零填充；单边功率对内部频点乘2，Nyquist点乘1。
    20点减均值；常值信号的形状和非DC频谱定义为0。
    所有运算逐样本，不拟合任何跨样本统计，幅值不变性仅针对形状表示。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 20 or not np.isfinite(x).all():
        raise ValueError("波形必须为有限的[N,C,20]")
    mean = x.mean(-1)
    centered = x - mean[..., None]
    std = np.sqrt(np.mean(centered ** 2, -1))
    time_shape = np.divide(centered, std[..., None], out=np.zeros_like(centered), where=std[..., None] > 0)
    if window == "hann":
        centered = centered * np.hanning(20)
    elif window != "rectangular":
        raise ValueError("只支持矩形或Hann窗")
    fft = np.fft.rfft(centered, axis=-1)
    power = np.abs(fft[..., 1:]) ** 2
    power[..., :-1] *= 2
    total = power.sum(-1, keepdims=True)
    spectral_shape = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    # 对数统计抑制极大值的数值量级，但不删除记录或改变标签。
    stats = np.stack((np.sign(mean) * np.log1p(np.abs(mean)), np.log1p(std),
                      np.log1p(np.ptp(x, axis=-1)), np.log1p(np.sqrt(np.mean(x ** 2, -1)))), -1)
    return {"stats": stats.reshape(len(x), -1), "time_shape": time_shape.reshape(len(x), -1),
            "spectral_shape": spectral_shape.reshape(len(x), -1), "power": power}


def paired_event_bootstrap(labels, pred_a, pred_b, blocks, classes, repeats=200):
    """以完整验证block配对重采样，区间只反映当前视图内事件变化。

    不把种子当独立样本，不根据区间选择配置，也不排除标签与时间批次混杂。
    """
    unique_blocks, inverse = np.unique(blocks, return_inverse=True)
    matrices = []
    for predictions in (pred_a, pred_b):
        per_block = np.zeros((len(unique_blocks), len(classes), len(classes)), dtype=np.int32)
        row, col = np.searchsorted(classes, labels), np.searchsorted(classes, predictions)
        np.add.at(per_block, (inverse, row, col), 1)
        matrices.append(per_block)
    rng = np.random.default_rng(20261009)
    differences = []
    for _ in range(repeats):
        selected = rng.integers(0, len(unique_blocks), len(unique_blocks))
        differences.append(metrics(matrices[1][selected].sum(0))["macro_f1"]
                           - metrics(matrices[0][selected].sum(0))["macro_f1"])
    return {"repeats": repeats, "mean_difference": float(np.mean(differences)),
            "percentile_95_interval": np.quantile(differences, [0.025, 0.975]).tolist()}
