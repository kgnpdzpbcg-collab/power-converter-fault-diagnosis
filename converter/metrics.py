"""固定八类混淆矩阵与宏平均指标，无外部 sklearn 依赖。"""

import numpy as np


def classification_metrics(confusion):
    """矩阵行是真实类别、列是预测类别；零分母按 0 处理，固定平均八类。

    数据门已要求各集合有八类样本；Balanced Accuracy 等于宏平均召回率。
    这里的零分母对应模型未预测某类，不可删除该类来抬高 Macro-F1。
    """
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros(8), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(8), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=(precision + recall) != 0)
    return {"macro_f1": float(f1.mean()), "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()), "balanced_accuracy": float(recall.mean()),
            "accuracy": float(true_positive.sum() / confusion.sum()), "per_class_f1": f1.tolist(),
            "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
            "support": support.tolist(), "confusion_matrix": confusion.tolist()}
