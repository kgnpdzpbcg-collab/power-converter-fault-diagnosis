# 电力变换器故障诊断实验

本项目使用 uv 管理 Python 3.12 环境。原始数据位于上级目录的 `数据/`，不复制或修改原始文件。

## 环境

在本项目目录打开 PowerShell。当前机器的 uv 路径为：

```powershell
$uvPath = 'D:\APP\miniconda3\miniconda\Scripts\uv.exe'
& $uvPath sync --locked
& $uvPath run python -c "import sys, numpy, torch; print(sys.executable); print(numpy.__version__); print(torch.__version__)"
```

也可以激活环境后运行 Python：

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

`pyproject.toml` 定义依赖，`uv.lock` 固定已解析的版本，`.venv/` 保存本项目的独立环境。
PyTorch 使用官方 CPU 软件源，供本机训练、数据加载与验证使用。后续若改为 GPU 训练，需要按机器的 CUDA 环境调整软件源并重新锁定依赖。

## 执行顺序

以下命令都在本项目目录执行，原始数据路径为 `../数据`。输出目录已存在时，准备和训练命令会拒绝覆盖，防止悄悄改变清单或结果。

```powershell
# 1. 构造联合样本、重建事件并固定划分；不会修改原始文件。
& $uvPath run python -m converter.prepare

# 2. 检查清单、训练集统计量、DataLoader，并从原始文件完整重建复核。
& $uvPath run python -m converter.verify --raw-dir '..\数据'

# 3. 核验关键边界逻辑。
& $uvPath run python -m unittest discover -s tests -v

# 4. 单独运行正式 V0，或使用下一条命令运行固定完整实验矩阵。
& $uvPath run python -m converter.train --fusion v0 --output-dir artifacts/experiments/v0_standalone

# 完整矩阵内已包含 V0，无需为其重复训练单独的 V0。
& $uvPath run python -m converter.experiments

# 5. 正式实验结束后，独立核验指标、最优轮次、checkpoint 和指纹。
& $uvPath run python -m converter.validate_results
```

短流程验证可指定 `--max-epochs 2 --validation-only`，结果不会混入正式实验汇总；该模式不评估测试指标。
详细协议、实验矩阵和解释边界见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

## 首轮执行结果

已生成 58,153 组联合样本，完成 48 次正式训练和全部结果核验。详见 [FINDINGS.md](FINDINGS.md)。
多测量融合有收益，初版 MA-MoE 尚未证明优于简单融合；原始极大值原因、门控分支抑制和正常/直流混淆仍待排查。

## 第二轮诊断与受控改进

已完成分支归一化、等权门控初始化的 6 个变体 × 3 种子比较，共 18 次训练，全部核验通过。
本轮只使用训练、验证集；等权平均的验证 Macro-F1 为 88.47%，动态对照为 86.12%，改动未带来平均收益。
正常/直流区分仍存在明显种子差异，下一步优先核对恢复段标签及分类损失。详见 [V2_FINDINGS.md](V2_FINDINGS.md) 和 [V2_PLAN.md](V2_PLAN.md)。

```powershell
# 按预先固定的计划运行第二轮；已完成的实验不会重复训练。
& $uvPath run python -m converter.experiments_v2

# 只重算验证指标、恢复验证预测，并对照首轮等权基线。
& $uvPath run python -m converter.validate_v2
```

## 文件与产物

第三轮方案见 [V3_PLAN.md](V3_PLAN.md)：三个全局特征 token 的单层四头自注意力，
再与原始三源特征跳跃拼接。B0–B3 各三个种子，共 12 次训练，只使用 Train/Val。
普通 MLP 对照与注意力主模型参数量相差 0.183%。

```powershell
& $uvPath run python -m unittest discover -s tests -v
& $uvPath run python scripts/check_v3.py
& $uvPath run python -m converter.experiments_v3
& $uvPath run python -m converter.validate_v3
```

短训练脚本拒绝覆盖已有目录。正式实验源码冻结后，不在同一轮中混入不同代码。
V3 原始模型、逐样本预测和数据只在本地；GitHub 归档源代码、方案和汇总核验报告。
远端保留早期 [V0 方案](docs/V0_EXPERIMENT_PLAN.md) 作为历史设计，不代表已运行的全部实验。

V3 已完成全部 12 次训练和独立核验。B0/B1/B2/B3 验证 Macro-F1 分别为
87.73% / 85.57% / 85.72% / 84.47%；主模型 B3 未优于对照。
完整结果与边界见 [V3_FINDINGS.md](V3_FINDINGS.md)，汇总核验见 `reports/v3/`。
版本对应关系见 [docs/VERSION_ARCHIVE.md](docs/VERSION_ARCHIVE.md)。

- `converter/prepare.py`：流式读取、最近时间匹配、八类映射、事件分层与训练集归一化。
- `converter/data.py`：共享固定样本池的 Dataset/DataLoader。
- `converter/models.py`：V0、匹配维度的拼接、等权平均、全局固定权重、动态门控。
- `converter/train.py`：加权交叉熵、验证集选择、早停、checkpoint 和一次最终测试。
- `converter/experiments.py`：按固定顺序训练并生成汇总。
- `converter/verify.py` 与 `tests/`：真实数据复核及关键逻辑测试。
- `converter/validate_results.py`：正式实验的清单、指纹、选模和预测指标核验。
- `converter/diagnose_gate.py`：验证集的分支表示幅值及权重诊断。
- `converter/diagnose_errors.py`：恢复段、直流细类、裁剪及类别权重的训练/验证诊断。
- `converter/models_v2.py` 与 `converter/experiments_v2.py`：第二轮配对初值的受控模型与固定矩阵。
- `converter/training_trace.py` 与 `converter/validate_v2.py`：训练权重/梯度观测及验证集独立核验。
- `artifacts/prepared/v0/`：样本 NPZ、固定 manifest、事件清单、两套归一化、数据审计和验证报告。
- `artifacts/checks/`：不评估测试集的流程及确定性验证。
- `artifacts/experiments/v1/`：正式实验，每次独立保存配置、日志、最优权重、验证/测试预测和结果。
- `artifacts/experiments/v2_validation/`：第二轮验证实验、源码快照、训练观测和核验报告。
- `artifacts/diagnostics/v2/`：恢复段和类别权重诊断，不包含测试预测或测试指标。

汇总文件为 `REPORT.md`、`summary.csv`、`per_class.csv` 和 `summary.json`。
读取 `progress.json` 可查看已完成实验数；只有 `result.json` 存在且 status=complete 的实验才计入汇总。
运行状态及最终指标以产物为准，不能用短流程验证分数代替正式测试结论。

## 细标签与频域可行性检查

已完成48细标签和20点短波形频域的Train/Val诊断，共132次固定简单分类器拟合。
本轮没有训练新神经网络。24项测试、132个训练参数重拟合/预测重放及24组事件bootstrap核验通过。
部分细标签可分，但谐波细类持续混淆，标签与时间批次高度绑定。
粗类频谱有判别信息，加入强时域分类器后的收益仍不稳定。
详细分析、边界与下一轮建议见 [FEASIBILITY_FINDINGS.md](FEASIBILITY_FINDINGS.md)，
冻结协议见 [FEASIBILITY_CHECK_PLAN.md](FEASIBILITY_CHECK_PLAN.md)，可存档表格在 `reports/feasibility/`。

```powershell
# 核验已保存的诊断，不使用Test。
.venv\Scripts\python.exe -m diagnostics.validate_routes
```
