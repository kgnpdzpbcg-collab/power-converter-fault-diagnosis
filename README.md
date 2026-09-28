# Power Converter Fault Diagnosis

基于 **Power Converter Fault Diagnosis Dataset** 的轻量级多源电气信号融合故障诊断研究。

## 项目定位

目标不是设计复杂网络，而是完成一篇结构完整、实验可信、可复现的电子设备/功率电子变换器故障诊断会议论文。

V0 主线固定为：

> **三种公开电气测量同步对齐 → 8 类粗粒度故障诊断 → episode-level 数据划分 → 轻量多分支神经网络融合 → 与单源/传统基线比较。**

## 数据源

- Dataset: *Power Converter Fault Diagnosis Dataset*
- Source: Zenodo record `20484338`
- DOI: `10.5281/zenodo.20484338`
- Related paper: García-Campos et al., *Hybrid Fault-Space Restructuring for Machine Learning-Based Fault Diagnosis in Power Electronic Converters*, Electronics, 2026.

公开压缩包中包含：

- `id_mea_10.json`: three-phase RMS voltages，3 维
- `id_mea_3.json`: instantaneous DC voltage and current，40 维，可重构为 `2 × 20`
- `id_mea_2.json`: instantaneous three-phase currents，60 维，可重构为 `3 × 20`

本项目不提交原始数据文件。请从 Zenodo 官方数据页下载。

## V0 文档

详细实验规范见：

- [`docs/V0_EXPERIMENT_PLAN.md`](docs/V0_EXPERIMENT_PLAN.md)

## 当前状态

- [x] 数据集与配套论文核验
- [x] 三种公开 measurement 结构核验
- [x] V0 任务定义固定
- [x] V0 标签映射固定
- [x] V0 数据对齐规则固定
- [x] V0 数据划分规则固定
- [x] V0 网络结构固定
- [ ] 数据预处理脚本
- [ ] episode manifest
- [ ] baseline
- [ ] V0 训练
- [ ] 消融实验
- [ ] 论文撰写

## 原则

1. 主指标使用 **Macro-F1**，不以 Accuracy 作为主要结论。
2. 主实验禁止 sample-level 随机拆分造成同一故障事件泄漏。
3. 所有归一化参数只允许由训练集估计。
4. 不为了“创新”堆叠 Transformer / Attention / BiLSTM 等复杂模块。
5. V0 先验证“多源信息是否真的有增益”，再决定是否进入 V1。
