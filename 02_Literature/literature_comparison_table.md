---
date: 2026-05-15
author: LitAgent
output_to: "02_Literature/literature_comparison_table.md"
status: final
---

# 核心文献对比表

## A. 氟中毒 DL 诊断 — 直接竞品

| # | Paper | Task | Method | Dataset | SOTA | Key Gap | 对我们的价值 |
|---|-------|------|--------|---------|------|---------|------------|
| 17 | Xu 2025 MLTrMR (JCVI) | DF 4-grade | Masked Latent Transformer + random masking | DFID 200 (50/class) | 80.19% Acc | 无 EDL, 无 ordinal loss, Mild-Moderate 边界最差 | **必须超越的 SOTA baseline** |
| 18 | Li 2026 LD2Net (JRTIP) | DF 4-grade | Depthwise separable conv + dual-axis attention | DFID 200 (50/class) | 80.00% Acc / 3.31M params | 轻量化方向, 无不确定性, 无 ordinal | **轻量级 SOTA, 不同方向可互补引用** |
| 16 | Xu 2025 Mwinc-Mamba (BSPC) | SF 4-grade | CNN+SSM dual-branch + MWCS | SFXRay 80 (N21/M34/Mo13/S12) | 66.67% Acc | 无 EDL, 仅 CE loss, 比放射科医生低 3.33% | **SF SOTA baseline, 放射科医生一致性差 → EDL 动机** |
| 01 | Liu 2021 SF Integrated (IEEE TII) | SF binary/multi | CNN ensemble (ResNet+DenseNet) | Private 128 (forearm only) | 82.8% binary | 私密数据, 仅前臂, 纯 CNN | 早期 baseline, 方法已被超越 |

## B. 氟中毒传统方法/临床

| # | Paper | Task | Method | Key Finding | 局限性 |
|---|-------|------|--------|-------------|--------|
| 09 | DF Quantum Fuzzy (2018) | DF segmentation | Fuzzy C-means + K-means + quantum clustering | 传统 ML 特征工程 | 无 DL, 私密小数据 |
| 02 | DF Web Detection (2017) | DF web system | Image processing + web framework | 构建了 web 诊断平台 | 工程导向, 无 model innovation |
| 03 | Differential Diagnosis DF (2019) | DF clinical | Clinical differential diagnosis protocol | DF vs 非氟斑牙鉴别 | 纯临床, 无计算 |
| 18 | Quadri 2016 SF Case (Front Oncol) | SF MRI | Case report | SF 可模拟多发性骨髓瘤 | 单病例, 无法泛化 |
| 04 | Fejerskov DF Nature Rev | DF mechanisms | Review | DF 病理机制权威总结 | 综述, 无计算 |

## C. EDL/不确定性方法论 — 迁移来源

| # | Paper | Venue | Method Core | Task Domain | SOTA | 可迁移组件 |
|---|-------|-------|-------------|-------------|------|-----------|
| 05 | MEDL (2023) | TMI | Mutual EDL: evidence exchange between views | Multi-view medical seg | 72.5% DSC | Evidence 互信息框架 |
| 06 | ORCU (2025) | MedIA | Ordered regression + calibration + unimodality | Ordinal classification (generic) | SOTA on age/severity | **最直接: ORCU loss 替换 CE** |
| 07 | EviVLM (2025) | AAAI | EDL + VLM for open-vocab seg | Referring segmentation | SOTA on RefCOCO | VLM+EDL 融合范式 |
| 08 | REDNet (2024) | Signal Process | Evidential discounting for noisy labels | Segmentation with label noise | — | 标注噪声建模 (SF 场景直接适用) |
| 11 | Unc Evidential Fusion (2023) | MedIA | Evidential fusion for semi-supervised seg | Semi-supervised medical seg | — | Evidence 融合用于少量标注 |
| 12 | DuEDL (2024) | PR | EDL from scribble annotations | Weakly-supervised seg | — | Weak label → EDL |

## D. 牙科/骨科 DL — 相关但非竞品

| # | Paper | Task | Relevance |
|---|-------|------|-----------|
| 14 | DL Dental Survey | Survey of DL in dentistry | DF 是牙科 DL 中最小众方向 — 创新空间大 |
| 15 | FewShot Osteopenia (Xie 2024) | Few-shot bone disease classification | Few-shot + 骨疾病 — 小样本方法论参考 |
| 13 | HyperDenseNet (2020) | Multimodal brain segmentation | 多模态融合架构参考, 但领域不相关 |
| 10 | SF ML Prediction (2024, SciRep) | ML clinical prediction model | 非影像 DL — 佐证 SF 自动化需求 |

## E. 流行病学

| # | Paper | Key Data | 论文用途 |
|---|-------|----------|----------|
| 19 | Gu 2024 Zunyi Survey | DF 检出率 16.4% (2023), 知晓率 23.3% | Introduction 背景 — 疾病负担 |
