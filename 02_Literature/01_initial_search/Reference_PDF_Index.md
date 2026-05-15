---
date: 2026-05-15
author: LitAgent
input_from: "02_Literature/01_initial_search/References/ (128 PDFs)"
output_to: "02_Literature/01_initial_search/Reference_PDF_Index.md"
status: draft
---

# References PDF 补充文献索引

本索引对 `References/` 文件夹中 128 篇 PDF 进行分类、标注相关度、指定精读优先级。与 `literature_list.md`（28 篇框架性文献）互补，共同构成项目文献底座。

---

## 统计概览

| 类别 | 数量 | 高相关 (≥⭐⭐⭐) | 精读优先级 |
|------|------|------------------|------------|
| A — 氟中毒直接相关 | 29 | 17 | **Tier 1: 6 (+2 NEW 2025)**, Tier 2: 2 |
| B — 证据深度学习 (EDL) | 10 | 10 | Tier 1: 4, Tier 2: 2 |
| C — DL 医学图像分割 | 15 | 8 | Tier 2: 2 |
| D — 多模态融合 | 5 | 4 | Tier 2: 1 |
| E — 牙科/口腔影像 DL | 12 | 6 | Tier 1: 1 |
| F — 骨病变/骨质疏松 DL | 10 | 5 | Tier 1: 1 |
| G — 大型教科书（参考） | 3 | — | 不精读 |
| H — 低相关（拟排除） | 18 | 0 | 排除 |
| U — 未分类（待确认） | 21 | — | 待确认 |
| **合计** | **134** | | **精读: 19 (15 original + 4 NEW), 排除: 18** |

---

## A. 氟中毒直接相关（25 篇）

### A1. 氟斑牙 DL/图像分析（6 篇）

| # | 文件名 | 大小 | 推断主题 | 相关度 | 精读 |
|---|--------|------|----------|--------|------|
| A1-1 | `Dental_Fluorosis_Analysis_A_Web-Based_Dental_Fluorosis_Severity_Detection.pdf` | 1.9MB | 基于 Web 的氟斑牙严重程度自动检测系统 | ⭐⭐⭐⭐⭐ | **Tier 1** |
| A1-2 | `Dental_Fluorosis_Segmentation_Using_Enhanced_Quantum-Inspired_Fuzzy_Clustering_Algorithm.pdf` | 969KB | 量子启发模糊聚类的氟斑牙病变区域分割 | ⭐⭐⭐⭐ | Tier 2 |
| A1-3 | `Detection_of_dental_Fluorosis_using_enhanced_K-means_and_Fuzzy_C_means.pdf` | 605KB | K-means + FCM 联合检测氟斑牙 | ⭐⭐⭐⭐ | — |
| A1-4 | `Differential_Diagnosis_of_Dental_Fluorosis.pdf` | 1.8MB | 氟斑牙鉴别诊断（vs MIH、龋齿等） | ⭐⭐⭐⭐⭐ | **Tier 1** |
| A1-5 | `The_Nature_and_Mechanisms_of_Dental_Fluorosis_in_Man.pdf` | 2MB | Fejerskov 氟斑牙病理机制经典文献 | ⭐⭐⭐⭐⭐ | **Tier 1** |
| A1-6 | `10.1177_0022034510384626.pdf` | 298KB | J Dental Research 氟斑牙研究 | ⭐⭐⭐ | — |
| A1-7 | `Xu 等 - 2025 - Masked latent transformer with random masking ratio to advance the diagnosis of dental fluorosis.pdf` | — | **NEW!** MLTrMR — 首个开源 DF 数据集 (DFID: 200 images) + ViT 掩码潜变换器, Acc=80.19%, QWK=81.28%. 贵州大学 Wu 组 | ⭐⭐⭐⭐⭐ | **Tier 1** |
| A1-D1 | `Ahmed 等 - 2024 - Dental Fluorosis Analysis A Web-Based Dental Fluorosis Severity Detection.pdf` | 1.9MB | ⚠️ 与 A1-1 重复（中文命名版本） | — | — |

### A2. 氟骨症 DL/诊断（10 篇）

| # | 文件名 | 大小 | 推断主题 | 相关度 | 精读 |
|---|--------|------|----------|--------|------|
| A2-1 | `Integrated_Learning_Approach_Based_on_Fused_Segmentation_Information_for_Skeletal_Fluorosis_Diagnosis_and_Severity_Grading.pdf` | 4.4MB | **唯一** SF DL 分级论文，融合分割信息集成学习 | ⭐⭐⭐⭐⭐ | **Tier 1** |
| A2-2 | `s41598-026-43429-4_reference.pdf` | 1.1MB | SF 严重程度预测（RF+SHAP，1309 例） | ⭐⭐⭐⭐ | Tier 2 |
| A2-3 | `neurology_of_endemic_skeletal_fluorosis.4.pdf` | 209KB | 地方性氟骨症神经病学表现 | ⭐⭐⭐ | — |
| A2-4 | `Treatment+and+Prevention+of+Skeletal+Fluorosis.pdf` | 644KB | 氟骨症治疗与预防综述 | ⭐⭐⭐ | — |
| A2-5 | `J Cellular Molecular Medi - 2019 - Wei - The pathogenesis of endemic fluorosis Research progress in the last 5 years.pdf` | 840KB | 地方性氟中毒发病机制 5 年进展 | ⭐⭐⭐⭐ | — |
| A2-6 | `Medical Journal of Australia - 1985 - Smith - Fluoride teeth and bone.pdf` | 602KB | 氟化物对牙齿和骨骼影响的经典综述 | ⭐⭐⭐ | — |
| A2-7 | `IJOrtho-53-324.pdf` | 2.7MB | 骨科视角氟骨症 | ⭐⭐ | — |
| A2-8 | `MJR-33-2-261.pdf` | 1MB | 放射学视角氟骨症 | ⭐⭐⭐ | — |
| A2-9 | `Xu 等 - 2025 - Convolutional state space model with multi-window cross-scan to advance the automated diagnosis of s.pdf` | — | **NEW!** Mwinc-Mamba — 首个开源 SF X-ray 数据集 (SFXRay: 80 images) + SSM+CNN 双分支, 二分类 83.33%, 四分级 66.67%. 贵州大学 Wu 组 | ⭐⭐⭐⭐⭐ | **Tier 1** |
| A2-10 | `Quadri 等 - 2016 - Multiple Myeloma-Like Spinal MRI Findings in Skeletal Fluorosis An Unusual Presentation of Fluoride.pdf` | — | **NEW** SF 脊柱 MRI 误诊为多发性骨髓瘤病例报告 (Frontiers in Oncology) — 非 DL, 用于 Intro/Discussion | ⭐⭐⭐ | — |
| A2-D1 | `Liu 等 - 2021 - Integrated Learning Approach Based on Fused Segmentation Information for Skeletal Fluorosis Diagnosi.pdf` | 4.4MB | ⚠️ 与 A2-1 重复（中文命名版本） | — | — |

### A3. 氟中毒流行病学/环境健康（12 篇）

用于 Introduction/Discussion 背景支撑，不直接贡献方法创新。

| # | 文件名 | 大小 | 相关度 |
|---|--------|------|--------|
| A3-1 | `Association+of+Dietary+Carotenoids+Intake+with+Skeletal+Fluorosis+in+the+Coal-burning+Fluorosis+Area+of+Guizhou+Province.pdf` | 3.7MB | ⭐⭐⭐ |
| A3-2 | `1-s2.0-S0048969721056795-main.pdf` | 2.4MB | ⭐⭐⭐ |
| A3-3 | `1-s2.0-S0147651325003549-main.pdf` (+ 1 副本) | 8.3MB | ⭐⭐⭐ |
| A3-4 | `1-s2.0-S0301479709002928-main.pdf` | 271KB | ⭐⭐ |
| A3-5 | `1-s2.0-S0301479724025507-main.pdf` | 7MB | ⭐⭐ |
| A3-6 | `s12011-020-02296-4.pdf` | 866KB | ⭐⭐⭐ |
| A3-7 | `s12011-022-03227-1.pdf` | 1.7MB | ⭐⭐⭐ |
| A3-8 | `s12011-024-04203-7.pdf` | 2.2MB | ⭐⭐⭐ |
| A3-9 | `s12940-025-01226-y.pdf` | 2.8MB | ⭐⭐⭐ |
| A3-10 | `s40572-023-00412-9.pdf` | 1.2MB | ⭐⭐ |
| A3-11 | `bmjopen-2012-001564.pdf` | 778KB | ⭐⭐ |
| A3-12 | `遵义市新民镇朝阳村2009与2023年地氟病情况调查分析-顾茂华.pdf` | — | **NEW** 遵义 2009 vs 2023 DF 检出率 (33.7%→16.4%) 流行病学调查 — Gu/Wu 2024, 同一课题组的田野调查 | ⭐⭐⭐ | — |

额外低相关度流行病学/环境文件：`s40572-020-00270-9.pdf`, `fpubh-10-849173.pdf`, `ijerph-20-03394-v2.pdf`, `s10653-021-01148-x.pdf`, `s10661-022-10888-x.pdf`, `s12517-020-06342-2.pdf`, `IJEM-21-190.pdf`, `1-s2.0-S0048969706008345-main.pdf`, `1-s2.0-S1382668915300235-main.pdf`, `1-s2.0-S1382668917302661-main.pdf`, `10.1177_1040638720962746.pdf`, `s41598-025-16860-2.pdf`, `s00204-024-03853-9.pdf` — 不单独列详情。

---

## B. 证据深度学习 EDL（10 篇）⭐⭐⭐⭐⭐ 创新方向关键

> **战略意义**：EDL 集群是 `literature_list.md` **完全缺失**的方向，可能成为项目核心创新点。EDL 量化预测不确定性，恰好解决氟中毒分级中医生间主观性差异大的痛点。

| # | 文件名 | 大小 | 推断主题 | 精读 |
|---|--------|------|----------|------|
| B-1 | `2505.12418-MEDL_Mutual_Evidential_Deep_Learning.pdf` | 3.9MB | 互证据深度学习（2025 最新，医学图像） | **Tier 1** |
| B-2 | `2410.15658-ORCU_Ordinal_Calibration_Unimodality.pdf` | 8.4MB | EDL 有序回归校准 — 与 Dean Index (0→1→2→3) 天然匹配 | **Tier 1** |
| B-3 | `EviVLM_When_Evidential_Learning_Meets_Vision_Language_Model_for_Medical_Image_Segmentation.pdf` | 3.7MB | EDL + 视觉语言模型多模态融合 | **Tier 1** |
| B-4 | `REDNet_Reliable_Evidential_Discounting_Network_for_Multi-Modality_Medical_Image_Segmentation.pdf` | 6.1MB | 可靠证据折扣网络 — 多模态不确定性融合 | **Tier 1** |
| B-5 | `2404.06177-Uncertainty_Evidential_Fusion_SemiSeg.pdf` | 1.8MB | 不确定性证据融合半监督分割 | Tier 2 |
| B-6 | `2405.14444-DuEDL_Scribble_Segmentation.pdf` | 1.7MB | 双重 EDL — 弱监督涂鸦分割 | Tier 2 |
| B-7 | `2410.18461-EDL_Biomedical_Segmentation.pdf` | 1.6MB | EDL 生物医学图像分割综述/方法 | — |
| B-8 | `2402.05685-Ordinal_Regression_for_Chest_Radiographs.pdf` | 224KB | 有序回归 + 证据学习（胸部 X 光） | — |
| B-9 | `2404.17126-EDL_Radiotherapy_Dose_Prediction.pdf` | 2.8MB | EDL 放疗剂量预测（方法参考） | — |
| B-10 | `2501.05656-EDL_Jet_Identification.pdf` | 2.1MB | EDL 粒子物理应用（仅方法学参考） | 排除 |

---

## C. DL 医学图像分割方法（15 篇）

| # | 文件名 | 大小 | 推断架构 | 精读 |
|---|--------|------|----------|------|
| C-1 | `CD-TransUNet_Enhanced_UNet_for_Medical_Image_Segmentation_via_Global_and_Local_Feature_Fusion.pdf` | 4.8MB | Cross-Dual TransUNet | Tier 2 |
| C-2 | `MaS-TransUNet_A_Multiattention_Swin_Transformer_U-Net_for_Medical_Image_Segmentation.pdf` | 7.2MB | Multi-Attention Swin Transformer U-Net | Tier 2 |
| C-3 | `MpVit-Unet_Multi-path_Vision_Transformer_Unet_for_Sellar_Region_Lesions_Segmentation.pdf` | 1.2MB | Multi-Path ViT U-Net | — |
| C-4 | `CPA-UNet_nnUNet-Based_Local_Pyramid_Aggregation_Network.pdf` | 831KB | Channel Pyramid Attention U-Net | — |
| C-5 | `DPCF-Net_2D_Medical_Image_Segmentation_Network_Based_on_Dual-Path_Cross-Fusion_Encoder.pdf` | 1.9MB | Dual-Path Cross-Fusion | — |
| C-6 | `LogTrans_Providing_Efficient_Local-Global_Fusion_with_Transformer_and_CNN_Parallel_Network_for_Biomedical_Image_Segmentation.pdf` | 2.3MB | Local-Global Transformer+CNN 并行 | — |
| C-7 | `Medical_Image_Segmentation_Using_Dual_Branch_Networks_with_Embedded_Attention_Mechanism.pdf` | 1.5MB | 双分支 + 嵌入注意力 | — |
| C-8 | `Asymmetric_Adaptive_Heterogeneous_Network_for_Multi-Modality_Medical_Image_Segmentation.pdf` | 6MB | 非对称自适应异构网络 | Tier 2 |
| C-9 | `Semantic_Segmentation_on_Panoramic_Dental_X-Ray_Images_Using_U-Net_Architectures.pdf` | 4.4MB | U-Net 牙科全景片分割 | — |
| C-10 | `1-s2.0-S0895611122000635-main.pdf` | 9.5MB | 医学图像分割方法 | — |
| C-11 | `2404.16371v1.pdf` | 1.4MB | arXiv 2024 分割 | — |
| C-12 | `2506.11691v1.pdf` | 2.1MB | arXiv 2025 分割 | — |
| C-13 | `2511.19046v1.pdf` | 5.0MB | arXiv 2025 分割 | — |
| C-14 | `Prior_Information_Guided_Coarse-to-fine_Dual-branch_Encoding_Network_for_Fovea_Localization_and_Optic_Disc_Cup_Segmentation.pdf` | 3.2MB | 先验引导粗到细（眼底图像） | 排除 |
| C-15 | `U-Shiftformer_Brain_Tumor_Segmentation_Using_A_Shifted_Attention_Mechanism.pdf` | 1.3MB | Shifted Attention U-Net（脑肿瘤） | 排除 |

---

## D. 多模态融合（5 篇）

| # | 文件名 | 大小 | 推断主题 | 精读 |
|---|--------|------|----------|------|
| D-1 | `HyperDense-Net_A_Hyper-Densely_Connected_CNN_for_Multi-Modal_Image_Segmentation.pdf` | 3.6MB | 超密集连接多模态分割（经典） | Tier 2 |
| D-2 | `SpectFusion_Cross-modal_Spectrum-aware_Attention_Network_for_Unsupervised_Multimodal_Medical_Image_Fusion.pdf` | 3.1MB | 频谱感知跨模态注意力融合 | — |
| D-3 | `Multi-Modal_Sensor_Medical_Image_Fusion_Based_on_Multiple_Salient_Features_With_Guided_Image_Filter.pdf` | 3.3MB | 多显著特征引导滤波融合 | — |
| D-4 | `fmed-10-1273441.pdf` | 1.5MB | Frontiers in Medicine 多模态 | — |
| D-5 | `fmed-12-1557449.pdf` | 3.1MB | Frontiers in Medicine 多模态 | — |

---

## E. 牙科/口腔影像 DL（12 篇）

| # | 文件名 | 大小 | 精读 |
|---|--------|------|------|
| E-1 | `Deep_Learning_Techniques_for_Dental_Image_Diagnostics_A_Survey.pdf` | 3.7MB | **Tier 1** |
| E-2 | `1-s2.0-S0010482525006304-main.pdf` | 15.2MB | — |
| E-3 | `1-s2.0-S0010482524006541-main.pdf` | 3.1MB | — |
| E-4 | `1-s2.0-S266597272300082X-main.pdf` | 5.3MB | — |
| E-5 | `1-s2.0-S1746809424005688-main.pdf` | 3.3MB | — |
| E-6 | `1-s2.0-S1746809424014976-main.pdf` | 3.0MB | — |
| E-7 | `1-s2.0-S1047320323001463-main.pdf` (+ 2 副本) | 2.0MB | — |
| E-8 | `1-s2.0-S1047320325001105-main.pdf` | 2.6MB | — |
| E-9 | `1-s2.0-S0950705123007372-main.pdf` | 2.3MB | — |
| E-10 | `artificial_intelligence_and_other_modern_digital.2.pdf` | 773KB | — |
| E-11 | `s11042-023-14435-9.pdf` | 4.3MB | — |
| E-12 | `s11042-024-20338-0.pdf` | 4.3MB | — |

---

## F. 骨病变/骨质疏松 DL（10 篇）

| # | 文件名 | 大小 | 精读 |
|---|--------|------|------|
| F-1 | `xie-et-al-2024-a-few-shot-learning-framework-for-the-diagnosis-of-osteopenia-and-osteoporosis-using-knee-x-ray-images.pdf` | 1.2MB | **Tier 1** |
| F-2 | `s00198-014-2707-4.pdf` | 8.8MB | — |
| F-3 | `s11914-021-00701-y.pdf` | 7.0MB | — |
| F-4 | `löffler-et-al-2020-a-vertebral-segmentation-dataset-with-fracture-grading.pdf` | 660KB | — |
| F-5 | `s41598-021-94303-4.pdf` | 1.8MB | — |
| F-6 | `s41598-022-27062-5.pdf` | 4.8MB | — |
| F-7 | `s41598-023-48883-y.pdf` | 2.9MB | — |
| F-8 | `s11517-022-02723-9.pdf` | 2.1MB | — |
| F-9 | `41598_2025_Article_33843.pdf` | 2.1MB | — |
| F-10 | `41598_2025_Article_91430.pdf` | 2.3MB | — |

---

## G. 大型教科书（3 本，不精读）

| # | 文件名 | 大小 | ISBN | 用途 |
|---|--------|------|------|------|
| G-1 | `978-3-030-87193-2.pdf` | 122MB | 978-3-030-87193-2 | Springer 医学图像计算参考书 |
| G-2 | `978-3-030-87196-3.pdf` | 99MB | 978-3-030-87196-3 | 同上 Companion 卷 |
| G-3 | `978-3-031-72120-5.pdf` | 110MB | 978-3-031-72120-5 | Springer 2024 医学图像新著 |

> 不建议将教科书直接作为论文引用，可在 Methods 方法论讨论中参考。

---

## H. 低相关 / 拟排除（18 篇）

| # | 文件名 | 排除原因 |
|---|--------|----------|
| H-1 | `U-Shiftformer_Brain_Tumor_Segmentation...pdf` | 脑肿瘤分割，领域无关 |
| H-2 | `Improved_SwinUnet...charcoal_slag...pdf` | 工业阻燃材料，领域无关 |
| H-3 | `Prior_Information_Guided...Fovea_Localization...pdf` | 视网膜眼底，领域无关 |
| H-4 | `2501.05656-EDL_Jet_Identification.pdf` | 粒子物理喷注，仅方法参考 |
| H-5 | `1-s2.0-S0957417423020766-main.pdf` | 食品科学，不相关 |
| H-6 | `1-s2.0-S037842741301312X-main.pdf` | 毒理学，非氟中毒影像 |
| H-7 | `1-s2.0-S0967586805005018-main.pdf` | 非影像方法 |
| H-8 | `fnins-16-1009581.pdf` | 神经科学影像，非氟中毒 |
| H-9 | `fnins-16-876065.pdf` | 神经科学影像，非氟中毒 |
| H-10 | `fonc-06-00245.pdf` | 肿瘤学，非氟中毒 |
| H-11 | `fphys-14-1088703.pdf` | 生理学，非影像 |
| H-12 | `fsurg-13-1695296.pdf` | 外科学，非氟中毒 |
| H-13 | `ijms-22-11932.pdf` | 分子生物学，非影像 |
| H-14 | `HTL.2017.0066.pdf` | 非氟中毒影像 |
| H-15 | `s42452-024-06127-2.pdf` | 非核心领域 |
| H-16 | `sustainability-15-12227.pdf` | 可持续发展，不相关 |
| H-17 | `s11554-026-01874-4.pdf` | 非核心领域 |
| H-18 | `Tian_2025_Biomed._Phys._Eng._Express_11_065019.pdf` | 非核心领域 |

---

## U. 未分类/待确认文件（21 篇）

以下文件名模糊或无法从文件名判断内容，需打开 PDF 才能确认：

| 文件名 | 大小 |
|--------|------|
| `5f8540a326db9_IJAR-33591.pdf` | 565KB |
| `5aa690be7a0a56099593bb22d71160d1cd0a.pdf` | 11.2MB |
| `6194d2a68304896a6404963681f9dcb47d98.pdf` | 5.3MB |
| `1997a8963dd2983246de11d123d7e6eda153.pdf` | 551KB |
| `673-Article Text-1329-1-10-20240626.pdf` | 371KB |
| `archdisch00877-0122.pdf` | 1.5MB |
| `e2200342.full.pdf` | 743KB |
| `jbres1431.pdf` | 403KB |
| `medip,+IJORO-1819+C.pdf` | 168KB |
| `moving_on_from__no_trace___wrong_place__for_the.1.pdf` | 244KB |
| `Int+J+Comm+Dent+20231110106.pdf` | 264KB |
| `nihms-1523432.pdf` | 1.6MB |
| `sensors-22-06229.pdf` | 8.3MB |
| `10.3934_era.2025129.pdf` | 1.7MB |
| `2109.04335v3.pdf` | 5.6MB |
| `2404.13564v1.pdf` | 29.7MB |
| `s12098-017-2574-z.pdf` | 740KB |
| `s11042-023-14435-9.pdf` | 4.3MB |
| `s11042-024-20338-0.pdf` | 4.3MB |
| `1-s2.0-S0950705123007372-main.pdf` | 2.3MB |
| `1-s2.0-S1047320323001463-main.pdf` | 2.0MB |

---

## 精读优先级汇总

### Tier 1 — 必读（10 篇）

| # | Ref ID | 标题关键词 | 入选理由 |
|---|--------|-----------|----------|
| 1 | A2-1 | Integrated Learning SF Diagnosis | 首个 SF DL 分级论文 (2021, IEEE TII) |
| **2** | **A2-9** | **Mwinc-Mamba SF Diagnosis (BSPC 2025)** | **🆕 SOTA SF 诊断！首个开源 SFXRay 数据集 + SSM** |
| **3** | **A1-7** | **MLTrMR DF Diagnosis (JVCIR 2025)** | **🆕 SOTA DF 诊断！首个开源 DFID 数据集 + ViT 掩码** |
| 4 | A1-1 | Dental Fluorosis Web Detection | DF DL 检测 Web 系统 |
| 5 | A1-4 | Differential Diagnosis of DF | 氟斑牙鉴别诊断标准 |
| 6 | A1-5 | Nature & Mechanisms of DF (Fejerskov) | 氟斑牙病理机制经典 |
| 7 | B-1 | MEDL Mutual Evidential DL | EDL 最新方法（2025） |
| 8 | B-2 | ORCU Ordinal Calibration | 有序回归 EDL — 与 Dean Index 天然匹配 |
| 9 | B-3 | EviVLM EDL + VLM | EDL 多模态融合前沿 |
| 10 | B-4 | REDNet Evidential Discounting | 多模态不确定性融合 |

### Tier 2 — 应读（7 篇）

| # | Ref ID | 标题关键词 | 入选理由 |
|---|--------|-----------|----------|
| 9 | A1-2 | DF Segmentation Quantum-Inspired | 氟斑牙分割方法 |
| 10 | A2-2 | SF Prediction ML (s41598) | SF 预测特征工程参考 |
| 11 | B-5 | Uncertainty Evidential Fusion | EDL 半监督不确定性 |
| 12 | B-6 | DuEDL Scribble Segmentation | 弱监督 EDL 方法 |
| 13 | D-1 | HyperDense-Net Multimodal | 多模态融合架构经典 |
| 14 | E-1 | DL Dental Image Survey | 牙科 DL 全景综述 |
| 15 | F-1 | Few-Shot Osteopenia Knee X-ray | 小样本骨病分类 — 与 SF 场景高度相似 |

---

> **2026-05-15 更新**：Phase 2 精读已完成（19 篇深读摘要，含 4 篇新增）。2025 年 Xu/Wu (贵州大学) 的 Mwinc-Mamba (SF) 和 MLTrMR (DF) 是最直接的 SOTA 竞品——均构建了世界首个开源数据集，是本项目必须超越的基线。见 `02_Literature/02_deep_read/16-19`。
