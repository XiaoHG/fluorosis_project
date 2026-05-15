---
date: 2026-05-15
author: LitAgent
input_from: "01_Knowledge_Base/ 全部文件"
output_to: "02_Literature/01_initial_search/literature_list.md"
status: draft
---

# 文献初筛列表

## 检索信息

| 项目 | 内容 |
|------|------|
| 检索日期 | 2026-05-15 |
| 数据库 | Semantic Scholar / PubMed / IEEE Xplore / arXiv / Springer Nature |
| 时间范围 | 2017–2026（经典方法不限时间） |
| 初筛数量 | 28 篇 |
| 排序原则 | 直接相关度 > 方法可迁移性 > 方法学参考价值 |

### 检索式

1. `"dental fluorosis" AND ("deep learning" OR "CNN" OR "neural network" OR "classification")`
2. `"skeletal fluorosis" AND ("deep learning" OR "machine learning" OR "X-ray" OR "diagnosis")`
3. `"fluorosis" AND ("computer-aided diagnosis" OR "automated" OR "screening")`
4. `("dental caries" OR "enamel" OR "hypomineralization") AND ("deep learning" AND ("intraoral" OR "photograph" OR "classification"))`
5. `("osteoporosis" OR "bone disease") AND ("X-ray" OR "radiograph") AND ("deep learning" AND "classification" AND "grading")`
6. `("Grad-CAM" OR "explainable AI") AND ("dental" OR "bone" OR "medical image") AND "classification"`

---

## 初筛文献列表

### 一、氟中毒直接相关 (3 篇)

| # | 文献 | 摘要 | 相关度 |
|---|------|------|--------|
| 1 | **Integrated Learning Approach Based on Fused Segmentation Information for Skeletal Fluorosis Diagnosis and Severity Grading**. IEEE Transactions, 2025. | 提出融合分割信息的集成学习方法，对氟骨症进行诊断和严重程度分级。目前唯一直接针对氟骨症 X 光影像做 DL 分级的工作。 | ⭐⭐⭐⭐⭐ |
| 2 | **Predicting skeletal fluorosis severity using machine learning across diverse fluoride-exposed populations in China**. Long H, Zeng J, Wei S et al., *Scientific Reports*, 2026. | 使用 RF + SHAP 对 1309 例中国三省氟中毒人群进行 SF 严重程度预测，AUC=0.832。基于临床/环境特征而非影像，但提供了重要的特征工程和可解释性参考。 | ⭐⭐⭐⭐ |
| 3 | **A machine learning approach to dental fluorosis classification**. *Arabian Journal of Geosciences*, 2021. | 使用 ANN/SVM/Naive Bayes 对氟斑牙进行分类。非 DL 方法且非图像输入（基于水质特征），但为少有的氟斑牙自动化分类尝试，展示了传统 ML 基线水平。 | ⭐⭐⭐ |

### 二、牙科影像 DL — 方法可迁移 (11 篇)

| # | 文献 | 摘要 | 相关度 |
|---|------|------|--------|
| 4 | **Detection and localization of caries and hypomineralization on dental photographs with a vision transformer model**. *npj Digital Medicine*, 2023. | 使用 SegFormer-B5 (ViT) 在 18,179 张口内照片上进行龋齿和 MIH（釉质发育不全）的像素级检测与定位，IoU=0.959。MIH 与氟斑牙的视觉特征高度相似，方法可直接迁移。 | ⭐⭐⭐⭐⭐ |
| 5 | **Automated Classification of Enamel Caries from Intraoral Images Using Deep Learning Models**. *J. Clinical Medicine*, 2025. | 提出 ExplainableDentalNet（轻量 CNN）和 Interpretable ResNet50-SE（+Squeeze-Excitation），在 2000 张口内照上做釉质龋三分类，准确率 98.30%。Grad-CAM 可解释性设计值得借鉴。 | ⭐⭐⭐⭐⭐ |
| 6 | **Caries detection with tooth surface segmentation on intraoral photographic images using deep learning**. *BMC Oral Health*, 2022. | U-Net 牙齿分割 + ResNet-18 分类 + Faster R-CNN 定位的级联流程，2348 张口内照片，AUC 从 0.731 提升至 0.837。证明了牙齿表面分割预处理能显著提升分类性能。 | ⭐⭐⭐⭐ |
| 7 | **The application of deep learning in early enamel demineralization detection**. *BMC Oral Health*, 2025. | 基于 Mask R-CNN (ConvNext backbone) 做早期釉质脱矿检测和分割，F1=0.856，显著提升低年资牙医诊断水平。场景与氟斑牙白垩色病变检测高度一致。 | ⭐⭐⭐⭐ |
| 8 | **Deep Learning for Caries Detection and Classification**. *Diagnostics*, 2021. | nnU-Net 分割 + DenseNet121 进行 D1/D2/D3 三级龋深度分类，1160 张全景片，模型与牙医表现无显著差异。nnU-Net 自适配能力适合小数据集。 | ⭐⭐⭐ |
| 9 | **CariesNet: a deep learning approach for segmentation of multi-stage caries lesion from oral panoramic X-ray image**. *Neural Computing and Applications*, 2022. | U-shape + Full-Scale Axial Attention，3127 张全景片，Dice=93.64%。全尺度轴向注意力模块对小目标分割有效。 | ⭐⭐⭐ |
| 10 | **An end-to-end deep-learning system for segmentation and classification of dental caries from radiovisiography images**. *J. Conservative Dentistry*, 2025. | U-Net 分割 + ResNet-50 三分类，1200 张 RVG，Acc=93.2%。端到端流程设计可作为氟斑牙分级 pipeline 参考。 | ⭐⭐⭐ |
| 11 | **CNN-based remote dental diagnosis model for caries detection with Grad-CAM**. *Scientific Reports*, 2025. | 提出 ResFC 模型直接在全口照片上检测龋齿，Acc=99.78%，结合 Grad-CAM 精确定位。证明无需牙齿裁剪即可端到端诊断。 | ⭐⭐⭐ |
| 12 | **CariesXplainer: enhancing dental caries detection using Gradient-weighted Class Activation Mapping and transfer learning**. *Multimedia Tools and Applications*, 2025. | MobileNetV3 + Grad-CAM，Acc=99.50%。轻量级迁移学习 + 可解释性方案，适合小样本场景。 | ⭐⭐⭐ |
| 13 | **Evaluation of transfer ensemble learning-based CNN models for identification of chronic gingivitis from oral photographs**. *BMC Oral Health*, 2024. | 迁移学习 + 集成学习（ResNet/GoogLeNet/VGG/AlexNet），683 张小样本口内照。小样本迁移学习方案高度可参考。 | ⭐⭐⭐ |
| 14 | **Multi Oral Disease Classification from Panoramic Radiograph using Transfer Learning and XGBoost**. *IJACSA*, 2022. | CNN 特征提取 + XGBoost 分类混合方案，6 种口腔疾病。CNN + 传统分类器的混合思路可参考。 | ⭐⭐ |

### 三、骨 X 光 DL — 方法可迁移 (6 篇)

| # | 文献 | 摘要 | 相关度 |
|---|------|------|--------|
| 15 | **Deep learning for screening primary osteopenia and osteoporosis using spine radiographs and patient clinical covariates in a Chinese population**. Mao L et al., *Frontiers in Endocrinology*, 2022. | DenseNet 双通道（正位+侧位）对腰椎 X 光进行骨质疏松三分类，AUC=0.937。双视图输入 + 临床协变量融合可迁移至氟骨症。 | ⭐⭐⭐⭐⭐ |
| 16 | **Comparative efficacy of AP and lateral X-ray based deep learning in detection of osteoporotic vertebral compression fracture**. *Scientific Reports*, 2024. | EfficientNet-B5，1507 例正侧位脊柱 X 光，AUC=0.953。多视图对比方案可迁移至多部位氟骨症诊断。 | ⭐⭐⭐⭐ |
| 17 | **Fusion of X-Ray Images and Clinical Data for a Multimodal Deep Learning Prediction Model of Osteoporosis**. *JMIR Medical Informatics*, 2025. | CNN + 临床特征多模态融合用于骨质疏松预测，AUC>0.90。影像 + 临床参数融合框架可直接移植。 | ⭐⭐⭐⭐ |
| 18 | **Convolutional neural network for automated classification of osteonecrosis and related mandibular trabecular patterns**. Baseri Saadi S et al., 2022. | ResNet152V2/InceptionResNetV2/MobileNetV2 分类下颌骨骨小梁模式（正常/受累/坏死），Acc=96%，含 Grad-CAM。骨小梁纹理分析与氟骨症骨小梁增粗/模糊评估高度相关。 | ⭐⭐⭐⭐ |
| 19 | **Deep Convolutional Neural Network for Automated Staging of Periodontal Bone Loss Severity on Bite-wing Radiographs: An Eigen-CAM Explainability Mapping Approach**. *J. Periodontology*, 2024. | YOLOv8 牙槽骨丢失四级分期 + Eigen-CAM。骨病变分级 + 可解释性方案直接可参考。 | ⭐⭐⭐ |
| 20 | **A new superfluity deep learning model for detecting knee osteoporosis and osteopenia in X-ray images**. *Scientific Reports*, 2024. | 提出 Superfluity DL 架构，膝 X 光三分类，Acc=85.42%。小样本骨病分类的前沿尝试。 | ⭐⭐ |

### 四、可解释性与注意力机制 (4 篇)

| # | 文献 | 摘要 | 相关度 |
|---|------|------|--------|
| 21 | **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**. Selvaraju RR et al., *ICCV*, 2017. | Grad-CAM 奠基性工作。所有 Agent 需理解其原理，用于氟斑牙/氟骨症模型可解释性设计。引用数 >15,000。 | ⭐⭐⭐⭐⭐ |
| 22 | **Attention-guided convolutional network for bias-mitigated and interpretable oral lesion classification**. *Scientific Reports*, 2024. | 提出 GAIN（Guided Attention Inference Network），用分割 mask 引导 CNN 注意力，16 类口腔病变。引导注意力方案可参考。 | ⭐⭐⭐ |
| 23 | **An Attention-Enhanced Deep Learning Framework for Multi-Label Dental Findings Classification from Panoramic Radiographs**. *MDPI Information*, 2026. | EfficientNet-B4 + CBAM + Grad-CAM，全景片多标签。双注意力 + Grad-CAM 双重可解释性方案。 | ⭐⭐⭐ |
| 24 | **Classification of Dental Radiographs Using Deep Learning**. *J. Clinical Medicine*, 2021. | ResNet-34/CapsNet 牙科 X 光分类 benchmark，Acc>98%，含 Grad-CAM。牙科 AI 分类标准评估流程。 | ⭐⭐⭐ |

### 五、经典 Backbone 与方法论 (4 篇)

| # | 文献 | 摘要 | 相关度 |
|---|------|------|--------|
| 25 | **Deep Residual Learning for Image Recognition**. He K et al., *CVPR*, 2016. | ResNet 原始论文。医学影像分类最常用的 backbone，所有对比实验必需。 | ⭐⭐⭐⭐⭐ |
| 26 | **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**. Tan M & Le QV, *ICML*, 2019. | EfficientNet 原始论文。轻量高精度，医学小样本场景表现优异。 | ⭐⭐⭐⭐ |
| 27 | **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**. Dosovitskiy A et al., *ICLR*, 2021. | Vision Transformer (ViT) 原始论文。在牙科影像中已有成功应用（如文献 4）。 | ⭐⭐⭐⭐ |
| 28 | **U-Net: Convolutional Networks for Biomedical Image Segmentation**. Ronneberger O et al., *MICCAI*, 2015. | U-Net 原始论文。医学图像分割基石，几乎所有牙科/骨病变分割方法的基础。 | ⭐⭐⭐⭐ |

---

## 统计摘要

| 类别 | 数量 | 相关度均值 |
|------|------|-----------|
| 氟中毒直接相关 | 3 | 4.0 |
| 牙科影像 DL（可迁移） | 11 | 3.5 |
| 骨 X 光 DL（可迁移） | 6 | 3.7 |
| 可解释性/注意力 | 4 | 3.5 |
| 经典 Backbone/方法 | 4 | 4.5 |
| **合计** | **28** | **—** |

## 初步缺口观察

1. **氟斑牙直接 DL 研究极度匮乏**：仅检索到 1 篇传统 ML（非 DL）的氟斑牙分类论文，基于图像 DL 的氟斑牙自动分级研究近乎空白。
2. **氟骨症 DL 研究极少**：仅 1 篇 IEEE 论文直接针对骨骼 X 光影像，另 1 篇基于非影像临床数据。
3. **多模态融合在氟中毒领域完全空白**：白光+荧光双模态、X 光+临床参数融合等均未见应用于氟中毒诊断。
4. **可解释性在牙科 DL 已成标配**：Grad-CAM 和注意力机制在龋齿/牙周病 DL 中广泛应用，为本课题提供了成熟的方法论基础。
5. **小样本迁移学习方案成熟**：多项龋齿/口腔病变研究在小样本（200–600 张）下通过迁移学习取得了良好效果，对本项目 200 张氟斑牙数据有直接参考价值。

---

> **下一步**：请人类审核初筛列表，确认文献覆盖度和相关性。确认后进入精读阶段（从 28 篇中选出 Top 10 生成深度摘要）。
