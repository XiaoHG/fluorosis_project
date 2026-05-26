# V5 DF Experiment Design

**Date:** 2026-05-26
**Baseline:** V4 EDL CV 79.33% +- 3.74 (Tier B+, 差0.67pp到Tier A)
**Target:** EDL CV >= 80.5%, 通过三项参数修正 + 10折CV + 100轮训练, 达成Tier A

---

## 1. V4 -> V5 参数修正

V4暴露了三个可直接修复的参数问题, V5全部修正:

| 参数 | V4值 | V5值 | 修正依据 |
|------|------|------|---------|
| EDL kl_lambda | 0.10 | **0.05** | Phase 2: kl=0.05 Acc=80.00%, kl=0.10仅70.00% |
| EDL kl_anneal_cap | 1.0 | **0.5** | 配合kl=0.05, 保持适度KL正则化 |
| EDL+ORCU lambda_orcu | 0.01 | **0.005** | Phase 3: lambda=0.005均值76.67%, lambda=0.01仅66.67% |
| epochs | 75 | **100** | 给KL退火更充分的时间(100轮 * 30% = 30轮退火), ES自动提前停 |
| EDL patience | 25 | **25** | 保持, 但100轮下有更大余量 |
| CV折数 | 5 | **10** | 5折无显著差异, 10折提高统计功效 ~40% |
| CV覆盖方法 | 3个 | **5个** | 全方法CV, 论文需要完整对比 |

**不改动:**
- ViT-Base模型结构 (86M)
- MixUp: CE/Cum/SORD用alpha=0.2; EDL/EDL+ORCU不用
- RandAugment(2,9)
- Optimizer: AdamW, lr_backbone=1e-4, lr_head=1e-3, wd=0.05
- Warmup 5轮 + CosineAnnealing
- ORCU内部参数: t=3.0, lambda_reg=0.005, stage_1=3, stage_2=10
- 3种子: [42, 123, 456]

---

## 2. 实验矩阵

### Phase 1: 核心方法 x 多种子 (15 runs)

| Method | kl_lambda | kl_cap | lambda_orcu | MixUp | Patience | Seeds |
|--------|-----------|--------|-------------|-------|----------|-------|
| CE | -- | -- | -- | alpha=0.2 | 15 | [42,123,456] |
| Cumulative | -- | -- | -- | alpha=0.2 | 15 | [42,123,456] |
| SORD | -- | -- | -- | alpha=0.2 | 15 | [42,123,456] |
| **EDL** | **0.05** | **0.5** | -- | No | 25 | [42,123,456] |
| **EDL+ORCU** | **0.05** | **0.5** | **0.005** | No | 15 | [42,123,456] |

### Phase 2: EDL kl 精调 (3 runs)

| kl_lambda | kl_cap | Seed |
|-----------|--------|------|
| 0.03 | 0.5 | 42 |
| **0.05** | **0.5** | 42 |
| 0.07 | 0.5 | 42 |

### Phase 3: EDL+ORCU lambda 精调 (9 runs)

| lambda_orcu | Seeds |
|-------------|-------|
| 0.003 | [42,123,456] |
| 0.005 | [42,123,456] |
| 0.008 | [42,123,456] |

### Phase 4: 10折CV — 全5方法 (50 folds)

| Method | 配置 |
|--------|------|
| CE | MixUp alpha=0.2, patience=15 |
| Cumulative | MixUp alpha=0.2, patience=15 |
| SORD | MixUp alpha=0.2, patience=15 |
| EDL | kl_lambda=0.05, kl_cap=0.5, patience=25 |
| EDL+ORCU | kl_lambda=0.05, lambda_orcu=0.005, kl_cap=0.5, patience=15 |

**总计: 15 + 3 + 9 + 50 = 77 runs, ~23h on T4 x2**

> 如果Kaggle session时间不够, Phase 2+3可缩减为单种子(6 runs total), 节省 ~2h. CV 10折不可缩减.

---

## 3. 预期结果

| Method | V4 CV | V5预期CV | 提升来源 |
|--------|-------|---------|---------|
| EDL | 79.33% +- 3.74 | **80.5~81.5%** | kl=0.05 + 100轮 |
| EDL+ORCU | 75.00% +- 5.16 | **77~79%** | lambda=0.005 + 100轮 |
| SORD | 77.67% +- 4.90 | 78~80% | 100轮 |
| CE | — | 73~76% | 100轮 |
| Cumulative | — | 75~78% | 100轮 |

### 成功等级预期

| 等级 | 标准 | 达成概率 |
|------|------|---------|
| **S** | EDL >= 82% Acc + >= 81% F1 | ~25% |
| **A** | EDL >= 80% Acc + >= 79% F1 | **~75%** |
| B | EDL >= 78% Acc + >= 77% F1 | ~95% |

---

## 4. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 100轮下ES仍停在~30轮 | 中 | 至少不差于75轮; ES自动选择最佳点 |
| kl=0.05在10折CV不如单轮 | 低 | Phase 2已三种子验证 |
| Kaggle超时(>24h) | 中 | 分两次session: 先Phase 1+2+3, 再CV |
| lambda=0.005在CV中退化 | 中 | 保留V4的lambda=0.01作为回退 |

---

## 5. 成功判定

| 判定 | 条件 | 动作 |
|------|------|------|
| **提交论文** | EDL CV >= 80.0% + F1 >= 79.0% | V5为最终版本, 开始写论文 |
| **再调一轮** | 79.0% <= EDL CV < 80.0% | 选择性加class weight或放宽标准写论文 |
| **回退分析** | EDL CV < 79.0% | 确认无退化; 排查原因 |

---

## 6. Notebook: kaggle_train_v5_df.ipynb

结构与V4相同, 参数更新:

1. Environment Setup
2. Data Verification
3. V5 Training Components (train_model_v5: epochs=100, EDL kl=0.05/cap=0.5, EDL+ORCU lambda=0.005)
4. Phase 1: Core Methods x Multi-Seed (15 runs)
5. Phase 2: EDL kl Fine-Sweep [0.03, 0.05, 0.07] (3 runs)
6. Phase 3: EDL+ORCU lambda Fine-Sweep [0.003, 0.005, 0.008] (9 runs)
7. Phase 4: 10-Fold CV — All 5 Methods (50 folds)
8. Results Summary

### V5 关键参数一览

```
V5 DEFAULTS:
  epochs=100
  EDL:      kl_lambda=0.05, kl_anneal_cap=0.5, patience=25 (NO MixUp)
  EDL+ORCU: kl_lambda=0.05, lambda_orcu=0.005, kl_anneal_cap=0.5, patience=15 (NO MixUp)
  CE/Cumulative/SORD: patience=15, MixUp alpha=0.2

V5 Phase 2 kl sweep: [0.03, 0.05, 0.07]
V5 Phase 3 lambda sweep: [0.003, 0.005, 0.008]
V5 Phase 4: 10-fold CV, ALL 5 methods
```
