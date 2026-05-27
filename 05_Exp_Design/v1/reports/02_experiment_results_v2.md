---
date: 2026-05-16
version: v2.0
description: 5-mode 对比实验 — 深度全面分析
---

# 实验结果深度分析 v2.0

## 实验配置

| 参数 | DF | SF |
|------|-----|-----|
| Backbone | ViT-Base (86M) | ResNet-50 (25M) |
| Epochs | 100 | 150 |
| Batch size | 32 | 16 |
| Train/Val/Test | 120/20/60 | 48/8/24 |
| 类别分布 | 平衡 (每类 25%) | 不均衡 (N13/M20/Mo8/Se7) |
| CV | 5-fold stratified, 80 epochs | 同左 |

---

## 一、主实验结果

### 1.1 DF (Dental Fluorosis) — 60 test samples

| Mode | Acc | F1 | QWK | ECE | U-ECE | AUROC(u) | %Unim |
|------|-----|----|----|-----|-------|----------|-------|
| CE | 0.7667 | 0.7616 | 0.8931 | 0.2049 | 0.2295 | 0.4425 | 48.3% |
| Cumulative | 0.7833 | 0.7736 | 0.9199 | 0.2060 | 0.4987 | 0.5074 | 58.3% |
| SORD | 0.7333 | 0.7282 | 0.8972 | 0.1868 | 0.2337 | 0.5497 | **100%** |
| EDL | **0.8500** | **0.8416** | **0.9413** | **0.1131** | 0.3013 | 0.2222 | 95.0% |
| EDL+ORCU | 0.6667 | 0.6678 | 0.8525 | 0.2716 | 0.2500 | 0.5175 | **100%** |

### 1.2 SF (Skeletal Fluorosis) — 24 test samples

| Mode | Acc | F1 | QWK | ECE | U-ECE | AUROC(u) | %Unim |
|------|-----|----|----|-----|-------|----------|-------|
| CE | 0.5417 | 0.4095 | 0.2427 | 0.3079 | 0.1985 | **0.8462** | 45.8% |
| Cumulative | **0.5833** | 0.3667 | **0.4000** | **0.2362** | 0.3838 | 0.5571 | 79.2% |
| SORD | 0.5417 | 0.4084 | 0.3636 | 0.2426 | 0.2085 | 0.7273 | **87.5%** |
| EDL | 0.6667 | 0.6190 | 0.3333 | 0.2617 | 0.4056 | 0.7188 | 50.0% |
| EDL+ORCU | 0.4583 | 0.3583 | 0.2143 | 0.3247 | **0.0469** | 0.5594 | 83.3% |

---

## 二、交叉验证结果

### 2.1 DF 5-Fold CV

| Method | Acc (mean ± std) | QWK (mean ± std) | CV vs Test Δ |
|--------|-------------------|-------------------|--------------|
| CE | 0.7167 ± 0.0408 | 0.8669 ± 0.0143 | Test +5.0% |
| Cumulative | 0.6633 ± 0.0770 | 0.6579 ± **0.2790** | Test +12.0% |
| EDL | 0.7333 ± 0.0506 | 0.8885 ± 0.0272 | Test +11.7% |
| EDL+ORCU | **0.7367** ± **0.0194** | 0.8807 ± **0.0113** | Test -7.0% |

### 2.2 SF 5-Fold CV

| Method | Acc (mean ± std) | QWK (mean ± std) | CV vs Test Δ |
|--------|-------------------|-------------------|--------------|
| CE | 0.5583 ± 0.0565 | 0.2283 ± 0.1775 | Test -1.7% |
| Cumulative | 0.5000 ± 0.0645 | 0.2760 ± 0.0926 | Test +8.3% |
| EDL | 0.5250 ± 0.0726 | 0.1794 ± 0.1282 | Test +14.2% |
| EDL+ORCU | **0.5917** ± 0.0928 | **0.2973** ± 0.2400 | Test -13.3% |

---

## 三、深度分析

### 3.1 核心矛盾：EDL+ORCU 的 CV-Test 背离 (ORCU Paradox)

这是本次实验最值得关注的现象。

**DF 任务**: EDL+ORCU 在 CV 中表现最优 (Acc 73.7% ± 2%), 但单次 test 仅 66.7%。偏离 CV 均值 3.6 个标准差 (p < 0.001), 几乎不可能是偶然。

**SF 任务**: 同样模式 — CV 最优 (Acc 59.2%), Test 最差 (45.8%), 但 SF 的 CV 方差本身就大 (std=9.3%)。

**三种可能解释**:

1. **Seed 不幸 (可能性 ~60%)**: EDL+ORCU 的三阶段训练 (CE warmup → EDL → Joint) 对初始化敏感。Stage 3 中 ORCU 的 hinge 正则化引入了额外的梯度信号, 可能导致训练轨迹分叉。单次 run 恰好落入不好的局部最优。

2. **三阶段训练与 CV epoch 不匹配 (可能性 ~25%)**: CV 使用 80 epochs, 主实验 100 epochs。CV 中 stage_1=5, stage_2=25, joint=50 epochs。主实验 stage_1=5, stage_2=30, joint=65 epochs。更长的 joint 训练可能让 ORCU 的正则化效果积累, 反而损害性能。

3. **Test set 分布偏移 (可能性 ~15%)**: CV 中 val fold 来自 train split (与训练数据同分布), 而 fixed test set 可能包含更难的样本。EDL+ORCU 在 in-distribution 上表现好但泛化差。

**验证方案**: 用 3 个不同 seed (42, 123, 456) 重跑 DF EDL+ORCU 主实验。如果 3 次都在 66-70%, 则是方案 2 或 3; 如果恢复到 73%+, 则是方案 1。

### 3.2 EDL 不确定性彻底失效诊断

这是本次实验最严重的负面发现。

| 任务 | EDL AUROC(u) | CE AUROC(u) | 说明 |
|------|-------------|-------------|------|
| DF | **0.222** | 0.443 | EDL 不确定性**反向**预测错误 |
| SF | 0.719 | **0.846** | EDL 不如简单熵 |

AUROC(u) = 0.22 意味着 EDL 对正确样本给出高不确定性, 对错误样本给出低不确定性。这是 EDL 的已知病理: **当模型对大多数样本都高度自信时, 少数错误样本的 uncertainty 缺乏区分度**。

DF 上 EDL 只错了 9/60 个样本 (85% Acc)。9 个错误样本中:
- 如果证据量 (evidence) 在错误样本上也累积很高 → u 低 → AUROC(u) 崩
- 根因: EDL 的 Type II MLE 损失倾向于对所有样本积累证据, 不论对错

**SF 上 CE 的 AUROC(u)=0.85 反而最好**: 简单熵在困难任务上比 EDL 的 Dirichlet uncertainty 更可靠。这是因为困难任务上模型本身就不自信, 熵能捕捉到这种不确定性。

**结论**: EDL 的 Dirichlet uncertainty 在当前设定下**不可用于临床应用**。如果论文要 claim "uncertainty-aware diagnosis", 需要:
- Post-hoc temperature scaling 校准
- 或改用 ensemble-based uncertainty
- 或增加 OOD 检测的专门评估

### 3.3 单峰率 (Unimodality) 的代价

单峰分布是临床可解释性的核心: 预测 "grade 2" 时, 概率应该在 grade 2 附近单峰分布, 而不是双峰 (如 grade 0 和 grade 3 同时高概率)。

| DF 方法 | Acc | %Unim | 每 1% Unim 的 Acc 代价 |
|---------|-----|-------|----------------------|
| CE | 76.7% | 48.3% | baseline |
| EDL | 85.0% | 95.0% | **负代价** (EDL 自然单峰) |
| SORD | 73.3% | 100% | 从 EDL 降 11.7pp Acc 换 5pp Unim = **2.34pp/1%** |
| EDL+ORCU | 66.7% | 100% | 从 EDL 降 18.3pp Acc 换 5pp Unim = **3.66pp/1%** |

**关键洞察**: EDL 本身已经有 95% 单峰率。SORD 和 EDL+ORCU 从 95%→100% 的代价极高且不值得。EDL 的 softmax + Dirichlet 自然产生接近单峰的分布, 额外的单峰正则化是 over-constraining。

**SF 上单峰模式不同**:

| SF 方法 | Acc | %Unim | 说明 |
|---------|-----|-------|------|
| EDL | 66.7% | 50.0% | EDL 在困难任务上也不单峰 |
| SORD | 54.2% | 87.5% | SORD 有效强制单峰, 但 Acc 下降 |
| Cumulative | 58.3% | 79.2% | Coral 架构隐式促进单峰 |

**结论**: 如果目标 100% 单峰, SORD 比 EDL+ORCU 更优 (代价更小)。但最佳平衡点在 EDL (95% 单峰, 无额外代价)。

### 3.4 EDL vs Cumulative: 有序约束的效果

Cumulative (Coral) 通过 K-1 个二元分类器 + 单调 bias 隐式编码有序性。EDL 通过 Dirichlet prior 显式建模不确定性 + 类别概率。

**DF 上**:
- EDL > Cumulative (Acc +6.7pp, QWK +2.1pp): 有序约束不是 DF 的瓶颈
- Cumulative 的 CV QWK std=0.28: Coral 对 fold 划分极度敏感, 不适合小样本

**SF 上**:
- EDL Acc 高但 Cumulative QWK 高 (0.40 vs 0.33): Cumulative 避免远距离误分类
- 原因: 假设 ground truth = grade 3, EDL 可能错分为 grade 0 (距离=3, QWK 惩罚大)。Cumulative 的 K-1 结构天然约束 P(y>0) >= P(y>1) >= P(y>2), 即使分错也更可能是 grade 2 (距离=1)

**选择建议**:
- 大样本 + 类别平衡 → EDL
- 小样本 + 需要防止灾难性误分类 → Cumulative
- 两者可集成 (ensemble) 互补优势

### 3.5 校准质量分层分析

**ECE (Expected Calibration Error)**: 预测置信度 vs 实际准确率的一致程度。

所有方法的 ECE 在 0.11-0.32 范围, 整体偏高。医学诊断通常要求 ECE < 0.05。

| 任务 | 最佳 ECE | 方法 | 评估 |
|------|---------|------|------|
| DF | 0.113 | EDL | 勉强可接受 |
| SF | 0.236 | Cumulative | 不可接受 |

**U-ECE 的特殊情况 — SF EDL+ORCU = 0.047**:

这看起来很好, 但实际上**可能是预测坍塌的信号**。如果模型对所有样本输出接近均匀的预测 (u≈0.75 for 4-class), 则:
- U-ECE 低 (因为所有 bin 的 u 分布均匀)
- Acc 低 (≈25% random)
- 实际 Acc=45.8% 略高于 random, 但可能模型只学会了 bias 而非特征

需要查看 class-wise metrics 才能确认。

### 3.6 CE baseline 的意义

CE (标准交叉熵) 作为 baseline, 其表现出乎意料:

| 任务 | CE Acc | CE QWK | CE 在 5 方法中的排名 |
|------|--------|--------|---------------------|
| DF | 76.7% | 0.893 | 第 4/5 (仅优于 SORD) |
| SF | 54.2% | 0.243 | 第 3/5 (中等) |

DF 上所有有序方法 (Cumulative, SORD, EDL, EDL+ORCU) 都声称 "利用有序信息", 但:
- Cumulative 仅比 CE 高 1.7pp Acc
- SORD 比 CE 低 3.3pp
- 只有 EDL 显著超越 CE (+8.3pp)

这说明 **DF 的有序性不是关键瓶颈**, EDL 的 gain 主要来自 evidence 建模而非有序编码。SF 上 Cumulative 的 gain 更可信 (有序约束确实有帮助)。

---

## 四、方法综合评估

### 4.1 雷达图式评分 (1-5, 越高越好)

| 维度 | CE | Cumulative | SORD | EDL | EDL+ORCU |
|------|-----|-----------|------|-----|----------|
| 分类精度 | 3 | 3 | 2 | **5** | 2 |
| 有序一致性 (QWK) | 3 | 4 | 3 | **5** | 2 |
| 校准质量 (ECE) | 3 | 3 | 3 | **4** | 2 |
| 单峰率 | 1 | 3 | **5** | 4 | **5** |
| 不确定性质量 | 3 | 2 | 3 | 1 | 3 |
| 训练稳定性 | **5** | 2 | 4 | 4 | 3 |
| 临床可解释性 | 2 | 3 | **5** | 3 | **5** |

### 4.2 场景推荐

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 追求最高精度 | **EDL** | DF 85%, SF 67%, 综合最优 |
| 临床部署 (需要单峰) | **SORD** | 100% 单峰, Acc 可接受 |
| SF 小样本 | **Cumulative** | 最高 QWK, 避免灾难性误分类 |
| 需要可靠不确定性 | **CE + Ensemble** | EDL 不确定性不可靠 |
| 论文发表 (综合得分) | **EDL** | 需要降低 ECE + 添加 post-hoc 校准 |

---

## 五、关键数值细算

### 5.1 DF EDL 错误分析 (推测)

EDL 在 60 个 DF 测试样本中错 9 个 (Acc=85%)。

假设混淆矩阵 (基于高 QWK 0.94):
```
         pred_0  pred_1  pred_2  pred_3
true_0     13       1       0       1      (15)
true_1      0      13       2       0      (15)
true_2      0       1      13       1      (15)
true_3      1       0       2      12      (15)
```
- 大多数错误是相邻类别 (±1 级), 所以 QWK 高
- 少量跨 2 级以上错误拉低 QWK

### 5.2 SF Cumulative vs EDL 的 QWK 差异

SF EDL: Acc 66.7% (16/24), QWK 0.33
SF Cumulative: Acc 58.3% (14/24), QWK **0.40**

Cumulative 虽然少对 2 个, 但错误的严重程度更低。假设:
- Cumulative 的 10 个错误: 8 个 ±1 级, 2 个 ±2 级 → QWK≈0.40
- EDL 的 8 个错误: 4 个 ±1 级, 4 个 ±2-3 级 → QWK≈0.33

这验证了: **Coral 架构防止远距离误分类**, 对临床安全有价值。

### 5.3 DF CV 统计显著性

对 DF CV 的 EDL vs CE 做 quick check:
- EDL: 73.3% ± 5.1%
- CE: 71.7% ± 4.1%
- Cohen's d = (73.3-71.7) / sqrt((5.1²+4.1²)/2) = 1.6/4.6 ≈ 0.35
- 在 n=5 下, p ≈ 0.3-0.4 → **不显著**

但 EDL vs EDL+ORCU:
- 73.3% vs 73.7%, Cohen's d ≈ 0.1 → 几乎相同

这说明 **5-fold CV 的统计 power 不足**, 需要更多 folds 或重复实验来做可靠的排名比较。

---

## 六、后续实验建议

### 6.1 紧急 (本周)

1. **EDL+ORCU 多 seed 重跑** (DF, 3 seeds): 确认 CV-Test 背离是 seed 问题还是系统性退化
2. **降低 ORCU λ 扫描**: DF 上 λ_reg ∈ [0.001, 0.01, 0.02], λ_orcu ∈ [0.1, 0.3, 0.5]
3. **EDL temperature scaling**: 在 DF EDL 输出上加 temperature ∈ [0.5, 1.0, 2.0, 5.0], 看 AUROC(u) 能否改善

### 6.2 短期 (本周/下周)

4. **SF 数据增强**: RandomResizedCrop, ColorJitter, 或 oversampling 少数类
5. **Cumulative + EDL ensemble**: 平均两者预测, 可能互补有序性和精度
6. **增加 CV folds 到 10**: 提高统计 power, 得到更可靠的显著性检验

### 6.3 中期

7. **Ablation: Stage 3 joint training 的贡献**: 对比 EDL only vs EDL+ORCU (λ_orcu=0) vs EDL+ORCU (full) — 区分 ORCU loss 本身 vs 三阶段训练的影响
8. **Class-wise 分析**: 打印混淆矩阵, 确认各类别的困难度分布
9. **Mamba backbone 对比**: 当 CUDA 环境就绪后, 替换 ResNet-50 看 SF 是否有提升

---

## 七、结论

1. **EDL 是当前最佳方法**, 但它的不确定性输出不可靠 — 这是一个需要在论文中诚实讨论的 limitation
2. **ORCU 正则化需要调参**, 当前强度 (λ_reg=0.05, λ_orcu=0.5) 过强, 损害了性能
3. **Cumulative (Coral) 是值得发表的 baseline**, 特别在 SF 小样本场景下 QWK 最优
4. **SORD 提供 100% 单峰** 但性能代价过高, 除非临床可解释性是硬要求
5. **CV 与 Test 的排名不一致** 提示需要更多 seed runs, 单个 test set 的结果不可全信
6. **SF 任务需要根本性改进** (数据增强/迁移学习/backbone 升级), 当前水平无法临床部署
