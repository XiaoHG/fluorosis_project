---
date: 2026-05-16
version: v2.1
description: 5-mode 对比实验 v2.1 — lambda_reg 0.05→0.01 修复 + Temperature Calibration + Lambda Sweep
---

# 实验结果分析 v2.1

## 实验变更 vs v2.0

| 变更 | v2.0 | v2.1 | 影响 |
|------|------|------|------|
| orcu_lambda_reg | 0.05 | 0.01 | EDL+ORCU hinge 惩罚减弱 5x |
| Temperature scaling | 无 | T ∈ [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0] | DF EDL 校准扫描 |
| Lambda sweep | 无 | λ_orcu × λ_reg (3×3 grid) | ORCU 超参寻优 |
| Confusion matrix | 无 | 输出到 test_metrics | 类别级错误分析 |
| Training code | 命令行 train.py | Notebook 内联 train_model() | 可能有 dispatch bug |

---

## 一、主实验结果

### 1.1 DF (Dental Fluorosis) — 60 test samples

| Mode | Acc | F1 | QWK | ECE | U-ECE | AUROC(u) | %Unim |
|------|-----|----|----|-----|-------|----------|-------|
| CE | 0.6667 | 0.6678 | 0.8525 | 0.2716 | 0.2500 | 0.5175 | 100.00% |
| Cumulative | 0.6333 | 0.6159 | 0.3593 | 0.1238 | 0.0915 | 0.5000 | 81.67% |
| SORD | **0.7000** | **0.6995** | **0.8816** | **0.1429** | 0.1987 | 0.5661 | 100.00% |
| EDL | **0.7833** | **0.7791** | **0.8972** | 0.2003 | 0.2793 | **0.6350** | 28.33% |
| EDL+ORCU | 0.6667 | 0.6678 | 0.8525 | 0.2716 | 0.2500 | 0.5175 | 100.00% |

### 1.2 SF (Skeletal Fluorosis) — 24 test samples

| Mode | Acc | F1 | QWK | ECE | U-ECE | AUROC(u) | %Unim |
|------|-----|----|----|-----|-------|----------|-------|
| CE | 0.4583 | 0.2867 | 0.1154 | 0.1839 | 0.0859 | 0.6434 | 66.67% |
| Cumulative | **0.5833** | 0.3614 | **0.3494** | **0.1260** | 0.3752 | **0.6857** | 87.50% |
| SORD | 0.4583 | **0.4294** | **0.4340** | 0.2811 | **0.1095** | 0.6643 | 75.00% |
| EDL | 0.3750 | 0.2944 | 0.1570 | 0.4142 | 0.1903 | 0.4889 | 87.50% |
| EDL+ORCU | 0.5417 | 0.3250 | 0.1064 | 0.2336 | 0.1014 | 0.6643 | 87.50% |

---

## 二、交叉验证结果

### 2.1 DF 5-Fold CV

| Method | Acc (mean ± std) | QWK (mean ± std) | CV vs Test Δ |
|--------|-------------------|-------------------|--------------|
| CE | 0.7233 ± 0.0523 | 0.8765 ± 0.0254 | Test -5.7% |
| Cumulative | 0.6233 ± 0.0672 | 0.5501 ± 0.2543 | Test +1.0% |
| EDL | 0.7367 ± 0.0386 | 0.8832 ± 0.0228 | Test +4.7% |
| EDL+ORCU | **0.7200** ± **0.0125** | **0.8748** ± **0.0109** | Test -5.3% |

### 2.2 SF 5-Fold CV

| Method | Acc (mean ± std) | QWK (mean ± std) | CV vs Test Δ |
|--------|-------------------|-------------------|--------------|
| CE | 0.5500 ± 0.0717 | 0.1806 ± 0.2246 | Test -9.2% |
| Cumulative | 0.5583 ± 0.0624 | 0.3755 ± 0.1259 | Test +2.5% |
| EDL | 0.5750 ± 0.0486 | 0.2401 ± 0.0920 | Test -20.0% |
| EDL+ORCU | **0.5833** ± **0.0456** | **0.2783** ± **0.0568** | Test -4.2% |

---

## 三、关键发现

### 3.1 严重 Bug：DF CE ≡ EDL+ORCU（数据异常）

DF CE 和 DF EDL+ORCU 的 7 个指标**完全相同**到小数点后 4 位:

| 指标 | CE | EDL+ORCU |
|------|-----|----------|
| Acc | 0.6667 | 0.6667 |
| F1 | 0.6678 | 0.6678 |
| QWK | 0.8525 | 0.8525 |
| ECE | 0.2716 | 0.2716 |
| U-ECE | 0.2500 | 0.2500 |
| AUROC(u) | 0.5175 | 0.5175 |
| %Unim | 100% | 100% |

且完全相同于 v2.0 的 EDL+ORCU (Acc=0.6667)。**不可能**是巧合。

**三种可能**:
1. **Notebook mode dispatch bug (~50%)**: train_model() 中 CE 分支未正确触发, 回退到 mode="edl_orcu" 默认值。检查 Cell 9 中 mode 参数的 if/elif 分支。
2. **输出文件覆盖 (~35%)**: CE 先跑完写入 test_metrics.json, 之后 EDL+ORCU 训练失败或用相同 output_dir, 覆盖了文件。
3. **Kernel restart + cell 重跑 (~15%)**: 如果训练过程中 kernel 崩溃, 重启后 CE cell 被跳过, 实际跑的是后续 cell。

**验证方法**: 检查 `/kaggle/working/df_ce/` 和 `/kaggle/working/df_edl_orcu/` 下的 `history.json` — 如果两者的 train_loss 曲线不同但 test_metrics 相同, 则是覆盖问题; 如果完全相同, 则是 dispatch bug。

### 3.2 EDL 单峰率崩塌：95% → 28.3% (v2.0 → v2.1)

这是 v2.1 最严重的性能退化。

| 版本 | DF EDL Acc | DF EDL QWK | DF EDL %Unim | SF EDL Acc |
|------|-----------|-----------|-------------|-----------|
| v2.0 | **0.850** | **0.941** | **95.0%** | **0.667** |
| v2.1 | 0.783 | 0.897 | 28.3% | 0.375 |
| Δ | -6.7pp | -4.4pp | **-66.7pp** | **-29.2pp** |

**分析**:
- DF EDL 从 95% 单峰跌至 28.3%, 意味着大部分预测变成双峰/多峰分布。概率质量分散到多个类别, 而非集中在正确类别附近。
- 单峰崩塌 + Acc 下降 6.7pp: 模型失去了对预测的"聚焦"能力, Dirichlet 证据在多个类别上同时累积。
- SF EDL 完全崩溃 (Acc 38% vs 67%): 在小样本困难任务上, v2.1 的 EDL 基本失效。

**可能原因**:
1. **KL 退火失效**: EDLLoss 的 KL 正则项 (λ_kl=0.1) 在 notebook 内联版本中可能未正常退火, 导致 Dirichlet 先验约束过强或过弱。
2. **build_model 差异**: Notebook 的 `build_model(task, mode)` 调用与命令行版本可能存在细微差异 (如模型初始化方式)。
3. **Seed 效应**: v2.0 seed=42, v2.1 seed=42 理论相同, 但 CUDA 随机性 + notebook 执行顺序可能导致不同的初始化。

**排除 seed 效应的验证**: 多 seed (42, 123, 456) 重跑 DF EDL, 如果三个 seed 都 < 50% Unim, 则是系统性退化。

### 3.3 Cumulative QWK 崩塌：0.92 → 0.36 (v2.0 → v2.1)

DF Cumulative 的 QWK 从 0.9199 跌至 0.3593, 降幅高达 **61%**。

| 版本 | DF Cum Acc | DF Cum QWK | DF Cum CV QWK std |
|------|-----------|-----------|-------------------|
| v2.0 | 0.783 | **0.920** | 0.279 |
| v2.1 | 0.633 | **0.359** | 0.254 |

QWK=0.36 意味着模型几乎在做随机猜测 (相邻类别区分度很低)。v2.0 中 Cumulative 击败了 CE baseline (78.3% vs 76.7%), v2.1 中远不如 CE。

**根因推测**: OrdinalHead 的 CumulativeClassifier (K-1 二元分类器 + monotonic bias) 在 notebook 环境中可能未正确实现。检查:
- `model.ordinal_logits` 是否正确暴露
- CumulativeLoss 是否正确接收 K-1 维 logits
- Monotonic bias constraint (b_0 >= b_1 >= b_2) 是否正确

### 3.4 SORD 崛起为 DF 第二佳方法

| DF 方法 | v2.0 Acc | v2.1 Acc | 排名变化 |
|---------|---------|---------|---------|
| EDL | 0.850 | 0.783 | 1 → 1 (稳) |
| SORD | 0.733 | **0.700** | 4 → **2** |
| CE | 0.767 | 0.667 | 3 → 3 |
| Cumulative | 0.783 | 0.633 | 2 → 4 |

SORD 的 Acc 从 73.3% 仅微降至 70.0% (-3.3pp), 而其他方法大幅下降。SORD 成为 DF 上**最稳定**的方法 (除 EDL 外)。

**SORD 的 100% 单峰 + 合理精度** 使其成为临床可解释性的最佳选择。如果后续验证确认 EDL 的单峰问题无法解决, SORD 可以作为论文的推荐方法。

### 3.5 EDL+ORCU CV 稳定性确认 — 跨版本最可靠发现

| 版本 | DF CV Acc ± std | DF CV QWK ± std | SF CV Acc ± std |
|------|----------------|----------------|----------------|
| v2.0 | 0.7367 ± **0.0194** | 0.8807 ± **0.0113** | 0.5917 ± 0.0928 |
| v2.1 | 0.7200 ± **0.0125** | 0.8748 ± **0.0109** | 0.5833 ± 0.0456 |

**EDL+ORCU 的 CV std 在所有方法中始终最小** (1-2% vs 其他方法的 4-7%), 跨两个版本均如此。这是唯一跨版本可靠的正面发现:
- EDL+ORCU 的三阶段训练 (CE warmup → EDL → Joint) 提供了**训练稳定性**优势
- 即使 lambda_reg 从 0.05 降至 0.01, 稳定性特征不变
- 这种稳定性在小样本医学数据上非常有价值

但 DF test Acc 66.7% 显著低于 CV 均值 72.0% (-5.3pp), 这个 **CV-Test gap** 仍需解释。

### 3.6 SF 上 EDL 完全崩溃

SF EDL Acc 从 v2.0 的 66.7% 跌至 v2.1 的 37.5% (-29.2pp)。37.5% 仅略高于 random (25%), 模型基本没学到东西。

**SF Cumulative 稳坐 SF 第一**: 两次实验均最优 (v2.0 58.3%, v2.1 58.3%), QWK 也是最佳 (0.40, 0.35)。Coral 的 K-1 有序结构确实适合 SF 小样本场景。

---

## 四、v2.0 vs v2.1 跨版本对比

### 4.1 一致发现 (跨版本可靠)

| 发现 | 证据 |
|------|------|
| EDL 是 DF 最佳方法 | v2.0: 85.0%, v2.1: 78.3% (均为第一) |
| Cumulative 是 SF 最佳方法 | v2.0: 58.3%, v2.1: 58.3% (均为第一) |
| EDL+ORCU CV 稳定性最优 | 两版本 std 均 ~1-2% |
| EDL 不确定性不可靠 | 两版本 AUROC(u) 均 < 0.64 |
| SORD = 100% 单峰 | 两版本 DF 均为 100% |

### 4.2 不一致发现 (需多 seed 验证)

| 发现 | v2.0 | v2.1 | 说明 |
|------|------|------|------|
| EDL 单峰率 | 95.0% | 28.3% | 巨大差异, 核心矛盾 |
| Cumulative QWK | 0.920 | 0.359 | 可能 v2.0 是 outlier |
| CE baseline | 76.7% | 66.7% | Seed 差异或代码差异 |
| SF EDL | 66.7% | 37.5% | 系统性退化 |
| SORD DF 排名 | 第 4 | 第 2 | SORD 意外受益 |

### 4.3 版本间差异的可能解释

1. **Notebook 代码差异 (~40%)**: v2.1 将 train.py 重写为 notebook 内联函数, 可能存在未发现的 bug (如 CE=EDL+ORCU)
2. **Seed + CUDA 非确定性 (~30%)**: 即使 seed 相同, notebook 执行顺序可能产生不同的 CUDA 随机状态
3. **训练被中断 (~20%)**: Kaggle 5-6h 超时可能截断部分训练
4. **数据加载差异 (~10%)**: Notebook 和 CLI 可能使用不同的 num_workers/DataLoader 设置

---

## 五、综合评估

### 5.1 方法评分 (1-5, 跨版本平均)

| 维度 | CE | Cumulative | SORD | EDL | EDL+ORCU |
|------|-----|-----------|------|-----|----------|
| DF 分类精度 | 3 | 2 | 3 | **5** | 2 |
| SF 分类精度 | 2 | **5** | 3 | 1 | 3 |
| 训练稳定性 | 4 | 2 | **5** | 3 | **5** |
| 单峰率 | 3 | 4 | **5** | 2 | **5** |
| 不确定性质量 | 3 | 3 | 3 | 2 | 3 |
| 跨版本一致性 | 3 | 1 | **4** | 1 | **5** |

### 5.2 场景推荐 (v2.1 更新)

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 追求 DF 最高精度 | **EDL** | 两版本均第一, 但需修复单峰问题 |
| 临床可解释性 (100% 单峰) | **SORD** | 跨版本稳定 + 100% 单峰, Acc 可接受 |
| SF 小样本 | **Cumulative** | 跨版本最优 SF, Coral 架构防远距离误分 |
| 训练稳定性 (低方差) | **EDL+ORCU** | 两版本 CV std ~1%, 但 Acc 不理想 |
| 可信不确定性 | **CE + Ensemble** | EDL 不确定性在两版本均不可靠 |

---

## 六、待验证和后续实验

### 6.1 紧急 (必须解决)

1. **验证 DF CE=EDL+ORCU bug**: 检查 Kaggle 输出文件的 history.json, 确认是 dispatch bug 还是文件覆盖
2. **多 seed 验证 (3 seeds)**: DF EDL, DF Cumulative, DF SORD — 确认 v2.1 结果是否可复现
3. **对比 notebook vs CLI**: 同一 seed 下两种运行方式的结果是否一致

### 6.2 如需重新训练

4. **修复 CE wrapper bug**: `L_ce: 0.0` 应改为 `loss.item()` (仅影响日志)
5. **检查 build_model mode dispatch**: 确保 mode="ce" 正确传递到模型构建
6. **添加输出验证**: 在 save test_metrics 时打印 mode 名称, 防止覆盖

### 6.3 论文策略

- **最安全的发现**: EDL+ORCU 的训练稳定性 (低 CV std) — 跨版本最可靠
- **需要多 seed 确认**: EDL 单峰率 (95% vs 28% 矛盾)
- **稳健的 baseline**: SORD 在 v2.1 意外成为综合最优 (Acc 70% + 100% 单峰)
- **SF 的可靠选择**: Cumulative 连续两版本最优

---

## 七、Lambda Sweep 和 Temperature Calibration

v2.1 notebook 包含 lambda sweep (Section 8) 和 temperature calibration (Section 7) 代码, 但 FINAL RESULTS 输出中未包含这两部分结果。可能:
- Cell 未执行 (Kaggle 超时或手动跳过)
- 结果保存在独立 JSON 文件中 (lambda_sweep_v2.json, temperature_sweep.json)

如果这两部分已运行, 需从 Kaggle 下载对应的 JSON 文件进行分析。

---

## 八、结论

1. **EDL 仍是 DF 最强方法**, 但单峰率从 95% 崩塌至 28% 是严重警示 — 需多 seed 验证
2. **DF CE=EDL+ORCU 数据异常** 需立即调查, 可能影响 CE/Cumulative/SORD 结果的有效性
3. **SORD 意外成为 v2.1 综合最优** — 100% 单峰 + 70% Acc, 是临床部署的安全选择
4. **Cumulative 是 SF 唯一可靠方法** — 跨版本稳定在 58.3% Acc, QWK 0.35-0.40
5. **EDL+ORCU 训练稳定性跨版本可靠** — 这个发现可以写入论文
6. **SF EDL 完全崩溃 (38%)** — v2.0 的 67% 可能是 outlier, 也可能是 v2.1 的 bug
7. **v2.1 整体质量低于 v2.0** — 多个方法退化, 建议修复 bug 后重跑 v2.2
