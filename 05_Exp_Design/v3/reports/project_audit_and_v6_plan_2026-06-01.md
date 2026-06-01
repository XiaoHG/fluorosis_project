# 氟中毒DL诊断 — 项目全面审计与V6实验规划报告

**日期:** 2026-06-01  
**审计范围:** 全项目（版本管理、代码实现、论文稿件、文献追踪、项目配置）  
**状态:** 审计完成，37个问题已发现，14项修复已完成

---

## 一、审计总览

四个维度审计发现 **37 个问题**，其中 7 个严重、12 个高优先级、18 个中等。**14 项已修复**，23 项待处理。

| 维度 | 审计范围 | 问题数 |
|------|---------|--------|
| 版本管理与实验数据 | 05_Exp_Design/ v1/v2/v3 | 9 |
| 代码实现 | 06_Implementation/ (17文件, ~2200行) | 18 |
| 论文稿件 | 08_Manuscript/ (7章节, 30引用) | 7 |
| 项目配置与文献 | 根目录, 02_Literature/, 00_Admin/ | 3 |

---

## 二、已完成的 14 项修复

### 严重修复 (4项)

**1. CLAUDE.md 数据声称错误** ✅
- "QWK CV 0.36%" 实际不存在 → 修正为真实值 "CV 范围 1.89%-3.00%"
- EDL 最佳 QWK 0.936 → 修正为 0.938
- EDL+ORCU "最稳定" 声称不实 → 修正为 "EDL 最稳定 (CV 1.89%)"

**2. ORCU 损失函数：hinge → log-barrier** ✅
- 设计文档要求 log-barrier，代码实际使用 ReLU hinge
- `t` 参数在 hinge 实现中完全未使用
- 新增 `log_barrier_penalty()` 函数，实现平滑 log-barrier + 线性尾部

**3. cross_validate.py 缺少 LR scheduler** ✅
- 已添加 LinearLR warmup + CosineAnnealingLR + 梯度裁剪
- CV 结果现在与主训练可比

**4. 论文图片符号链接损坏** ✅
- 已创建 `05_Exp_Design/figures/` 并复制所有 20 张图片

### 高优先级修复 (6项)

**5-10: train.py 全面升级** ✅
- EarlyStopping (patience-basED, 内存中最佳模型恢复)
- CUDA determinism (set_seed: cudnn.deterministic + benchmark)
- 梯度裁剪 (clip_grad_norm max_norm=1.0)
- AMP 混合精度 (--use_amp)
- YAML 配置加载 (--config)
- JSON NaN 修复

### 中等修复 (4项)

**11-14:** .md.md 文件重命名 ✅ | Review Agent 系统提示创建 ✅ | v3/ 脚手架 (plan.md + README.md) ✅ | PROJECT_STATUS 日期更新 ✅

---

## 三、待修复的 23 项问题

### 代码 (8项)

| # | 问题 | 优先级 |
|---|------|--------|
| 1 | SF 缺少 5-rater 软标签支持 | 🔴 |
| 2 | lambda_kl 4个来源4个不同值 (0.1, 0.07, 0.02, 0.1) | 🔴 |
| 3 | SF 类别权重未实现 (Normal=21, Severe=12) | 🟠 |
| 4 | SF mask 未使用 (59 masks可用) | 🟠 |
| 5 | A2/A6 消融实验未实现 | 🟠 |
| 6 | Mamba 骨干缺失 (已知限制) | 🟡 |
| 7 | MixUp 增强未实现 | 🟡 |
| 8 | edl_loss.py mean_evidence 命名误导 | 🟡 |

### 论文稿件 (7项)

| # | 问题 | 优先级 |
|---|------|--------|
| 9 | 表格跨版本不一致 — 表1 CE seed=42 为 81.67%，表3 为 61.7% | 🔴 |
| 10 | CV 数据矛盾 — 稿件 73.00% vs CLAUDE.md 79.33% | 🔴 |
| 11 | 8 个未引用标签 (fig:f6/f9/f11/f12/f13, tab:cv/perclass, sec:multiseed) | 🔴 |
| 12 | 资助编号 XXXXXX 占位符 | 🟠 |
| 13 | 作者名 "xiaohg" 需替换为正式姓名 | 🟠 |
| 14 | main.bbl 过时 (含已删除的 HiFuse) | 🟠 |
| 15 | 死引用 quadri2016sf | 🟡 |

### 其他 (8项)

| # | 问题 | 优先级 |
|---|------|--------|
| 16 | v1 缺少 plan.md | 🟠 |
| 17 | v2/figures/ 完全为空 | 🟠 |
| 18 | SF 结果全部缺失 (v2/results/sf/) | 🟠 |
| 19 | FusionDentNet 不一致 | 🟠 |
| 20 | HiFuse 残留 | 🟡 |
| 21 | 07_Visualization/ 大部分为空 | 🟡 |
| 22 | manuscript_overleaf.zip (6.4MB) git跟踪 | 🟡 |
| 23 | CLAUDE.md agent 数量与实际不符 | 🟡 |

---

## 四、V6 实验方案 (v3/)

### 核心目标
超越 MLTrMR (80.19%), ECE < 0.15, 完成 SF 全管线

### Phase 1: 代码基础设施 (2-3天) — 7 项任务, 5 项已完成

### Phase 2: DF V6 实验 (3-4天)

| Mode | KL | ORCU | Folds | Seeds |
|------|-----|------|-------|-------|
| CE | - | - | 10 | 42,123,456 |
| Cumulative | - | - | 10 | 42,123,456 |
| SORD | - | - | 10 | 42,123,456 |
| **EDL** | **0.07** | - | **10** | **42,123,456,789,1024** |
| **EDL+ORCU** | **0.07** | **0.10** (log-barrier) | **10** | **42,123,456,789,1024** |

### Phase 3: SF V2 实验 (2-3天)

| Mode | Labels | Priority |
|------|--------|----------|
| CE/Cumulative/EDL/EDL+ORCU | Hard (majority vote) | Baseline |
| **EDL + EDL+ORCU** | **Soft (5-rater)** ⭐ | **创新点** |

### 成功标准
- DF V6 Acc > 80% | QWK > 0.91 | ECE < 0.15
- Log-barrier 单峰性 > hinge
- SF soft-label EDL > hard-label EDL

---

## 五、项目快照

| 指标 | 值 |
|------|-----|
| 最佳结果 | EDL CV Acc 79.33% ± 3.74 (V4) |
| 竞争者 | MLTrMR 80.19% (556M), LD2Net 80.00% (3.3M) |
| 代码量 | 17 Python 文件, ~2200 行 |
| 论文 | 7 章节, 30 引用, PDF 已编译 |
| Git | 60 commits, 干净工作区 |

---

*报告由 Claude Code 生成于 2026-06-01 全面项目审计。详细修复日志见 git diff。*
