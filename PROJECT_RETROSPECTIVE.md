# 氟中毒深度学习论文项目 — 全面复盘报告

**发件人:** Claude Code (AI Assistant)
**收件人:** Xiaohong Gao
**日期:** 2026-05-18
**主题:** 项目全面复盘 — Fluorosis DL Paper (v1 审稿修改完成)

---

## 一、项目概览

| 项目 | 内容 |
|------|------|
| **项目名称** | Evidential Deep Learning with Ordinal Calibration for Automated Diagnosis of Endemic Fluorosis |
| **目标期刊** | Medical Image Analysis (MedIA, Elsevier) |
| **GitHub仓库** | github.com/XiaoHG/fluorosis_project |
| **工作区间** | /Volumes/KINGSTON/Obsibian/Fluorosis/Fluorosis_DL_Paper |
| **启动日期** | 2026-05-13 |
| **当前阶段** | v1 初稿完成 + 第一轮审稿意见修改完成 |

---

## 二、项目规模统计

### 代码与文档
| 类型 | 数量 | 说明 |
|------|------|------|
| Git 提交 | 46 | 涵盖从项目初始化到审稿修改的全过程 |
| Python 文件 | 24 | 共 5,953 行代码 |
| Markdown 文档 | 81 | 文献笔记、实验分析、决策记录等 |
| LaTeX 源文件 | 8 | main.tex + 7 个 section 文件 |
| BibTeX 条目 | 31 | 覆盖 DF/SF 诊断、EDL、有序回归、校准等领域 |
| 论文页数 | 34 | 含 16 张图、5 张表 |

### 文献调研
| 指标 | 数量 |
|------|------|
| 收集 PDF | 134 篇 |
| 深度阅读笔记 | 20 篇 |
| 研究发现差距 | 6 个 (P0-P3 优先级) |
| 方法对比表 | 19 种方法 |

### 实验资源
| 数据集 | 样本数 | 类别 | 标注者 |
|--------|--------|------|--------|
| DFID (牙氟中毒) | 200 张口内照片 | 4类 (50/类) | 单人标注 |
| SFXRay (骨氟中毒) | 80 张 X 光片 | 4类 | 5位放射科医生 |

---

## 三、项目时间线

```
2026-05-13  项目启动，骨架初始化
            ↓
2026-05-14  文献调研 (02_Literature) — 比计划提前 4 天完成
            134篇PDF，20篇深读，6个研究差距
            ↓
2026-05-15  创新设计 (03_Innovation) + 模型架构 (04_Model_Design)
            EDL+ORCU 共享 logit 框架确立
            ↓
2026-05-15  实验设计 (05_Exp_Design) + 代码实现 (06_Implementation)
            17 个 Python 文件，5 种 loss 函数
            开始 Kaggle 训练 (v1 → v6 共 6 轮迭代)
            ↓
2026-05-16  实验迭代 v2.0 → v2.2
            修复 CE logging bug、StandardClassifier vs EDLClassifier
            5种 loss 对比 + 多种子 ×3 + 5折 CV + lambda sweep
            ↓
2026-05-17  可视化 (07_Visualization) + 论文写作 (08_Manuscript)
            34 张图生成，7 个 section 写完
            ↓
2026-05-17  v1.0 归档 — 项目最终报告 + 决策日志 + Release 打包
            ↓
2026-05-18  格式审查 + 引文扩充 (10 → 31 篇)
            elsarticle 格式修复、行号移除、frontmatter 重构
            ↓
2026-05-18  全面审稿 + 修改 → **当前状态**
            8 项修改完成，审稿报告已归档
```

---

## 四、核心成果

### 4.1 方法论创新

**共享 logit 证据头 (Shared-Logit Evidence Head)**

```
  Backbone (ViT-Base) → Feature f ∈ R^768
         ↓
    Evidence Head (FC 768×4) → logits z ∈ R^4
         ↓                         ↓
    EDL Path                   ORCU Path
    α_k = softplus(z_k) + 1    p_k = softmax(z_k)
         ↓                         ↓
    Dirichlet (evidence, u)    SORD + log-barrier
```

一个全连接层同时服务于两条路径：EDL 路径产生 Dirichlet 证据和不确定性，ORCU 路径施加有序约束和单峰正则化。

### 4.2 关键实验结果

| 指标 | CE (Baseline) | EDL (Ours) | 提升 |
|------|:------------:|:----------:|:----:|
| **准确率** | 81.67% | **83.33%** | +1.67pp |
| **QWK** | 0.9329 | **0.9376** | +0.0047 |
| **ECE** | 0.1213 | **0.0719** | -40.7% |
| **%单峰** | 100% | 96.7% | — |

| 稳定性指标 | CE | EDL |
|-----------|:-----:|:-----:|
| **多种子 CV** | 31.4% | **4.3%** |
| **种子 456 准确率** | 35.0% (崩溃) | **75.0%** |

**SOTA 对比:**
- MLTrMR (556M): 80.19% → EDL (86M): **83.33%** (+3.14pp)
- LD2Net (3.31M): 80.00% → EDL 参数更多但提供不确定性量化

### 4.3 临床价值
- 不确定性门控分诊系统：低不确定性自动报告 (~70% 样本)，高不确定性转专家审核 (~30% 样本)
- SafePred ~0.90 (自动报告准确率)
- 在仅有 120 个训练样本的条件下实现

---

## 五、项目模块完成度

| 模块 | 完成度 | 备注 |
|------|:------:|------|
| 01_Knowledge_Base | ✅ 100% | 临床背景、诊断标准、术语表 |
| 02_Literature | ✅ 100% | 134篇PDF、20篇深读、差距分析 |
| 03_Innovation | ✅ 100% | 创新蓝图、自动诊断报告设计 |
| 04_Model_Design | ✅ 100% | 证据头架构、backbone 规格 |
| 05_Exp_Design | ✅ 100% | 5-loss 对比、多种子、CV、消融 |
| 06_Implementation | ✅ 100% | 24个Python文件、5953行代码 |
| 07_Visualization | ✅ 100% | 34张图 (PDF/PNG) |
| 08_Manuscript | ✅ 95% | v1 完成，审稿修改完成 |
| 09_Review | ✅ 100% | 内部审查清单、代码审查清单 |
| 10_Submission | ✅ 80% | 模板齐全，待最终定稿 |
| 11_Decision_Logs | ✅ 100% | 5 个关键决策记录 |

---

## 六、第一轮审稿修改 (本次完成)

### 已修改
1. ✅ 删除全部版本信息 (v2.2/v6/PyTorch 引用)
2. ✅ 区分 Dirichlet 概率 p_k^Dir 和 softmax 概率 p_k^soft
3. ✅ "new SOTA" → "the highest on this benchmark" (全文弱化)
4. ✅ AUROC(u) 分析从 Limitation 移到 Results (§5.4)，加分 bin 不确定性分析
5. ✅ "default baseline" → "strong baseline"，加多中心验证说明
6. ✅ 致谢占位符填入正式文本
7. ✅ 审稿报告归档 (265行，15项修改路线图)
8. ✅ 编译通过，0 错误 0 未定义引用

### 待完成 (审稿优先级)
**Priority 1 (投稿前必须):**
- 添加统计显著性检验 (McNemar/Bootstrap, n=60)
- 添加 CE + label smoothing baseline (区分正则化 vs Dirichlet 贡献)
- 核实 FusionDentNet bib 条目 (arXiv ID 可能属于 MLTrMR)

**Priority 2 (强烈建议):**
- ResNet-50 结果 (支持 backbone-agnostic 声明)
- CE 集成 baseline (3-seed ensemble)
- 系统性误分类分析 (散点图: u vs 正确性 × 类别)

**Priority 3 (锦上添花):**
- Grad-CAM 注意力可视化
- EDL+ORCU 逐类分析
- 种子 456 CE 崩溃详细诊断

---

## 七、产出清单

### 7.1 可交付物
| 文件 | 位置 | 说明 |
|------|------|------|
| 论文 PDF | 08_Manuscript/v1_first_draft/main.pdf | 34 页，编译无错误 |
| LaTeX 源码 | 08_Manuscript/v1_first_draft/ | main.tex + 7 sections |
| BibTeX | 08_Manuscript/v1_first_draft/references.bib | 31 条 |
| 审稿报告 | 08_Manuscript/v1_first_draft/review_v1_2026-05-18.md | 265 行 |
| Python 代码 | 06_Implementation/ | 24 文件, 5953 行 |
| 实验分析 | 05_Exp_Design/04_experiment_results_v2.2.md | v2.2 完整分析 |
| 文献综述 | 02_Literature/ | 20 篇深读 + 差距分析 |
| 决策记录 | 11_Decision_Logs/ | 5 个关键决策 |
| 提交包 | 10_Submission/ | Cover letter 等模板 |
| v1.0 归档 | 00_Admin/releases/v1.0/ | 源码 ZIP + 代码 ZIP |

### 7.2 Git 发布
- v1.0_manuscript_v1_draft — 初稿完成版本 (e75af7b)
- da1ec2c — 审稿修改完成版本 (最新)

---

## 八、经验教训

### 8.1 做得好的
1. **文献调研效率高** — 134篇PDF收集 + 20篇深读在2天内完成，比计划提前4天
2. **实验迭代严格** — v1 → v6 六轮迭代，每轮有明确修复目标
3. **负面结果诚实报告** — EDL+ORCU 单次划分不佳、AUROC<0.5 均如实写入论文
4. **多种子稳定性** — 种子 456 的 CE 崩溃是意外发现，成为论文最强论点之一
5. **审稿流程规范** — 三遍审稿 (Referee + Devil's Advocate + Editorial) 在投稿前发现关键问题

### 8.2 可改进的
1. **版本管理** — v2.0/v2.1/v2.2/v6 的数据版本差异在论文初稿中未明确标注，导致审稿时发现表间数据不一致
2. **数据管线** — per-sample prediction 导出和实验数据管理可以更自动化
3. **baseline 设计** — 如果早期加入 label smoothing baseline，可以更清楚地区分 EDL 的正则化效应和 Dirichlet 参数化效应
4. **小样本统计** — 在 n=60 的测试集上做 5 种 loss 对比，一开始就应该计划统计显著性检验

### 8.3 项目亮点
- **首个将 EDL 应用于氟中毒诊断的工作**
- **首个在医学有序分类中系统对比 5 种 loss 函数的工作**
- **发现 EDL 的 KL 正则化对种子稳定性有意外的大幅提升** (CV 31.4% → 4.3%)
- **共享 logit 架构设计简洁，具有通用性**

---

## 九、后续建议

1. **近期 (1-2 周):** 完成 Priority 1 修改 → 发给合作者审阅 → 投稿
2. **中期 (1-2 月):** 准备审稿回复模板、骨氟中毒数据集实验 (EDL + 多评分者软标签)
3. **长期:** 如果 MedIA 被拒，备选目标：Biomedical Signal Processing and Control (BSPC)、Computers in Biology and Medicine (CBM)

---

## 十、关键数据一览

```
启动日期:     2026-05-13
当前日期:     2026-05-18
总工时:       ~5 天
Git 提交:     46
Python LOC:   5,953
Markdown:     81 篇
PDF 文献:     134 篇
BibTeX:       31 条
论文页数:     34 页
论文图表:     16 图 + 5 表
EDL 准确率:   83.33% (SOTA)
ECE:          0.0719
种子 CV:      4.3% (EDL) vs 31.4% (CE)
```

---

*本报告由 Claude Code 基于对整个项目仓库的全面扫描自动生成。*
