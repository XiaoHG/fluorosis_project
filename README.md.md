---
date: 2026-05-15
author: Human
status: final
---

# 深度学习在氟中毒诊断中的应用 · 论文协作项目

## 项目简介

本项目旨在产出一篇可投稿至 **Medical Image Analysis** (MedIA, Elsevier) 的原创研究论文，研究方向为深度学习在氟斑牙（Dental Fluorosis）和氟骨症（Skeletal Fluorosis）诊断中的应用。

采用 **7 个专职 Claude Agent + 人类项目经理** 的协作模式，总周期 **12 周**（2026.05.13 – 2026.08.04）。

## 数据概览

| 数据集 | 模态 | 数量 | 分级 |
|--------|------|------|------|
| 氟斑牙 (Dental Fluorosis) | 自然光口腔照片 (512×256) | 200 张，4 类各 50 | 0=正常, 1=轻度, 2=中度, 3=重度 |
| 氟骨症 (Skeletal Fluorosis) | X 光平片 (512×1024) | 80 张 + 59 张 mask | 二分类 (0/1) + 四分级 (0–3) |

## 目录结构与工作流

```
00_Admin/           项目计划、期刊要求、7 个 Agent System Prompt
01_Knowledge_Base/  领域知识底座（临床背景、诊断标准、数据描述、术语表）
02_Literature/      ← LitAgent     文献检索、精读、对比、缺口分析
03_Innovation/      ← InnoAgent    创新点提案、技术路线图
04_Model_Design/    ← ModAgent     模型架构设计、损失函数、训练配置
05_Exp_Design/      ← ExpAgent     数据划分、增强、指标、基线、消融方案
06_Implementation/  ← ImplAgent    代码框架、训练脚本、实验配置
07_Visualization/   ← VizAgent     绘图脚本、图表、图注
08_Manuscript/      ← WriteAgent   论文 LaTeX 稿件及多轮修订
09_Review/          多轮审核报告
10_Submission/      投稿包
11_Decision_Logs/   关键决策记录
```

Agent 按阶段串行启动：LitAgent → InnoAgent → ModAgent → ExpDesignAgent → ImplAgent → VizAgent → WriteAgent。

## 阶段时间表

| 阶段 | 周期 | 内容 | 里程碑 |
|------|------|------|--------|
| 0 | 第 1 周 (5.13–5.19) | 环境与知识准备 | M0: 知识底座就绪 |
| 1 | 第 2 周 (5.20–5.26) | 文献深挖 | M1: 文献分析完成 |
| 2 | 第 3–4 周 (5.27–6.09) | 创新点凝练 | M2: 创新点锁定 |
| 3 | 第 5–6 周 (6.10–6.23) | 模型框架设计 | M3: 架构冻结 |
| 4 | 第 7 周 (6.24–6.30) | 实验设计 | M4: 实验方案锁定 |
| 5 | 第 8–10 周 (7.01–7.21) | 实验实施 | M5: 实验数据产出 |
| 6 | 第 11 周前半 (7.22–7.25) | 图表制作 | M6: 图表就绪 |
| 7 | 第 11 周后半–12 周 (7.26–8.04) | 论文撰写与审核 | M7: 投稿包就绪 |

## 人类决策节点

| 编号 | 时间 | 决策内容 |
|------|------|----------|
| D1 | 6.2 前 | 从 3 个创新点提案中选择主攻方向 |
| D2 | 6.16 前 | 从 2 个备选架构中确认最终模型 |
| D3 | 6.30 | 审核并批准实验方案 |
| D4 | 7.23 | 审阅图表草图 |
| D5 | 8.04 | 终审全文，签署定稿 |

## 协作规则

- 所有 Agent 输出需包含 YAML 元数据头部，注明 `input_from` 和 `output_to`
- 每完成一个阶段进行 Git commit 并打 tag
- 关键决策记录存入 `11_Decision_Logs/`
- 所有术语以 `01_Knowledge_Base/glossary.md` 为准

## 快速链接

- [项目计划书](00_Admin/project_plan.md.md)
- [期刊投稿要求](00_Admin/journal_requirements.md.md)
- [氟中毒临床背景](01_Knowledge_Base/fluorosis_intro.md.md)
- [诊断标准](01_Knowledge_Base/diagnostic_criteria.md.md)
- [数据集描述](01_Knowledge_Base/data_description.md.md)
- [统一术语表](01_Knowledge_Base/glossary.md.md)
