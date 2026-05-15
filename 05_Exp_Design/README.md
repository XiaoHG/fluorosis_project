---
date: 2026-05-15
author: ExpDesignAgent
output_to: "05_Exp_Design/README.md"
status: draft
---

# 05_Exp_Design — 实验设计

## 实验概览

- **Baselines**: 7 (3 DF + 4 SF, 含人类基线)
- **Our methods**: 6 (3 DF + 3 SF)
- **Ablations**: 6
- **Uncertainty analysis**: 5
- **Total experiments**: ~24

## 关键设计决策

1. DF 120/20/60 split — 与 MLTrMR 的 140/60 test 对齐
2. SF 48/8/24 split — 与 Mwinc-Mamba 的 56/24 test 对齐
3. 复现策略: 不完整复现竞品架构，以相同 backbone + CE 为 proxy baseline
4. U5 是关键创新分析: 医生一致性 vs 模型不确定性相关性

## 文件

| 文件 | 内容 |
|------|------|
| `01_experiment_plan.md` | 完整实验矩阵 |

## 下一步

→ `06_Implementation/` — 训练脚本、数据加载器、评估工具
