---
date: 2026-05-15
author: ModAgent
output_to: "04_Model_Design/README.md"
status: draft
---

# 04_Model_Design — 模型架构设计

## 核心设计

**共享 logits 的 EDL+ORCU 架构**

- Backbone (ViT-B / Mwinc-SSM) → Evidence Head (FC) → 同一个 logits z 分两路
- EDL 路径: z → Softplus → α → Dirichlet → L_EDL
- ORCU 路径: z → SORD + logit-diff regularization → L_ORCU
- L_total = L_EDL + λ·L_ORCU

## 文件

| 文件 | 内容 |
|------|------|
| `01_model_architecture.md` | 完整架构: Evidence Head, Loss 伪代码, 训练配置, 评估指标 |

## 下一步

→ `05_Exp_Design/` — 数据划分、增强策略、baseline 复现、消融实验设计
