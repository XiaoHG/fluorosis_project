---
date: 2026-05-15
author: InnoAgent
output_to: "03_Innovation/README.md"
status: draft
---

# 03_Innovation — 创新点设计

## 目录

| 文件 | 内容 | 状态 |
|------|------|:----:|
| `01_innovation_blueprint.md` | 核心创新蓝图: EDL + ORCU + Multi-rater | draft |
| `02_idea_proposals/` | 备选创新方案 | 待填充 |

## 核心创新方向

**EDL + ORCU + Multi-Rater Modeling for Fluorosis Diagnosis**

- **EDL**: Dirichlet evidence 替代 softmax，输出不确定性
- **ORCU**: 有序回归 + 校准 + 单峰性约束替代 CE loss
- **Multi-Rater**: 5 位医生标注分布替代 majority-vote one-hot

## 决策节点

| 编号 | 时间 | 内容 |
|------|------|------|
| D1 | 6.2 前 | 从创新蓝图中确认主攻方向 |
| D2 | 6.16 前 | 确认最终模型架构 |

## 下一步

→ `04_Model_Design/` — 将创新蓝图转化为具体的模型架构设计和 PyTorch 伪代码
