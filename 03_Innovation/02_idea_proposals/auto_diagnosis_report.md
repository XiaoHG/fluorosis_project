---
date: 2026-05-15
author: InnoAgent
input_from:
  - "04_Model_Design/01_model_architecture.md"
  - "05_Exp_Design/01_experiment_plan.md"
output_to: "03_Innovation/02_idea_proposals/auto_diagnosis_report.md"
status: draft
---

# 自动诊断报告：基于 EDL 输出参数

## 动机

EDL 模型输出不仅是预测标签，还包括 evidence (α)、uncertainty (u)、预测概率分布 (p) 等丰富信息。这些参数天然支撑"自动诊断报告"——将模型输出转化为临床可读的诊断描述，附带置信度和不确定性标注。

这不是核心创新，而是 EDL 方法论的 **自然延伸应用**，可作为论文的 Clinical Application 章节或 Discussion 亮点。

## 一、可用参数清单

### 1.1 直接来自 EDL 模型

| 参数 | 符号 | 形状 | 含义 |
|------|------|------|------|
| Dirichlet 浓度 | α = [α₀...α₃] | (4,) | 每个等级的证据浓度 |
| Evidence | e = [e₀...e₃] | (4,) | e_k = α_k - 1，支持每个等级的证据量 |
| 预测概率 | p = [p₀...p₃] | (4,) | p_k = α_k / Σα_j，期望类别概率 |
| 预测等级 | ŷ | scalar | ŷ = argmax(p)，Dean/SF 0-3 |
| 总不确定性 | u | scalar | u = 4 / Σα_k，∈ (0, 1) |
| 信念质量 (belief) | b = [b₀...b₃] | (4,) | b_k = e_k / Σe_j，归一化 evidence |
| Dirichlet 众数 | mode_k = (α_k - 1) / (Σα_j - 4) | (4,) | Dirichlet 分布的顶点位置 |
| 证据总量 | S = Σα_k | scalar | S 越大 → 越确定 |

### 1.2 间接计算参数

| 参数 | 推导 | 含义 |
|------|------|------|
| 预测熵 | H = -Σ p_k log p_k | 概率分布的分散度 |
| 等级置信区间 | [ŷ - δ, ŷ + δ] 满足 P ≥ τ | 预测的不确定性区间 |
| 单峰性 | is_unimodal(p) | ORCU 约束下应始终为 True |
| 相邻等级混淆度 | confusion = 1 - (p_ŷ - p_neighbor) | 预测在两个等级间的模糊性 |

### 1.3 任务特定参数

**DF (氟斑牙）**:
| 参数 | 来源 | 含义 |
|------|------|------|
| 白垩色比例估计 | Segmentation (如有) | 牙面白斑占比 |
| 着色程度 | ColorJitter 统计 | 棕褐色斑块的严重度 |
| 缺损检测 | Segmentation mask | 釉质缺损区域 |

**SF (氟骨症）**:
| 参数 | 来源 | 含义 |
|------|------|------|
| 身体部位 | 元数据 | 前臂/小腿/骨盆 |
| 钙化面积 | Mask (如有) | 骨间膜/韧带钙化面积 |
| 骨密度评分 | 像素强度统计 | ROI 内平均骨密度 |
| 医生间一致性 | 5 医生标注方差 | 该样本的"客观难度" |

### 1.4 与现有 SOTA 的差距参数

| 参数 | 计算方式 | 含义 |
|------|---------|------|
| SOTA 预测差异 | ŷ_ours - ŷ_sota | 与竞品模型的诊断差异 |
| 诊断修正标志 | ŷ_ours ≠ ŷ_sota AND u_ours < u_sota | 我们的模型更确定的修正 |
| 不确定性降低 | u_sota - u_ours | 相对 SOTA 的不确定性改善 |

## 二、报告模板设计

### 2.1 DF 诊断报告模板

```
============================================
      氟斑牙自动诊断报告
============================================
患者 ID:    PT-{id}
检查日期:   {date}
图像类型:   口腔自然光照片（前牙区）
模型版本:   EDL-ORCU + ViT-B v1.0

--- 诊断结果 ---
Dean 分级:   {grade_name} (Grade {ŷ})
             0=正常  1=轻度  2=中度  3=重度

置信度:      {p_ŷ:.1%}
不确定性:     {u:.2%}  [{uncertainty_level}]

--- 分级概率 ---
Normal:       {p₀:.1%}  {'█' * bar(p₀)}
Mild:         {p₁:.1%}  {'█' * bar(p₁)}
Moderate:     {p₂:.1%}  {'█' * bar(p₂)}
Severe:       {p₃:.1%}  {'█' * bar(p₃)}

--- 证据分布 (Evidence) ---
Evidence(N)={e₀:.1f}  E(Mi)={e₁:.1f}  E(Mo)={e₂:.1f}  E(Se)={e₃:.1f}
总证据量 S={S:.1f} (越高越确定)

--- 临床建议 ---
{generate_recommendation(ŷ, u, S, p)}
============================================
报告生成时间: {timestamp}
```

### 2.2 recommendation 逻辑

```python
def generate_recommendation(y_hat, u, S, p):
    if u > 0.60:
        return "高不确定性 — 建议由资深牙科医生复核。"
    if u > 0.30:
        return "中等不确定性 — 建议结合临床检查确认。"

    if y_hat == 0:
        return "未检测到氟斑牙病变。建议定期口腔检查。"
    elif y_hat == 1:
        return "轻度氟斑牙 — 白垩色条纹 <25% 牙面。建议改善饮水质量，监测进展。"
    elif y_hat == 2:
        return "中度氟斑牙 — 明显白垩色伴轻微着色。建议专业诊断并评估是否需要美学修复。"
    elif y_hat == 3:
        return "重度氟斑牙 — 大面积白垩色、着色、釉质缺损。强烈建议就医进行综合治疗评估。"
```

### 2.3 SF 诊断报告（增强版）

SF 报告额外包含：

```
--- 多部位综合分析 (如有) ---
前臂 (Forearm):    {pred_forearm}   u={u_forearm:.2%}
小腿 (Calf):       {pred_calf}      u={u_calf:.2%}
骨盆 (Pelvis):     {pred_pelvis}    u={u_pelvis:.2%}

--- 放射科医生一致参考 ---
该样本医生诊断分布: [N=0/5, Mi=1/5, Mo=3/5, Se=1/5]
医生间 Entropy:     {rater_entropy:.2f} (越低 = 越一致)
模型不确定性 vs 医生一致性: {correlation_assessment}

--- 与现行 SOTA 对比 ---
MLTrMR/Mwinc-Mamba 预测: Grade {y_sota}
差异: {diff_assessment}
我们的不确定性: {u_ours:.2%} vs SOTA: 无可用不确定性
```

## 三、可行性评估

### 3.1 优势

| 点 | 说明 |
|----|------|
| **天然产出** | EDL 的 α, u, p 直接可用，无需额外训练 |
| **差异化** | MLTrMR/Mwinc-Mamba 无法生成这种报告 — 他们没有 uncertainty |
| **临床价值** | 不确定性标注 = 自动分诊 → 高 u 送专家，低 u 直接出结果 |
| **可解释性** | Evidence 分布可视化比 softmax 单点概率更有信息量 |
| **论文亮点** | Clinical Application 章节可展示完整的报告生成 pipeline |

### 3.2 风险和限制

| 风险 | 缓解 |
|------|------|
| DF 无多标注者 → 报告缺少医生一致性参考 | 只标注"无多标注者数据可用"，不影响核心 |
| 报告模板需要医学校验 | 请吴云组或合作医生审查模板 |
| 这不是核心贡献 | 放在 Discussion 或 Supplementary，不作为 Method 主体 |

### 3.3 建议的论文定位

**主论文 (8 页）**: 专注于 EDL+ORCU 方法论 + 实验

**Supplementary / Appendix**:
- 自动诊断报告模板（DF + SF）
- 2-3 个病例演示（正常/轻度/重度各一例，含 evidence 分布图和不确定性标注）
- 用户界面 mockup

**Discussion**: 一段讨论"EDL 输出如何支撑临床决策支持系统 (CDSS)"

## 四、对实验设计的补充

在现有实验矩阵中增加两个报告相关指标：

| 新增指标 | 定义 | 用途 |
|---------|------|------|
| Recommendation Accuracy | u > θ → 推荐复核 vs 实际是否误诊 | 评估不确定性分诊的准确率 |
| Safe Prediction Rate | u < θ 的样本中 Acc | 低不确定性样本是否确实可靠 |

这两个指标可直接增强 Fig. 5 (uncertainty vs accuracy 散点图) 的故事性。
