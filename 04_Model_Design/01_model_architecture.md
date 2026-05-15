---
date: 2026-05-15
author: ModAgent
input_from:
  - "03_Innovation/01_innovation_blueprint.md"
  - "02_Literature/02_deep_read/06_ORCU_Ordinal_Calibration.md"
  - "02_Literature/02_deep_read/16_Mwinc-Mamba_SF_Diagnosis_BSPC2025.md"
  - "02_Literature/02_deep_read/17_MLTrMR_DF_Diagnosis_JCVI2025.md"
output_to: "04_Model_Design/01_model_architecture.md"
status: draft
---

# 模型架构设计：EDL + ORCU 氟中毒诊断模型

## 一、设计原则

1. **EDL 和 ORCU 共享 logits**: Backbone → Evidence Head 输出 logits z，EDL 从 z 经 Softplus 得 α，ORCU 直接在 z 上做有序正则 — 不需要多套参数
2. **Backbone-agnostic**: DF 用 ViT-B, SF 用 ResNet-50 — EDL+ORCU loss 是 backbone-agnostic 创新，可替换任何 backbone
3. **Drop-in loss**: 整个模型只需要替换 classifier head + loss function，不改变 backbone 结构

---

## 二、总体架构

```
                    Input Image
                         │
                         ▼
        ┌─────────────────────────────────┐
        │         Backbone                 │
        │  DF:  ViT-Base (86M)             │
        │  SF:  ResNet-50 (25M)            │
        │  Output: feature vector f ∈ R^D  │
        └─────────────────────────────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │     Evidence Head               │
        │  FC(D → 4) → z (logits)         │
        └─────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐
    │  Softplus + 1    │   │  Softmax        │
    │  α = [α₀...α₃]   │   │  p = [p₀...p₃]  │
    │  Dirichlet params│   │  class probs    │
    └─────────────────┘   └─────────────────┘
              │                     │
              ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐
    │  L_EDL(α, y)     │   │  L_ORCU(z, y)   │
    │  Evidential loss │   │  Ordinal reg.    │
    └─────────────────┘   └─────────────────┘
              │                     │
              └──────────┬──────────┘
                         ▼
              L_total = L_EDL + λ·L_ORCU
```

### 关键设计决策：logits z 作为 EDL 和 ORCU 的桥梁

这是本文的核心架构创新。同一个 logits z 同时服务两个目标：
- **EDL 路径**: z → softplus → α → Dirichlet → L_EDL
- **ORCU 路径**: z → softmax → p + logit-diff regularization → L_ORCU

相比分别维护两套预测头的设计，共享 logits 确保了 evidence 和 ordinal constraints 的一致性。

---

## 三、Evidence Head 详细设计

```python
class EvidenceHead(nn.Module):
    """Replaces standard classifier head. Outputs Dirichlet concentration params."""
    def __init__(self, in_dim: int, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)  # D → 4

    def forward(self, x):
        z = self.fc(x)                    # logits, shape: (B, 4)
        alpha = F.softplus(z) + 1.0       # Dirichlet params, each > 1
        return alpha, z                    # α for EDL, z for ORCU
```

**为何 α > 1**: Dirichlet 分布在 α_k > 1 时才有 mode (单峰)。Softplus(z) + 1 确保所有 α_k ∈ (1, ∞)。

**推理时**:
- 预测类别: `y_hat = argmax(α - 1)` (Dirichlet mode)
- 不确定性: `u = K / Σα_k` (K=4, u ∈ [0,1])

---

## 四、Loss Function 设计

### 4.1 L_EDL: Evidential Loss

```python
def edl_loss(alpha, y, lambda_kl=0.1, epoch, total_epochs):
    """
    alpha: Dirichlet concentration params (B, K)
    y: target labels (B,) — integer class indices
    """
    K = alpha.shape[-1]
    y_onehot = F.one_hot(y, num_classes=K).float()

    # Type II Maximum Likelihood
    S = alpha.sum(dim=-1, keepdim=True)          # (B, 1)
    ll = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

    # KL divergence for non-target evidence
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha  # set target α_k to 1
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (torch.lgamma(S_tilde) - torch.lgamma(S)
          - torch.lgamma(alpha_tilde).sum(dim=-1) + torch.lgamma(alpha).sum(dim=-1)
          + ((alpha_tilde - alpha) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1))

    # Annealing coefficient
    t = min(1.0, epoch / (total_epochs * 0.3))  # linear warmup for 30% epochs
    loss = ll.mean() + lambda_kl * t * kl.mean()
    return loss
```

### 4.2 L_ORCU: Ordinal Calibration Loss

```python
def orcu_loss(z, y, t=3.0):
    """
    z: logits from Evidence Head (B, K)
    y: target labels (B,) — integer class indices
    t: temperature for log-barrier (default 3.0 from paper)

    Returns: L_ORCU = L_SCE + L_REG
    """
    K = z.shape[-1]
    B = z.shape[0]

    # ---- L_SCE: Soft-Encoded Cross-Entropy ----
    # SORD encoding: soft label based on ordinal distance
    y_sord = compute_sord_encoding(y, K)       # (B, K) soft labels
    log_probs = F.log_softmax(z, dim=-1)        # (B, K)
    L_sce = -(y_sord * log_probs).sum(dim=-1).mean()

    # ---- L_REG: Ordinal-Aware Regularization ----
    reg = 0.0
    for k in range(K - 1):
        r_k = z[:, k] - z[:, k+1]  # logit difference (B,)

        # For classes k < y: r should be negative (logits increase toward y)
        mask_below = (y > k).float()           # classes below true class
        # For classes k >= y: r should be negative (logits decrease after y)
        mask_at_above = (y <= k).float()       # classes at/above true class

        # Log-barrier penalty: I(r)
        threshold = -1.0 / (t ** 2)
        penalty = torch.where(
            r_k <= threshold,
            -(1.0 / t) * torch.log(-r_k),       # log-barrier zone
            t * r_k - (1.0 / t) * torch.log(1.0 / (t ** 2)) + t  # linear zone
        )

        reg += (mask_below * penalty).mean()
        reg += (mask_at_above * penalty).mean()

    return L_sce + reg
```

### 4.3 L_total: 联合损失

```python
def total_loss(alpha, z, y, epoch, total_epochs, lambda_orcu=0.5, lambda_kl=0.1):
    L_edl = edl_loss(alpha, y, lambda_kl, epoch, total_epochs)
    L_orcu = orcu_loss(z, y, t=3.0)
    return L_edl + lambda_orcu * L_orcu
```

---

## 五、Backbone 规格

### 5.1 DF Backbone: ViT-Base

| 参数 | 值 |
|------|-----|
| 模型 | ViT-Base (vit_base_patch16_224) |
| 参数量 | ~86M |
| 输入尺寸 | 512×256 → resize 224×224 |
| Patch size | 16×16 |
| 输出维度 | 768 (CLS token) |
| 预训练 | ImageNet-21k + ImageNet-1k |

### 5.2 SF Backbone: ResNet-50

| 参数 | 值 |
|------|-----|
| 模型 | ResNet-50 (torchvision) |
| 参数量 | ~25M |
| 输入尺寸 | 512×1024 → resize 224×224 |
| 输出维度 | 2048 (after avg pool) |
| 预训练 | ImageNet-1k V2 |

> **Note**: Mwinc-Mamba (Xu et al. 2025) is the reference SOTA for SF but its custom CUDA kernels (causal-conv1d, mamba-ssm) are not portable. ResNet-50 serves as a proxy backbone. The contribution is the loss function (EDL+ORCU), which is backbone-agnostic. Comparison between ResNet-50+CE and Mwinc-Mamba (66.67%) requires cautious interpretation; see Limitations (Section 6).

---

## 六、训练配置

### 6.1 通用设置

| 参数 | DF | SF |
|------|-----|-----|
| Optimizer | AdamW | AdamW |
| LR (backbone) | 1e-4 | 1e-4 |
| LR (evidence head) | 1e-3 | 1e-3 |
| Weight decay | 0.05 | 0.05 |
| Batch size | 32 | 16 |
| Epochs | 100 | 150 |
| LR schedule | Cosine annealing | Cosine annealing |
| Warmup epochs | 5 | 10 |

### 6.2 三阶段训练策略

| 阶段 | Epochs | Loss | 说明 |
|------|--------|------|------|
| Warmup | 0–N_warmup | CE only | 标准分类预训练 evidence head |
| EDL | N_warmup–N_edl | L_EDL only | 引入 Dirichlet evidence |
| Joint | N_edl–end | L_EDL + λ·L_ORCU | 联合训练，λ 从 0 线性增到目标值 |

### 6.3 数据增强

| 增强 | DF | SF | 参数 |
|------|:--:|:--:|------|
| RandomResizedCrop | ✓ | ✓ | scale=(0.8, 1.0) |
| RandomHorizontalFlip | ✓ | ✓ | p=0.5 |
| RandomRotation | ✓ | ✓ | ±10° |
| ColorJitter | ✓ | — | brightness=0.2, contrast=0.2 |
| RandomAffine | — | ✓ | translate=0.1, scale=0.1 |

---

## 七、评估指标

| 指标 | 公式/说明 | 用途 |
|------|-----------|------|
| Acc | 正确/总数 | 主指标 |
| F1 | macro-F1 | 类不平衡鲁棒 |
| QWK | Quadratic Weighted Kappa | 有序分类一致性 |
| ECE | E[|P(correct\|conf) - conf|] | 校准度 (越低越好) |
| SCE | Static Calibration Error | 类级校准 |
| %Unimodal | 预测分布单峰比例 | ORCU 效果验证 |
| Uncertainty | K/Σα_k | EDL 不确定性 (新贡献) |
| U-ECE | Uncertainty-calibrated ECE | 不确定性校准 (新贡献) |

---

## 八、与竞品架构的差异

```
MLTrMR (Xu 2025):        Mwinc-Mamba (Xu 2025):       Ours:
┌─────────────┐          ┌─────────────┐              ┌─────────────┐
│ Latent Emb   │          │ CNN Branch   │              │ Backbone    │
│ (CNN)        │          │ SSM Branch   │              │ (ViT/RN50)  │
├─────────────┤          ├─────────────┤              ├─────────────┤
│ Encoder      │          │ Concat       │              │ Evidence    │
│ (LT Blocks)  │          ├─────────────┤              │ Head        │
├─────────────┤          │ Classifier   │              │ (FC → z)    │
│ Decoder      │          │ (FC → 4)     │              ├──────┬──────┤
├─────────────┤          ├─────────────┤              │ EDL  │ ORCU │
│ CE Loss      │          │ CE Loss      │              │ α    │ z→p  │
└─────────────┘          └─────────────┘              ├──────┴──────┤
                                                       │ L_EDL+L_ORCU│
  Point estimate           Point estimate               └─────────────┘
  Nominal classification   Nominal classification
  No uncertainty           No uncertainty               Evidence + uncertainty
                                                        Ordinal + calibration
                                                        Unimodal distribution
```
