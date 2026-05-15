# Response to Reviewers — Template

> To be completed after receiving first-round reviews.

---

Dear Editor and Reviewers,

We thank the reviewers for their careful reading and constructive feedback. We have addressed all comments point by point. Our responses are in blue, and revised text is highlighted in red in the manuscript.

---

## Reviewer 1

### Comment 1.1
[Copy reviewer comment]

**Response**: [Our response]

**Action**: [What we changed, with page/line references]

---

### Comment 1.2
[Copy reviewer comment]

**Response**: [Our response]

**Action**: [What we changed]

---

## Reviewer 2

### Comment 2.1
[Copy reviewer comment]

**Response**: [Our response]

**Action**: [What we changed]

---

## Reviewer 3

### Comment 3.1
[Copy reviewer comment]

**Response**: [Our response]

**Action**: [What we changed]

---

## Common Reviewer Concerns — Prepared Responses

### Concern: "Why not MC Dropout or Deep Ensembles for uncertainty?"
**Prepared**: We include both as baselines (Section 4.2). EDL offers two advantages: (1) uncertainty from a single forward pass (no sampling overhead), and (2) interpretable per-class evidence (e_k = alpha_k - 1), unlike variance-based uncertainty from ensembles.

### Concern: "ResNet-50 is not Mamba — comparison with Mwinc-Mamba is unfair."
**Prepared**: We acknowledge this in Limitations (Section 6). Our contribution is the loss function (EDL+ORCU), which is backbone-agnostic. We compare under fixed ResNet-50 to isolate the loss effect, and report Mwinc-Mamba as a reference point only.

### Concern: "The dataset is too small (80/200 images)."
**Prepared**: We agree (Limitations, Section 6). The small-sample setting is precisely why uncertainty quantification is critical. With only 48 SF training samples, knowing when the model is uncertain is essential. Our results demonstrate EDL produces informative uncertainty even in this extreme regime.

### Concern: "Where is external validation?"
**Prepared**: External validation is planned as future work (Section 6). The current datasets are the only annotated fluorosis datasets available from our clinical collaborators. Multi-center data collection is ongoing.

### Concern: "DF has no multi-rater annotations — asymmetry?"
**Prepared**: For DF, we use SORD soft encoding to approximate ordinal soft targets. We acknowledge this limitation and note multi-rater DF annotation is planned (Section 6).
