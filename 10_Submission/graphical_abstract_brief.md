# Graphical Abstract — Design Brief

## Layout: Top-to-bottom flow

```
[Input Images: DF photo + SF X-ray]
              |
              v
[Backbone: ViT-B (DF) / ResNet-50 (SF)] -> [Feature f]
              |
              v
[Evidence Head: FC -> z (shared logits)]
         |            |
         v            v
[EDL Path]      [ORCU Path]
z->Softplus+1   z->Softmax->p
alpha_k>1       SORD encoding
S=sum(alpha)    r_k=z_k-z_{k+1}
u=K/S           log-barrier I(r_k)
         |            |
         v            v
[L_EDL = MLE+KL]  [L_ORCU = SCE+REG]
         \            /
          v          v
      [L_total = L_EDL + lambda*L_ORCU]
              |
              v
[Output: Grade + Evidence + Uncertainty + Report]
```

## Right Panel: Two Sample Outputs
- Top: Evidence-per-class bar chart with uncertainty gauge
- Bottom: Auto-diagnosis report snippet with u-threshold recommendation

## Style
- Clean flat vector, colors matching paper (blue=EDL, orange=ORCU, red=Joint)
- Minimal text, self-explanatory flow
