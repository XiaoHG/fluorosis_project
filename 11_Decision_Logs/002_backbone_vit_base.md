# Decision Log #002: Backbone — ViT-Base with ImageNet-21k Pretraining

**Date:** 2026-05-15
**Status:** Final
**Decided by:** Xiaohong Gao

## Context

Backbone selection for 120-sample fluorosis classification. Prior methods: MLTrMR (556M-param transformer), LD2Net (3.31M CNN), FusionDentNet (201M), HiFuse (164M).

## Options Considered

| Option | Params | Pretraining |
|--------|--------|-------------|
| ResNet-50 | 25M | ImageNet-1k |
| ViT-Base (chosen) | 86M | ImageNet-21k |
| ConvNeXt | 89M | ImageNet-1k |
| Custom transformer | 100M+ | None |

## Decision

ViT-Base with ImageNet-21k pretraining as the controlled backbone for all experiments.

## Rationale

- ImageNet-21k provides stronger inductive bias than ImageNet-1k for small medical datasets
- CE baseline (81.67%) surpassed MLTrMR (80.19%), confirming strong transfer learning
- Backbone-agnostic design: evidence head attaches to any backbone

## Consequences

- ViT-Base CE became the strongest CE result on DFID
- 6.5x parameter efficiency vs MLTrMR with better accuracy
- Limitation: not edge-deployable; LD2Net wins on efficiency
