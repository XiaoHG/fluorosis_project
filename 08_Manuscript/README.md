# 08_Manuscript — Paper Writing

## Directory Structure

```
08_Manuscript/
├── v1_first_draft/           # First complete draft
│   ├── main.tex              # Main LaTeX (elsarticle, target: MedIA)
│   ├── references.bib        # Bibliography
│   ├── sections/
│   │   ├── 01_introduction.tex
│   │   ├── 02_related_work.tex
│   │   ├── 03_method.tex
│   │   ├── 04_experiments.tex
│   │   ├── 05_results.tex    # [PLACEHOLDER: needs experimental data]
│   │   ├── 06_discussion.tex
│   │   └── 07_conclusion.tex
│   └── figures/
├── v2_internal_review/
├── v3_collaborator_feedback/
└── README.md
```

## Compilation

```bash
cd 08_Manuscript/v1_first_draft
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Current Status

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | Draft | Complete, needs author info |
| 1. Introduction | Draft | 4 contributions stated |
| 2. Related Work | Draft | Fluorosis DL, EDL, ORCU, Multi-Rater |
| 3. Method | Draft | Full equations: shared-logit bridge, losses, training |
| 4. Experiments | Draft | Datasets, baselines, 8+2 metrics, 6 ablations, 6 U-analyses |
| 5. Results | PLACEHOLDER | Tables structured, all values need experimental data |
| 6. Discussion | Draft | 4 findings, 4 limitations, 5 future directions |
| 7. Conclusion | Draft | 4 contribution summary |
| References | Draft | 14 entries, [Authors] need completion |

## TODO Before Submission

1. [ ] Fill all [X] placeholders after running experiments
2. [ ] Complete [Author]/[Journal]/[Year] in references
3. [ ] Add author info (name, email, affiliation)
4. [ ] Add funding acknowledgment
5. [ ] Insert figures from 07_Visualization
6. [ ] Add supplementary material (auto-report examples)
7. [ ] Proofread equations and formatting
8. [ ] Reference format check
