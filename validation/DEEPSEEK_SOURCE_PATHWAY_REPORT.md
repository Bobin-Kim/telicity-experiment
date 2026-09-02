# DeepSeek depth-wise telicity source-path audit

## Result

The existing DeepSeek output contained the completed behavioral, moving-erasure, Attention-vs-MLP, and decision-margin stages, but no Residual/Context-V/Answer-Q source-path decomposition. Only that missing Experiment-B analysis was run. It reused the existing DeepSeek cache, fold assignments, erasure bases/ranks, and Experiment-A outputs.

Using the locked Qwen/Mistral definitions, interventions, same-label length-matched controls, 0.25 joint-profile rule, minimum three-layer regime length, BIC segmentation, and 500 joint-fold bootstraps gives:

- L0-L2: not identifiable (L0 is degenerate; L1-L2 have zero valid erasure folds).
- L3-L10: Residual + Context-V.
- L11-L18: Context-V.
- L19-L27: Residual.
- L28-L30: boundary-limited.
- Stable Answer-Q regime: absent.

The selected three-regime solution occurs in 79.4% of fold bootstraps. Boundary support is 90.8% for L11 and 79.4% for L19.

The early regime label must be interpreted according to the locked rule: it is assigned when both the mean Residual and Context-V shares are at least 0.25. It is not an inference from Attention dominance. Answer-Q is also appreciable in this early interval, but does not form a fold-stable, neighboring-layer regime.

## Layer audit

Effects are fold means of the original null-adjusted `specific_causal_contribution_z`. Controls are shown as Residual/Context-V/Answer-Q same-label matched-control effects. Fold agreement is `winning folds / valid folds`; “yes” means unanimous agreement, because the original pipeline stores fold consistency but supplies no separate 4/5 binary stability threshold.

| Layer | Valid folds | Residual | Context-V | Answer-Q | Matched controls (R/C/Q) | Winning mechanism | Fold agreement | Audit | Answer-Q terminal/local audit |
|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| L0 | 0 | — | — | — | — | not identifiable | N/A | N/A | degenerate answer state |
| L1 | 0 | — | — | — | — | not identifiable | N/A | N/A | no valid erasure folds |
| L2 | 0 | — | — | — | — | not identifiable | N/A | N/A | no valid erasure folds |
| L3 | 4 | -0.003 | +0.024 | +0.112 | +0.017/+0.028/-0.014 | Answer-Q | 2/4 (no) | PASS | fail: isolated/local; no 3-layer persistence |
| L4 | 5 | +0.086 | +0.108 | +0.025 | +0.009/+0.011/-0.003 | Residual + Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L5 | 5 | +0.087 | +0.158 | -0.014 | +0.011/-0.004/+0.002 | Residual + Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L6 | 5 | +0.146 | +0.041 | +0.124 | -0.008/+0.006/-0.029 | Residual | 4/5 (no) | PASS | not an Answer-Q winner |
| L7 | 5 | +0.072 | +0.017 | +0.038 | +0.007/-0.004/-0.013 | Residual | 3/5 (no) | PASS | not an Answer-Q winner |
| L8 | 5 | +0.069 | +0.135 | +0.061 | -0.013/-0.004/-0.015 | Residual + Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L9 | 5 | -0.035 | +0.016 | +0.020 | +0.019/+0.005/-0.018 | Answer-Q | 2/5 (no) | PASS | fail: isolated/local; no 3-layer persistence |
| L10 | 5 | +0.038 | -0.001 | +0.028 | -0.012/+0.027/-0.003 | Residual | 2/5 (no) | PASS | not an Answer-Q winner |
| L11 | 5 | -0.055 | +0.035 | -0.046 | -0.005/+0.062/+0.031 | Context-V | 3/5 (no) | PASS | not an Answer-Q winner |
| L12 | 5 | +0.014 | +0.253 | -0.026 | -0.021/+0.215/+0.013 | Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L13 | 5 | -0.051 | +0.271 | +0.002 | -0.000/+0.230/+0.013 | Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L14 | 5 | -0.005 | +0.776 | +0.011 | -0.012/+0.305/-0.008 | Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L15 | 5 | -0.149 | +1.072 | +0.019 | +0.054/+0.327/-0.011 | Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L16 | 5 | +0.001 | +0.791 | +0.042 | +0.003/+0.328/+0.067 | Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L17 | 5 | +0.070 | +0.901 | -0.178 | -0.136/+0.126/+0.124 | Context-V | 5/5 (yes) | PASS | not an Answer-Q winner |
| L18 | 5 | +0.077 | +0.431 | +0.197 | +0.010/+0.119/+0.025 | Context-V | 3/5 (no) | PASS | not an Answer-Q winner |
| L19 | 5 | +0.547 | +0.163 | -0.114 | +0.076/+0.070/-0.070 | Residual | 4/5 (no) | PASS | not an Answer-Q winner |
| L20 | 5 | +0.484 | +0.098 | +0.010 | +0.067/+0.052/-0.135 | Residual | 4/5 (no) | PASS | not an Answer-Q winner |
| L21 | 5 | -0.298 | +0.033 | +0.479 | +0.372/+0.035/-0.163 | Answer-Q | 3/5 (no) | PASS | fail: isolated/local; no 3-layer persistence |
| L22 | 5 | +0.677 | +0.109 | -0.016 | -0.251/+0.039/+0.152 | Residual | 3/5 (no) | PASS | not an Answer-Q winner |
| L23 | 5 | +0.537 | +0.115 | -0.051 | -0.007/-0.004/-0.035 | Residual | 4/5 (no) | PASS | not an Answer-Q winner |
| L24 | 5 | +0.358 | +0.097 | +0.075 | +0.206/-0.038/-0.080 | Residual | 3/5 (no) | PASS | not an Answer-Q winner |
| L25 | 5 | +0.614 | +0.126 | +0.003 | -0.143/-0.013/+0.056 | Residual | 4/5 (no) | PASS | not an Answer-Q winner |
| L26 | 5 | +0.126 | +0.096 | +0.079 | +0.041/+0.020/-0.031 | Residual + Context-V | 4/5 (no) | PASS | not an Answer-Q winner |
| L27 | 5 | +0.318 | +0.173 | -0.032 | -0.028/+0.019/-0.033 | Residual + Context-V | 4/5 (no) | PASS | not an Answer-Q winner |
| L28 | 0 | — | — | — | — | boundary-limited | N/A | N/A | no completed moving-erasure transition |
| L29 | 0 | — | — | — | — | boundary-limited | N/A | N/A | no completed moving-erasure transition |
| L30 | 0 | — | — | — | — | boundary-limited | N/A | N/A | terminal block has no next block |

## Late Answer-Q audit

Layerwise Answer-Q profile winners occur only at L3, L9, and L21. Each is isolated, none is unanimous across valid folds, and none persists for the original minimum of three neighboring layers. The only late candidate, L21, has Answer-Q effect +0.479 but wins only 3/5 folds; L20 and L22 are Residual winners. L27 has a negative Answer-Q effect (-0.032). Thus L21 is a local effect, not a stable late Answer-Q regime.

## Reproducibility audit

- Raw layer/fold/path rows: 405/405.
- Paired restoration checks: 270.
- Maximum Experiment-A metric mismatch: 2.22e-16.
- Maximum raw restoration probe-score drift: 0.0121478; removed by the locked paired subtraction rather than judged with a new DeepSeek-specific cutoff.
- Raw-results SHA-256: `df35f9e72cf4f097786cc28a221ce8151390de8c1a30e0e2f8004ba677b6e5ed`.

## Qwen/Mistral-format final row

DeepSeek – Preference:

- L0-L2: not identifiable
- L3-L10: Residual + Context-V
- L11-L18: Context-V
- L19-L27: Residual
- L28-L30: boundary-limited
- stable Answer-Q regime: absent
