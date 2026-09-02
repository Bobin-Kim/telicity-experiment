# DeepSeek Telicity vs Decision-Control Validation

Status: **PASS**
Classification: **C. Separable representations**

The completed reference run used five verb-disjoint held-out folds and 17,200 unique example-condition forwards. It retained the locked first-token estimand `m = z(True) - z(Unknown)` with DeepSeek token IDs `8173` and `21483`.

| Test | Random-adjusted bidirectional 1-SD effect | 95% CI | p | Fold sign | Primary success |
|---|---:|---:|---:|---:|---:|
| Telicity L19 | -0.0004 | [-0.0099, +0.0091] | 0.9146 | 3/5 | no |
| Decision L30 | +0.0442 | [+0.0384, +0.0501] | 3.041e-05 | 5/5 | yes |
| Exact final readout | +2.9116 | [+2.7325, +3.0907] | 1.440e-06 | 5/5 | positive actuator control |
| Final-norm Telicity off-target | -0.0863 | [-0.1162, -0.0564] | 0.001314 | 5/5 | ineligible secondary signal |

Telicity L19 failed the random, dose-response, and label-permutation criteria. Decision L30 passed all three. The decision contrast decomposed into `True +0.0166`, `False +0.0070`, and `Unknown -0.0276`, so it was not merely a common logit offset.

The final-norm off-target Telicity signal is retained as a limitation. That condition has only the canonical `±1` secondary contrast, lacks the primary six-dose and label-permutation design, and was declared ineligible before classification. It does not override failure of the preregistered L19 Telicity controller test.

Geometry also supports separation: mean foldwise absolute cosine was `0.038296` between final-norm Telicity and decision axes, `0.029945` between Telicity and exact readout, and `0.167689` between decision and exact readout. Geometry was not used as causal proof.

Audit checks all passed: 17,200/17,200 unique raw rows, 860 layer/fold/group rows, all 43 conditions represented by 400 examples, exact `Δm = ΔTrue - ΔUnknown`, all finite values, same-batch cache reproduction max error `0.0`, and BF16 realized-norm relative error mean `0.0475%`/maximum `2.3884%` against canonical 1%/5% limits.

The executable and full handoff instructions are in `run_deepseek_telicity_decision_control.py` and `DEEPSEEK_TELICITY_DECISION_CONTROL_PROTOCOL.md`. Machine-readable outputs remain under `results/deepseek-7b-telicity-rq/telicity_decision_control/`.

Exact table cell:

> DeepSeek – Preference: Separable representations; Telicity L19 -0.0004 [-0.0099, +0.0091], p=0.9146, 3/5 folds; decision L30 +0.0442 [+0.0384, +0.0501], p=3.041e-05, 5/5 folds; exact readout +2.9116 [+2.7325, +3.0907], p=1.44e-06, 5/5 folds.
