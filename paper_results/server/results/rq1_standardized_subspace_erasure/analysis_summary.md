# Standardized RQ1 subspace erasure

AUDIT: **PASS**

This file is generated only after all 75 fold-depth jobs are valid.

| Model | Rank min/median/max | Inner BA | Before → erased → attention → MLP | Recovery |
|---|---:|---:|---:|---:|
| Qwen | 6/20.0/23 | 0.558 | 0.933 → 0.778 → 0.903 → 0.893 | 74.3% |
| Mistral | 1/9.0/23 | 0.500 | 0.893 → 0.500 → 0.855 → 0.842 | 87.0% |
| DeepSeek | 9/17.0/24 | 0.508 | 0.915 → 0.570 → 0.900 → 0.898 | 95.6% |
| Llama | 2/11.0/21 | 0.497 | 0.918 → 0.495 → 0.893 → 0.887 | 92.1% |
| Gemma | 19/22.0/24 | 0.556 | 0.928 → 0.615 → 0.905 → 0.898 | 89.0% |

## Required questions

1. Common target-BA protocol: worked and passed all audits; rank-cap selections=6.
2. Selected ranks: see `selected_ranks.csv` and the per-depth histogram in `selected_rank_distribution.csv`.
3. Target closeness: `mean_selected_inner_cv_ba` and `max_target_gap` are reported without post-hoc rank expansion.
4. Held-out erasure magnitude: `erase_delta_ba` and its fold-t CI are in `model_summary.csv`.
5. Immediate-block recovery: recovery uses the unchanged primary formula and is reported per model.
6. Attention versus MLP: the two increments are reported separately; recovery after attention is temporal evidence only.
7. Mean-direction agreement: `comparison_with_mean_direction.csv` gives a descriptive matched-model comparison; estimator-independence is not assumed.
8. Model dependence: differences in erasure magnitude, target gap, and rank distribution must be retained rather than averaged away.
9. Replacement suitability: a PASS establishes protocol validity, but rows with large target gaps or rank-cap flags must be disclosed before replacing a primary table.

Interpretation boundary: most recovery may be observed after the next-block attention sublayer; this result alone does not establish that attention uniquely reconstructs telicity.
