# RQ3 matched-budget <=35

AUDIT: **PASS**

| Model | Candidates | Baseline C → corrected C | ΔC | ΔA/ΔB/ΔD | Constraints |
|---|---:|---:|---:|---:|---|
| Qwen | 12 | 0.41 → 0.55 | +0.14 | +0.00/+0.00/-0.02 | PASS |
| Mistral | 35 | 0.01 → 0.56 | +0.55 | +0.00/+0.00/-0.01 | PASS |
| DeepSeek | 35 | 0.00 → 0.79 | +0.79 | +0.00/+0.00/+0.00 | PASS |
| Llama | 35 | 0.07 → 0.53 | +0.46 | +0.00/-0.01/-0.03 | PASS |
| Gemma | 35 | 0.16 → 0.68 | +0.52 | -0.01/+0.00/-0.02 | PASS |

## Required questions

1. Cap respected: True; counts={'Qwen': 12, 'Mistral': 35, 'DeepSeek': 35, 'Llama': 35, 'Gemma': 35}.
2. Mistral subset: seven fixed layer/quantile/strength strata per fold, defined before matched-budget outcomes; see `mistral_subset_rule.txt`.
3. Training-only selection: True.
4. Positive Group-C improvement remained for: Qwen, Mistral, DeepSeek, Llama, Gemma.
5. Observed A/B/D constraints failed for: none.
6. Change from original ΔC: Qwen -0.000, Mistral -0.010, DeepSeek +0.000, Llama -0.000, Gemma -0.000.
7. Robustness conclusion is conditional on the PASS audit and must be weakened for any model without positive C gain or with a constraint failure.
8. The cap equalizes the maximum number of searched configurations; it does not make model-specific detector or actuator families identical.
