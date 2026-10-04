# Qwen-7B / Qwen-32B Telicity-utilization follow-up

## Compact comparison

| Metric | Qwen-7B | Qwen-32B | Interpretation |
|---|---:|---:|---|
| Group C accuracy | 0.41 | 0.87 | behavioral performance |
| Telicity representation (mean held-out BA) | 0.811265664160401 | 0.8774999999999998 | fold-held-out information availability |
| Representation in incorrect C (mean z / detection rate) | -0.15871 / 25.424% | -0.395531 / 23.077% | one-class held-out score; not a within-C decoding BA |
| Telicity→margin coupling C/D (mean fold Pearson r) | -0.866093039419742 | -0.9908471757985357 | within-model standardized held-out projection |
| Telicity→margin coupling Group C (mean fold Pearson r) | -0.8952674165745449 | -0.9727915763026653 | association within the telic target group |
| Causal decision evidence | ±1-SD gain C=2.7141 | L64 OOF R2=0.999673; -1SD C Δm=-1.38797 | downstream decision direction; not the Telicity axis |

## Three preregistered questions

1. Internally represented in both: **True**.
2. Qwen-7B retains Telicity in incorrect C cases: **False**.
3. Stronger representation–decision relationship in Qwen-32B: **True**.

The causal decision-direction evidence is supporting downstream evidence only; L64 is not treated as a Telicity representation.
Group-C error-conditioned values are one-class score/detection summaries, because Group C contains no atelic examples and therefore has no within-subset balanced accuracy.
Cross-model association does not establish that scaling causally improves utilization.

## Audit

CPU-only derivation from existing PASS artifacts; no model/tokenizer load, forward pass, probe refit beyond the original stored fold axes, or source-file modification.
