# RQ3 matched-direction control and verb-cluster bootstrap

AUDIT: **PASS**

The control freezes each matched-budget rule and substitutes only a deterministic random direction at the identical site and target L2 dose. It is therefore a direction-ablation control, not a second hyperparameter search.

| Model | Proposed ΔC (verb-bootstrap 95% CI) | Random ΔC (95% CI) | Proposed−random C (95% CI) | Proposed ΔA/B/D | Random ΔA/B/D |
|---|---:|---:|---:|---:|---:|
| Qwen | +14.0 [+8.0, +21.0] | +0.0 [+0.0, +0.0] | +14.0 [+8.0, +21.0] | +0.0/+0.0/-2.0 | +0.0/+0.0/+0.0 |
| Mistral | +55.0 [+45.0, +65.0] | +0.0 [+0.0, +0.0] | +55.0 [+45.0, +65.0] | +0.0/+0.0/-1.0 | +0.0/+0.0/+0.0 |
| DeepSeek | +79.0 [+71.0, +87.0] | +0.0 [+0.0, +0.0] | +79.0 [+71.0, +87.0] | +0.0/+0.0/+0.0 | +0.0/+0.0/+0.0 |
| Llama | +46.0 [+36.0, +56.0] | +1.0 [+0.0, +3.0] | +45.0 [+35.0, +55.0] | +0.0/-1.0/-3.0 | +0.0/+0.0/+0.0 |
| Gemma | +52.0 [+42.0, +62.0] | +1.0 [+0.0, +3.0] | +51.0 [+41.0, +61.0] | -1.0/+0.0/-2.0 | +0.0/+0.0/+0.0 |

Mistral and DeepSeek cannot have an independent ‘simple output-readout’ comparator: that is the proposed actuator itself. The random-direction substitution is the valid matched actuator control for them.

The matched random direction produced 0 pp C gain in Qwen/Mistral/DeepSeek and +1 pp in Llama/Gemma. This rules out detector selection plus intervention magnitude alone as the explanation. It does not justify calling all five corrections mechanistic: Mistral and DeepSeek remain explicitly final-readout steering, while the other three use internal learned directions.
