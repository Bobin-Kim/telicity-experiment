# Qwen-7B/32B matched Telicity causal scaling

AUDIT: **PASS**

Fixed site rule: 75% normalized depth, block-output answer representation (7B L21; 32B L48).

| Model | Group-C baseline ACC | Baseline margin | Telicity BA | Projection-margin r | m(-1 SD) | m(+1 SD) | G_tel |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 0.410 | +0.738 | 0.765 | -0.986 | +4.318 | -4.559 | -4.438 |
| Qwen2.5-32B-Instruct | 0.870 | -11.921 | 0.882 | -0.984 | -2.811 | -17.176 | -7.183 |

Paired delta G (32B-7B) = -2.7441 [-3.1613, -2.3269] (95% fold-t CI).
Conservative verdict: paired difference supports stronger absolute causal influence in 32B.

The two-size comparison does not establish a general causal law of scaling, and it does not by itself show that stronger Telicity influence causes higher Group-C accuracy.
