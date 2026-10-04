# RQ1 mean-difference direction robustness

AUDIT: **PASS_WITH_PRIMARY_PROVENANCE_LIMITATION**

| Model | Before | Erased | After attention | After MLP | Attention ΔBA | MLP ΔBA | Recovery | Direction cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 0.933 | 0.873 | 0.932 | 0.930 | +0.058 | -0.002 | 93.4% | 0.423 |
| Mistral | 0.893 | 0.505 | 0.883 | 0.882 | +0.378 | -0.002 | 97.4% | 0.761 |
| DeepSeek | 0.915 | 0.500 | 0.900 | 0.903 | +0.400 | +0.003 | 96.9% | 0.428 |
| Llama | 0.918 | 0.502 | 0.907 | 0.902 | +0.405 | -0.005 | 96.5% | 0.699 |
| Gemma | 0.928 | 0.537 | 0.917 | 0.905 | +0.380 | -0.012 | 93.6% | 0.272 |

## Conservative interpretation

The mean-difference coordinate is held-out decodable in all five models. Erasure produces a large BA reduction in Mistral, DeepSeek, Llama, and Gemma, followed by near-complete next-block recovery. Qwen shows the same temporal ordering but only a modest 0.060 mean BA reduction, so robustness of the strong local-erasability magnitude is attenuated for Qwen.
In all five models, the mean attention-stage increment is positive and much larger than the mean MLP increment; the MLP means are close to zero.
Recovery observed after the attention sublayer is reported temporally; it is not, by itself, evidence that attention uniquely causes or reconstructs telicity.

## Provenance limitation

The saved publication primary comparison is not uniformly a single normalized probe-weight erasure: its underlying artifacts include calibrated multidimensional INLP/subspace removal. Therefore numeric primary-versus-mean differences are descriptive and do not isolate only the direction estimator. The new mean-difference experiment itself is internally matched across models, depths, folds, prompts, answer positions, erasure equation, and fresh-probe scoring.
