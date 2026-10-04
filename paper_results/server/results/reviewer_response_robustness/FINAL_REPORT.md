# Reviewer-response robustness experiments

Status: **COMPLETE / AUDIT PASS**

All five models and all five canonical verb-disjoint folds completed offline through local Python. Original result files were not modified.

## Direct reviewer answers

1. **Was ±1 SD small relative to C/D separation?** Class separation (fold mean SD): qwen 2.000, mistral 1.925, deepseek 2.000, llama 1.991, gemma 2.000. See CSV for fold values.
2. **Dose response and decision comparison:** see `rq2_telicity_multidose/model_summary.csv` and `rq2_decision_comparison.csv`; effects are reported without assuming linearity.
3. **Class-distance intervention:** see `rq2_class_distance/model_summary.csv`.
4. **Mean-difference robustness:** see `rq2_mean_difference/model_summary.csv`.
5. **Direction similarity:** see each model's fold config (`original_vs_mean_difference_cosine`).
6–8. **True common-dose RQ3 results/tradeoffs:** see `rq3_common_dose/model_summary.csv`. This does not remove the pre-existing Qwen/DeepSeek selection caveats.

## Completion and provenance

- Models completed: Qwen, Mistral, DeepSeek, Llama, Gemma; folds: 5/5 each; failed folds: 0.
- Dataset SHA-256: `ddd01d06b18b9bab24abb5bc92a8eb740bb145123658fa9a10bd86e38b4bdb5a`; canonical seed-42 verb-disjoint folds.
- Internet/downloads: NO. Inference: local Python/CUDA only.
- Mistral RQ2 site: zero-based decoder block L29; obsolete L32 excluded.
- Requested 'logistic' direction nomenclature: the original executable estimator is class-balanced dual-ridge, alpha=.01; that exact direction was retained.

Detailed numerical claims and classifications should be drawn from the generated compact CSVs.
