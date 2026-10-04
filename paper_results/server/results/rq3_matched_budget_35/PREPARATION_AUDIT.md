# RQ3 matched-budget preparation audit

Status: `CODE_PREP_PASS`; no GPU inference was run.

## Original adaptive-result provenance

- Qwen: `telicity_intervention/run_qwen_experiment6_adaptive.py` and `results/qwen25_7b_experiment6_adaptive_ta_l20_l27_A27/per_example.csv`.
- Mistral: `telicity_intervention/analyze_mistral_rq2_detector_correction.py`, `run_mistral_rq2_actual_forward_correction.py`, and `results/Mistral_RQ2_experiments_3_4_5/selective_correction*`.
- DeepSeek: `/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/validation/analyze_deepseek_rq24_d_safe.py` and its `rq2_4_d_safe_exploratory` outputs.
- Llama/Gemma: `telicity_intervention/run_rq3_llama_gemma_confirmatory.py`; publication Gemma follow-up additionally uses `run_gemma_d5_budget.py`.
- Cross-model audit: `results/rq3_followup_audit/FINAL_REPORT.md` and `rq3_rule_audit.md`.

## Matched-budget implementation

`telicity_intervention/run_rq3_matched_budget_35.py` preregisters 12 Qwen candidates and 35 candidates each for Mistral, DeepSeek, Llama, and Gemma. Qwen is newly selected leave-one-outer-fold-out from saved candidate rows. DeepSeek/Llama/Gemma reuse already-completed cap-compliant training-selected held-out rows after contract checks. Mistral alone receives new inference, using seven outcome-blind detector-layer/threshold-quantile/strength strata per fold and an actual-forward final-RMSNorm intervention for the training-selected winner.

The cap means a comparable maximum search budget, not identical detector or actuator families.

