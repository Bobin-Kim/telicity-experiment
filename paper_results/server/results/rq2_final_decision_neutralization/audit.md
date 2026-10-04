# Final RQ2 baseline-decision neutralization audit

**AUDIT = PASS**

- Margin: `z(True) - z(Unknown)`.
- Coordinate: `a = h · v_dec`; every fold direction has unit L2 norm and was oriented on training rows so `beta_1 > 0`.
- Boundary: training-only OLS `m = beta_0 + beta_1 a`, `a_0 = -beta_0 / beta_1`, and `sigma_train = std(a_train, ddof=0)`.
- Held-out distance: `d = (a-a_0)/sigma_train`; positive is the True-preferring side.
- Intervention: actual-forward `h_neutral = h + (a_0-a)v_dec` at the answer position and exact final site.
- Aggregation: each fold is averaged first; five folds receive equal weight. CIs use 10,000 fold-stratified example bootstraps with seed 42.
- Sign: `delta_m = m_neutral - m_baseline`; a reduction of True preference is negative.
- Mistral uses the fixed zero-based L29 block-output causal direction. Legacy L32 final-RMSNorm predictive rows and the exact output-readout control are excluded.

## Direction/site provenance

| Model | Zero-based block(s), folds 0–4 | Hook/site | Direction artifact | Actual-forward rows |
|---|---|---|---|---|
| qwen | 27,27,27,27,27 | block_input | /home/bobin/imperfective_paradox_repro/results/qwen25_7b_rq2bc_continuous_preference_A23/b1_preference_axes.npy | 200 |
| mistral | 29,29,29,29,29 | block_output | /home/bobin/imperfective_paradox_repro/results/mistral7b_matched_magnitude_gain_A33/intervention_assets.npz | 200 |
| deepseek | 28,28,28,28,28 | block_output | /home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/results/deepseek-7b-telicity-rq/full_depth_cache.npz | 200 |
| llama | 31,31,31,31,28 | block_output | /home/bobin/imperfective_paradox_repro/results/rq2_core_replication/llama31_8b/intervention_assets.npz | 200 |
| gemma | 40,40,38,39,38 | block_output | /home/bobin/imperfective_paradox_repro/results/rq2_core_replication/gemma2_9b/intervention_assets.npz | 200 |

Per-fold vector hashes and regression parameters are in `final_calibration.csv`; numerical results are in `manuscript_ready.csv` and `fold_summary.csv`.
