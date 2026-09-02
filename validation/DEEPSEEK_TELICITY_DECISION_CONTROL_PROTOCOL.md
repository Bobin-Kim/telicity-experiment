# DeepSeek Telicity vs Decision-Control: Reproduction Protocol

## Scientific question

This workflow tests whether DeepSeek's Telicity representation itself controls the continuous final preference margin, or whether control is carried by a separate decision representation. The only estimand is

```text
m = z(True_first_token) - z(Unknown_first_token)
```

For the registered DeepSeek tokenizer the scoring IDs are `True=8173` and `Unknown=21483`. No generated response, full verbalizer sequence, binary response classifier, or thresholded True/Unknown target is used.

## Canonical protocol boundary

The historical Qwen and Mistral implementations are not identical. The executable DeepSeek protocol uses their common, declared intersection:

- Qwen source: `run_qwen_rq2a_continuous_preference.py`, `run_qwen_rq2bc_continuous.py`, and `run_qwen_rq2c_continuous.py` under `/home/bobin/imperfective_paradox_repro/telicity_intervention/`. These establish train-only continuous preference axes, held-out folds, actual forwards, signed doses, and orthogonal random controls.
- Mistral source: `run_mistral_rq1_experiment5.py`, `aggregate_mistral_rq1_experiment5.py`, `run_mistral_matched_magnitude_gain_audit.py`, and `analyze_mistral_matched_magnitude_gain_audit.py` in the same directory. These establish the Telicity/decision/readout comparison, common train-only matched L2 scale, exact output readout, and BF16 realized-norm audit.

DeepSeek effect sizes were not inspected to select a layer or threshold. Existing full-depth predictive results froze strongest Telicity at L19 and strongest continuous-margin prediction at L30 before causal forwards began. Predictive R² is candidate-selection evidence only.

## Reused prerequisites

The workflow refuses to proceed unless these completed artifacts are present and valid:

- `results/deepseek-7b-telicity-rq/full_depth_cache.npz`: 400 × 30 × 4096 residual cache and the continuous margin.
- `results/deepseek-7b-telicity-rq/encoding_decision_summary.json` and `encoding_decision_audit.json`: PASS, with L19/L30 selection.
- `results/deepseek-7b-telicity-rq/removal_sweep/summary.json`: prior moving-erasure result, referenced but not rerun.
- `results/deepseek-7b-telicity-rq/source_pathway/layer_fold_pathway_results.csv`: prior source-pathway result, referenced and hashed but not rerun.

Behavior, Telicity probes, moving erasure, re-emergence, Attention/MLP analysis, source pathways, and decision decoding are never recomputed by this workflow. The only extra integrity forward reproduces the 400 cached margins using their original sequential batch-8 grouping; it does not regenerate a behavioral table or activation cache.

## Axis estimation and intervention sites

All fits use four outer-training verb folds and evaluate the fifth fold. There are five folds of 80 examples each.

- Telicity L19: class-balanced dual ridge (`alpha=1e-2`) on C/D examples only, with telic groups C versus atelic group D. The normalized primal coefficient is the 1D axis.
- Decision L30: standardized ridge (`alpha=10`) from residual state to continuous cached margin using all outer-training A/B/C/D examples. The normalized coefficient in original feature units is the 1D axis.
- Exact readout: normalized `lm_head.weight[True] - lm_head.weight[Unknown]`, applied at the final-normalization output.
- Final-norm Telicity: train-only C/D Telicity axis at the final-normalization output. This is an off-target control, not a primary causal candidate.

L19 and L30 hooks modify the decoder-layer output at the last non-padding prompt token—the position whose next-token logits are scored. The readout/off-target hook modifies the final-normalization output at that same position. Every hook performs a real model forward and is removed in `finally`.

## Scale, doses, and controls

For each fold, 64 seeded random directions estimate the median train-only projection SD at L19, L30, and final norm. Their geometric mean is the common 1-SD target L2 norm. Every actuator is unit normalized and receives the same target norm.

Because the model runs BF16, eight calibration iterations adjust pre-cast amplitude against the realized post-cast displacement. The inherited Mistral audit requires mean relative norm error ≤1% and maximum ≤5%.

Each example has 43 conditions:

- one fresh baseline;
- L19 Telicity: actual and orthogonal random at `±0.5, ±1, ±2`, plus label permutation at `±1`;
- L30 decision: actual and orthogonal random at `±0.5, ±1, ±2`, plus label permutation at `±1`;
- final norm: exact readout and orthogonal random at `±0.5, ±1, ±2`, plus final-norm Telicity off-target at `±1`.

Thus the full run contains `400 × 43 = 17,200` unique rows. Seed `42042 + fold` controls permutation and orthogonal controls. The fold assignment itself comes from the locked generic configuration.

## Estimation and decision rule

The primary effect is the fold-level, random-adjusted, bidirectional 1-SD contrast:

```text
[(Δm actual +1) - (Δm actual -1)]
- [(Δm random +1) - (Δm random -1)]
```

The report uses a two-sided one-sample t test and t confidence interval over five held-out-fold effects. A primary learned axis succeeds only when:

1. the effect CI excludes zero in the axis's predefined direction and at least four of five folds have that sign;
2. the random-adjusted signed dose slope across all six doses has a CI excluding zero in the same direction;
3. for Telicity/decision axes, the actual-minus-label-permutation contrast has a CI excluding zero in that direction.

The decision and readout axes are oriented a priori toward increasing `m`; a sign-reversed effect cannot be rescued post hoc. A/B/C/D group effects, True/False/Unknown logit decomposition, and prediction-change rates are reported as specificity checks. Geometry includes foldwise cosine/principal-angle summaries and cross-fold 90%-energy subspace overlap. Geometry and decoding alone never establish control.

The final-norm Telicity off-target condition has only `±1` and no label-permutation series. It is intentionally ineligible for the A–D primary classification even if it has a nonzero secondary signal.

Classification is fail-closed:

- A: primary Telicity succeeds and decision does not.
- B: primary Telicity and the separately estimated decision axis both succeed.
- C: primary Telicity fails and the decision axis succeeds.
- D: neither primary axis succeeds.
- E: any required intervention, control, scorer, cache, row-grid, or numerical audit fails.

## One-command execution and resume

From the repository root:

```bash
python scripts/run_deepseek_telicity_decision_control.py --gpus 0,1,2,3
```

For one GPU:

```bash
python scripts/run_deepseek_telicity_decision_control.py --gpus 0
```

The launcher prints the immutable plan before CUDA work, prepares axes, assigns folds round-robin across the supplied GPUs, performs the same-batch cache audit, aggregates, and runs fail-closed handoff validation. Completed preparation, folds, and cache audit are reused. Re-running the command therefore resumes missing work and does not rerun completed folds. Per-GPU logs are saved in the output directory.

Individual phases are also available:

```bash
python validation/run_deepseek_telicity_decision_control.py plan
python validation/run_deepseek_telicity_decision_control.py prepare --gpu 0
python validation/run_deepseek_telicity_decision_control.py run --gpu 0 --folds 0 4
python validation/run_deepseek_telicity_decision_control.py audit-baseline --gpu 0 --batch-size 8
python validation/run_deepseek_telicity_decision_control.py aggregate
python validation/run_deepseek_telicity_decision_control.py validate
```

`validate` is CPU-only. It checks required files, 17,200 unique rows, all 43 conditions with 400 rows each, 860 layer/fold/group summary rows, all five fold markers, locked token IDs, exact `Δm = ΔTrue - ΔUnknown`, input hashes, cache reproduction, and the scientific audit.

## Output contract

All files are under `results/deepseek-7b-telicity-rq/telicity_decision_control/`:

- `execution_plan.json`: frozen inputs, candidate layers, scorer, and planned row count.
- `intervention_assets.npz`, `fold_axis_config.csv`: train-only axes and matched scales.
- `axis_geometry_foldwise.csv`, `axis_geometry_layerwise.csv`: alignment and subspace geometry.
- `folds/fold{0..4}.csv` and `.json`: resumable raw shards and completion markers.
- `interventions_all_examples.csv`: 17,200 example-condition rows, logits, margin, deltas, norm, and prediction changes.
- `layer_fold_causal_results.csv`: machine-readable layer/fold/group causal results.
- `summary.json`: effects, CIs, p-values, fold consistency, controls, logit decomposition, group specificity, geometry, and A–E result.
- `audit.json`, `baseline_cache_reproduction.json`, `handoff_validation.json`: scientific, cache, and handoff audits.
- `REPORT.md`, `RQ_TABLE_CELL.txt`: concise human report and exact multimodel-table text.

The completed reference run is classified **C. Separable representations**. Its report records a nonzero final-norm Telicity off-target signal as a limitation; that secondary result is not promoted to primary causal evidence.
