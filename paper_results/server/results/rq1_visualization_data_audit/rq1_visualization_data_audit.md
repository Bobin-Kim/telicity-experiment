# RQ1 visualization data audit

Status: **PASS**

## Conventions

- `layer_index` is the original one-based post-decoder-block representation label (`L1..LN`).
- The corresponding zero-based decoder block hook is `layer_index - 1`; the immediately following block has zero-based index `layer_index`.
- `normalized_depth` in the consolidated baseline CSV is the derived plotting coordinate `layer_index / total_transformer_layers`.
- `heldout_ba` is the equal mean of the five saved verb-disjoint held-out-fold balanced accuracies. BA is the mean of class recall for C (telic) and D (atelic).
- The main erasure CSV averages each site over five folds. Because erasure rank is fold-specific, its `selected_rank` is explicitly the arithmetic fold mean; exact integer ranks are in the by-fold CSV.
- The current-paper check averages the three depths within each fold first, then averages the five fold means equally.

## Availability by model

| Model | Layers | Baseline layer-wise BA range | 25/50/75% sites | Selected-site recovery coverage |
|---|---:|---:|---|---|
| Qwen2.5-7B-Instruct | 28 | 0.725–0.975 | L7/L14/L21 | Only the immediately following attention/MLP for L7/L14/L21. A separate full L11→L28 trajectory exists, but it is not a trajectory for the three selected sites. |
| Mistral-7B-Instruct-v0.3 | 32 | 0.690–0.960 | L8/L16/L24 | Only the immediately following attention/MLP. No full subsequent-layer trajectory for the selected 25/50/75% erasures is saved. |
| DeepSeek-LLM-7B-Chat | 30 | 0.745–0.970 | L8/L15/L23 | Only the immediately following attention/MLP. No full subsequent-layer trajectory for the selected 25/50/75% erasures is saved. |
| Llama-3.1-8B-Instruct | 32 | 0.710–0.970 | L8/L16/L24 | Only the immediately following attention/MLP. No full subsequent-layer trajectory for the selected 25/50/75% erasures is saved. |
| Gemma-2-9B-it | 42 | 0.775–0.965 | L11/L21/L32 | Only the immediately following attention/MLP. No full subsequent-layer trajectory for the selected 25/50/75% erasures is saved. |

## Source files

### Qwen2.5-7B-Instruct

- `/home/bobin/imperfective_paradox_repro/results/qwen25_7b_telicity_validation_suite_A4/stepB_token_level_probe_grid.csv`
- `/home/bobin/imperfective_paradox_repro/results/qwen25_7b_experimentA_immediate_reemergence_A28/removal_layer_sweep_fold_results.csv`
- `/home/bobin/imperfective_paradox_repro/results/qwen25_7b_experimentA_immediate_reemergence_A28/next_block_components_fold_results.csv`
- `/home/bobin/imperfective_paradox_repro/results/rq1_harmonized_four_stage_reaggregation/depth_fold_stage_values.csv`

### Mistral-7B-Instruct-v0.3

- `/home/bobin/imperfective_paradox_repro/results/Mistral_RQ1_sections_1_2/encoding_foldwise.csv`
- `/home/bobin/imperfective_paradox_repro/results/Mistral_RQ1_sections_3_4/attention_mlp_foldwise.csv`
- `/home/bobin/imperfective_paradox_repro/results/rq1_harmonized_four_stage_reaggregation/depth_fold_stage_values.csv`

### DeepSeek-LLM-7B-Chat

- `/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/results/deepseek-7b-telicity-rq/telicity_probe_foldwise.csv`
- `/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/results/deepseek-7b-telicity-rq/removal_sweep/removal_layer_sweep_fold_results.csv`
- `/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/results/deepseek-7b-telicity-rq/removal_sweep/next_block_components_fold_results.csv`
- `/home/bobin/imperfective_paradox_repro/results/rq1_harmonized_four_stage_reaggregation/depth_fold_stage_values.csv`

### Llama-3.1-8B-Instruct

- `/home/bobin/imperfective_paradox_repro/results/rq1_c1_c3_replication/llama31_8b/layerwise_probe_fold_results.csv`
- `/home/bobin/imperfective_paradox_repro/results/rq1_c1_c3_replication/llama31_8b/reemergence_fold_summary.csv`
- `/home/bobin/imperfective_paradox_repro/results/rq1_harmonized_four_stage_reaggregation/depth_fold_stage_values.csv`

### Gemma-2-9B-it

- `/home/bobin/imperfective_paradox_repro/results/rq1_c1_c3_replication/gemma2_9b/layerwise_probe_fold_results.csv`
- `/home/bobin/imperfective_paradox_repro/results/rq1_c1_c3_replication/gemma2_9b/reemergence_fold_summary.csv`
- `/home/bobin/imperfective_paradox_repro/results/rq1_harmonized_four_stage_reaggregation/depth_fold_stage_values.csv`

## Recovery-scope audit

- Baseline BA at every Transformer layer: available for all five models, with all five fold values.
- Erasure at 25/50/75% depth: available for all five models and all five folds.
- Primary selected-site recovery: measured only at the input, post-attention, and post-MLP states of the immediately following block.
- Full downstream trajectories for each selected site: **unavailable for all five models**.
- Qwen exception/qualification: `results/qwen25_7b_telicity_validation_suite_A4/stepC_reemergence_probe_grid.csv` stores a separate L11 erasure trajectory over L11–L28. It must not be presented as the L7/L14/L21 trajectory.
- The Qwen Experiment-B and Mistral moving-layerwise files test immediate L→L+1 recovery at many different erasure sites; they are not repeated measurements of one intervention propagated through every later layer.

## Current-paper sanity check

All extracted values reproduce `results/rq1_harmonized_four_stage_reaggregation/publication_table.csv` at full precision (tolerance 1e-10).

| Model | Pre | Erased | Next attention | Next MLP | Max absolute difference |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 0.933 | 0.703 | 0.888 | 0.872 | 0 |
| Mistral-7B-Instruct-v0.3 | 0.893 | 0.490 | 0.848 | 0.832 | 1.11e-16 |
| DeepSeek-LLM-7B-Chat | 0.915 | 0.690 | 0.882 | 0.888 | 1.11e-16 |
| Llama-3.1-8B-Instruct | 0.918 | 0.492 | 0.883 | 0.888 | 2.22e-16 |
| Gemma-2-9B-it | 0.928 | 0.588 | 0.908 | 0.898 | 0 |

## Missing values and ambiguities

- No requested baseline or selected-site immediate-recovery BA is missing.
- There is no single fold-invariant selected rank; use the by-fold CSV when rank is visualized or annotated.
- The first three models' historical component files omit example IDs, so exact stage-wise held-out ID equality is not directly recoverable from those files. Llama and Gemma retain per-example auditability.
- Do not substitute the later `rq1_standardized_subspace_erasure` rerun: it uses a different common ridge/rank-calibration protocol and therefore different post-erasure values.

## Output integrity

- Baseline rows: 164 (expected 164).
- Erasure aggregate rows: 15 (expected 15).
- Erasure fold rows: 75 (expected 75).
- Selected-site pre-erasure BA matches the independent layer-wise baseline BA at all 15 model/depth sites (tolerance 1e-10).
- GPU/model inference: not run.
