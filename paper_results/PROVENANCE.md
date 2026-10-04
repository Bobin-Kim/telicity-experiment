# Paper ↔ results provenance

Draft: "How Language Models Reason about Telicity: A Causal Analysis of Internal Mechanisms" (Overleaf snapshot of 2026-10-04).

`server/` is a read-only copy (2026-10-04) of small summary files from
`nlpgpu7:/home/bobin/imperfective_paradox_repro/`, with the original relative paths kept.
No caches (`.npz`/`.npy`) are included. Sha256 prefixes are listed at the end.

Status codes: **EXACT** (matches to the printed precision) · **STALE** (the paper has a value from an older run; a newer audited file differs) · **MIXED** (consistent values, but taken from a different run than a related table) · **OPEN** (source not yet located).

## Claim → source

| Paper item | Source file (under `server/`) | Status | Note |
|---|---|---|---|
| §4.1 / Fig. 2: pre 0.893–0.933; post 0.495–0.778; attn +0.125–0.398; MLP −0.013…−0.002; recovery 74.3–95.6% | `results/rq1_standardized_subspace_erasure/model_summary.csv` | EXACT | Each value averages 3 depth sites × 5 folds. Fold-t CIs are in the file. 6 rank-cap selections (DeepSeek 2, Gemma 4); Qwen target gap 0.19. |
| Fig. 2 plotting | `figures/plot_rq1_four_stage_ba.py`, `results/rq1_visualization_data_audit/` | EXACT | |
| Table 8 (rank-1 mean-difference erasure) | `results/rq1_mean_difference_robustness/appendix_table.csv` | EXACT | |
| Table 3 (telicity ±1 SD, Group C) | `results/reviewer_response_robustness/rq2_telicity_multidose/model_summary.csv`; CIs in `results/rq2_reviewer_multidose_independent/telicity_vs_decision_multidose.csv` | EXACT | Probe = class-balanced dual ridge, α = .01. |
| Table 4 G_d (0.5/1/2/3 SD) | `results/rq2_table4_decision_multidose/model_summary.csv` | EXACT | Llama fold 4 uses layer 29 (others 32) → fold gain 0.45 vs ~1.7. |
| Table 4 R² | `results/rq2_core_replication/<model>/predictive_fold_summary.csv` (held-out R²) | EXACT for Qwen .998, DeepSeek .937, Llama .990, Gemma .988; **OPEN for Mistral .778** | The core-replication Mistral value is .985 at the obsolete L32 site; the L29 R² is not yet located. |
| Table 5 R_gain | `results/reviewer_response_robustness/rq2_decision_comparison.csv`; audit `results/rq2_table5_rgain_strict_audit/REPORT_KO.md` | EXACT | The newer multi-dose values (Qwen 189, DeepSeek 62) are a different site and axis; the audit says not to use them in Table 5. |
| Table 5 R_mag | `results/rq2_signal_comparison_completion/representation_magnitude_summary.csv` (C_HARMONIZED_1D); Mistral: `results/mistral_l29_same_site_telicity/magnitude_summary.csv` | EXACT | |
| Table 5 cosine (telicity vs decision) | `results/rq2_core_replication/<model>/selected_layers_and_directions.json`, `results/mistral_l29_same_site_telicity/publication_row.csv` | EXACT | Mistral −0.0015. |
| §4.2.1 cosine with local gradient 0.162–0.285 | `results/controls/decision_direction/summaries/output_readout_geometry_fold.csv` | EXACT | Measured at the final-calibration sites (Qwen L27 block input; DeepSeek block 28), not at the Table 4 sites. |
| §4.2.1 local-gradient control 0.535–8.678 | `results/controls/decision_direction/summaries/output_readout_summary.csv` | EXACT | Same sites as above. |
| Table 6 (neutralization), Qwen/Mistral/DeepSeek | `results/rq2_final_decision_neutralization/manuscript_ready.csv` | EXACT | |
| Table 6, Gemma row | same | **STALE (small)** | Margins match (0.677 / 0.825). The paper has c 1.272 / 1.540 and ∆m −0.535 / −0.647; the file has 1.274 / 1.539 and −0.527 / −0.640. |
| Table 6, **Llama row** | same | **STALE** | The paper has c_C 1.610, c_D 1.586, ∆m_C −0.999, ∆m_D −0.978. The file has 1.558, 1.531, **−2.279, −2.273**, with flip rates C 0.73 and D 0.79. |
| §4.2.2 Qwen-32B: G_tel −4.438 / −7.183; C accuracy 41% / 87% | `results/qwen_7b_32b_telicity_causal_scaling/model_summary.csv` (audit PASS, Oct 1) | EXACT | Paired ∆G = −2.74 [−3.16, −2.33]. |
| §4.2.2 BA 0.877 vs 0.811; \|r\| 0.991 vs 0.866 | older `results/qwen7_qwen32_telicity_utilization/` | **STALE** | The Oct 1 run has BA **0.882 vs 0.765** and \|r\| **0.984 vs 0.986**, so there is no stronger association in 32B. |
| Fig. 3 (ΔC +14/+55/+79/+46/+52; A/B/D within limits) | `figures/plot_group_c_before_after_ba.py` (hardcoded) ← `results/rq3_matched_budget_35/model_summary.csv` | EXACT | Fold-t CIs are in the file. Selection was training-only within a ≤35 budget (audit PASS). |
| Table 10 counts (12/35/35/35/35) | `results/rq3_matched_budget_35/candidate_spaces.csv` | EXACT | The row descriptions in the paper describe the older search spaces. |
| Table 7 (fixed dose) | `results/reviewer_response_final/matched_fixed_dose_model_summary.csv` | **MIXED** | Built around a different "adaptive" run (Qwen +10, Mistral +56, DeepSeek +41), not the Fig. 3 run. |
| §3.6 / Appendix B: "all components selected on training folds" | `results/reviewer_response_final/correction_search_space.md` | **Needs disclosure** | That file records the original Qwen rule as HELD_OUT_POST_SELECTION and the DeepSeek family as post-hoc/exploratory. The matched-budget re-selection is training-only, but the rule families were developed with knowledge of held-out results. |
| Baseline A–D accuracies (not in the paper) | `results/reviewer_response_final/matched_fixed_dose_model_summary.csv` (baseline rows), `results/rq3_matched_budget_35/model_summary.csv` | — | Suggested baseline table. |

## Results available but not in the paper

| Analysis | Source | Finding |
|---|---|---|
| Class-distance intervention (shift the full C→D distance along the telicity direction) | `results/reviewer_response_robustness/rq2_class_distance/model_summary.csv` | \|∆m\| ≤ 0.12 in every model; zero True→Unknown flips in Group C for 4 of 5 models (Gemma 1). |
| Matched random direction at the Table 3/4 site | `results/rq2_core_replication/combined_five_model/summary.csv`; Mistral: `results/mistral_l29_same_site_telicity/model_summary.csv` | Group C random gains: Qwen 0.162 [−0.153, 0.477], DeepSeek −0.018 [−0.072, 0.036], Llama 0.003 [−0.192, 0.198], Gemma −0.027 [−0.075, 0.022], Mistral −0.0003 (one direction). Telicity gains lie within the random interval for 4/5 models; for Mistral the telicity effect (−0.058) exceeds random but stays small. The `combined_five_model` Mistral row is the obsolete L32 run (decision gain 0), so don't use it. |
| Prompt-wording and label-mapping transfer of the decision direction | `results/controls/decision_direction/summaries/{prompt,label}_*` | Not yet summarized. |
| Paired fold-stratified bootstrap of correction ∆acc | `results/reviewer_response_final/bootstrap_ci.csv` | |
| Norm-matched gain ratio (derived) | projection SDs in `results/rq2_core_replication/<model>/selected_layers_and_directions.json` and `results/mistral_l29_same_site_telicity/fold_config.csv` | At equal L2 step the decision direction is **5.7–57.8×** stronger (Qwen 50.6, Mistral 5.7, DeepSeek 15.7, Llama 57.8, Gemma 9.8), vs 22.4–320.3× per own-SD. This is a linear rescaling of the existing gains, not a new run. |

## Note on `local_results/` (generic pipeline)

The Llama/Gemma files in `local_results/` come from the generic multimodel pipeline (8-component ridge basis, block-input hook). They are not the source of any paper table. For example, their Gemma Group C margin is 0.625, while the paper uses 0.677 from `rq2_core_replication`.

## File hashes (sha256, first 16 hex characters)

```
e53d1ed173543e67  figures/plot_group_c_before_after_ba.py
16c3b5f8b4aafd3a  figures/plot_rq1_four_stage_ba.py
3d876d37813e40d3  results/controls/decision_direction/DESIGN.md
99097ed560c88a8c  results/controls/decision_direction/summaries/aggregate_complete.json
11614c0097f3cc4c  results/controls/decision_direction/summaries/combined_five_model_summary.csv
66d0577406ede8f3  results/controls/decision_direction/summaries/label_aggregate.json
dd424f0585561615  results/controls/decision_direction/summaries/label_compact_cross_condition.csv
885960654cf7e2e7  results/controls/decision_direction/summaries/label_fold_summary.csv
b8368e63aab58933  results/controls/decision_direction/summaries/label_summary.csv
c5bb975f21b25744  results/controls/decision_direction/summaries/output_readout_aggregate.json
73016e22c613317f  results/controls/decision_direction/summaries/output_readout_fold_summary.csv
ba96212bb95e73d9  results/controls/decision_direction/summaries/output_readout_geometry_fold.csv
7eb77f246211d333  results/controls/decision_direction/summaries/output_readout_summary.csv
96b805614aee2b5e  results/controls/decision_direction/summaries/prompt_aggregate.json
88c4e0ffad9d1d90  results/controls/decision_direction/summaries/prompt_compact_cross_condition.csv
2e74f22495548409  results/controls/decision_direction/summaries/prompt_fold_summary.csv
348b540c2be1ed3a  results/controls/decision_direction/summaries/prompt_summary.csv
8f31d293398059c5  results/controls/decision_direction/summaries/publication/label_control.csv
fa7d89bea4b0e5d3  results/controls/decision_direction/summaries/publication/label_control.tex
1e04208086a28943  results/controls/decision_direction/summaries/publication/output_readout_control.csv
e09188df69ec05b9  results/controls/decision_direction/summaries/publication/output_readout_control.tex
a553b013eb9ac002  results/controls/decision_direction/summaries/publication/prompt_control.csv
7a98ea487fa52028  results/controls/decision_direction/summaries/publication/prompt_control.tex
474909760049af75  results/mistral_l29_same_site_telicity/audit.md
2a0fc44e69fb3efe  results/mistral_l29_same_site_telicity/cosine_summary.csv
b6e3e59ab6722361  results/mistral_l29_same_site_telicity/decision_telicity_comparison.csv
5cad7372f3f822b4  results/mistral_l29_same_site_telicity/fold_config.csv
936b3504e96dcaab  results/mistral_l29_same_site_telicity/magnitude_summary.csv
45814af0afb74a38  results/mistral_l29_same_site_telicity/model_summary.csv
c80ea08b3ba20587  results/mistral_l29_same_site_telicity/publication_row.csv
8ca6c30814988766  results/qwen7_qwen32_telicity_utilization/REPORT.md
d1719600fa08e24c  results/qwen7_qwen32_telicity_utilization/compact_summary.json
8c2bcd70b9fe3902  results/qwen7_qwen32_telicity_utilization/comparison_table.csv
817cd01e9899a476  results/qwen_7b_32b_telicity_causal_scaling/analysis_summary.md
053005934407a6d5  results/qwen_7b_32b_telicity_causal_scaling/audit_report.txt
c588c65061be965f  results/qwen_7b_32b_telicity_causal_scaling/model_summary.csv
956ee753702e6ef6  results/reviewer_response_final/FINAL_REPORT_KO.md
ebd3b65bf2150c9b  results/reviewer_response_final/bootstrap_ci.csv
5f7d10123f5e0248  results/reviewer_response_final/correction_search_space.md
7a35a437e8e1fab3  results/reviewer_response_final/label_distributions.csv
e838321d3465a01f  results/reviewer_response_final/label_transitions.csv
30906035a1a76e9f  results/reviewer_response_final/manuscript_ready_tables.tex
3f749d7049c2863a  results/reviewer_response_final/matched_fixed_dose_fold_results.csv
baec39669c1e26f8  results/reviewer_response_final/matched_fixed_dose_model_summary.csv
cf005fe8507dc900  results/reviewer_response_final/selected_vs_matched_gain_provenance.csv
3f62635a5d388f46  results/reviewer_response_final/telicity_multidose_summary.csv
f4826798d7384b21  results/reviewer_response_robustness/FINAL_REPORT.md
02d5e78af6ebfc30  results/reviewer_response_robustness/PROVENANCE.md
718b5e23f83642e7  results/reviewer_response_robustness/README.md
29c116f8d7c3afe9  results/reviewer_response_robustness/rq2_class_distance/model_summary.csv
fdbabaf1a611b51e  results/reviewer_response_robustness/rq2_decision_comparison.csv
28473134745a1309  results/reviewer_response_robustness/rq2_mean_difference/model_summary.csv
36b790e93d411e05  results/reviewer_response_robustness/rq2_telicity_multidose/model_summary.csv
c87116031d88a562  results/reviewer_response_robustness/rq3_common_dose/model_summary.csv
cd48b60788eb36de  results/rq1_mean_difference_robustness/analysis_summary.md
1e4321c524155408  results/rq1_mean_difference_robustness/appendix_table.csv
57631ae3286fcf0e  results/rq1_mean_difference_robustness/audit_report.txt
1e4321c524155408  results/rq1_mean_difference_robustness/model_summary.csv
be6b40c26a941fcc  results/rq1_standardized_subspace_erasure/analysis_summary.md
2a7d9856c32002b8  results/rq1_standardized_subspace_erasure/audit_report.txt
4e4c19023c9e4c29  results/rq1_standardized_subspace_erasure/comparison_with_mean_direction.csv
b938ce114d2ac135  results/rq1_standardized_subspace_erasure/model_summary.csv
4c4f30329f6cef2d  results/rq1_standardized_subspace_erasure/selected_ranks.csv
9663f476a71d2334  results/rq1_visualization_data_audit/rq1_erasure_sites_25_50_75.csv
3dfec89d66a84094  results/rq1_visualization_data_audit/rq1_erasure_sites_25_50_75_by_fold.csv
717c41db8347b248  results/rq1_visualization_data_audit/rq1_layerwise_telicity_ba.csv
88a51e6c762269cb  results/rq1_visualization_data_audit/rq1_visualization_data_audit.md
97faf26ba9f7bbd5  results/rq2_core_replication/combined/llama_gemma_rq2_summary.csv
ec2eb6b9ccd51416  results/rq2_core_replication/combined_five_model/summary.csv
7f185d5f14cf89ff  results/rq2_core_replication/deepseek/causal_gain_fold_summary.csv
b3989da68bc2ee14  results/rq2_core_replication/deepseek/control_summary.csv
be8a34588d505822  results/rq2_core_replication/deepseek/dissociation_summary.csv
c66156fc26f866de  results/rq2_core_replication/deepseek/experiment_config.json
5c359b61910d9de2  results/rq2_core_replication/deepseek/predictive_fold_summary.csv
f36507f9b795f454  results/rq2_core_replication/deepseek/selected_layers_and_directions.json
393b75c612e3277b  results/rq2_core_replication/gemma2_9b/causal_gain_fold_summary.csv
cbccf62433753c85  results/rq2_core_replication/gemma2_9b/control_summary.csv
e8bc0d1b04814bac  results/rq2_core_replication/gemma2_9b/dissociation_summary.csv
9e630df1c3998249  results/rq2_core_replication/gemma2_9b/experiment_config.json
881d6cd655750eed  results/rq2_core_replication/gemma2_9b/predictive_fold_summary.csv
b5001e05007952af  results/rq2_core_replication/gemma2_9b/selected_layers_and_directions.json
73dd57ab41e5f69e  results/rq2_core_replication/llama31_8b/causal_gain_fold_summary.csv
8273154f2235d61f  results/rq2_core_replication/llama31_8b/control_summary.csv
c0833af371b6df91  results/rq2_core_replication/llama31_8b/dissociation_summary.csv
6a18aa31aaa5b417  results/rq2_core_replication/llama31_8b/experiment_config.json
8eefe0fa212bb0bb  results/rq2_core_replication/llama31_8b/predictive_fold_summary.csv
e3852458f896c6d6  results/rq2_core_replication/llama31_8b/selected_layers_and_directions.json
98064a606ef180a5  results/rq2_core_replication/mistral/causal_gain_fold_summary.csv
4490a32ddbfa4962  results/rq2_core_replication/mistral/control_summary.csv
2cbcf78ffca8ddf1  results/rq2_core_replication/mistral/dissociation_summary.csv
08b83d04d9f84b6b  results/rq2_core_replication/mistral/experiment_config.json
343f10c1569a0844  results/rq2_core_replication/mistral/predictive_fold_summary.csv
4e4925dc0fd35d4d  results/rq2_core_replication/mistral/selected_layers_and_directions.json
c35aac0b41cf5a80  results/rq2_core_replication/qwen/causal_gain_fold_summary.csv
984c5e02aa4f9a6a  results/rq2_core_replication/qwen/control_summary.csv
8a3deeb371a90117  results/rq2_core_replication/qwen/dissociation_summary.csv
89aa5eda29a12cf0  results/rq2_core_replication/qwen/experiment_config.json
c13fd8fe5c3b799e  results/rq2_core_replication/qwen/predictive_fold_summary.csv
c30e73e4579c61b1  results/rq2_core_replication/qwen/selected_layers_and_directions.json
d37d53ac03e5ea30  results/rq2_final_decision_neutralization/audit.json
6cbac82e7216db46  results/rq2_final_decision_neutralization/audit.md
c4c34835522a7de1  results/rq2_final_decision_neutralization/fold_summary.csv
3f59935b85f984e5  results/rq2_final_decision_neutralization/manuscript_ready.csv
a00e042832813f25  results/rq2_reviewer_multidose_independent/audit.json
45b5a2cd76dbda28  results/rq2_reviewer_multidose_independent/fold_summary.csv
88050f80397896f1  results/rq2_reviewer_multidose_independent/model_summary.csv
8e0bc9a94d10f18b  results/rq2_reviewer_multidose_independent/telicity_vs_decision_multidose.csv
4c6d049ac89e9aa6  results/rq2_signal_comparison_completion/REPORT_KO.md
dfc08508a3760ccf  results/rq2_signal_comparison_completion/direction_cosine_summary.csv
ab286ae4192b10d5  results/rq2_signal_comparison_completion/representation_magnitude_summary.csv
b3df256a4813a6df  results/rq2_table4_decision_multidose/audit.json
7f7e7f30ad982efa  results/rq2_table4_decision_multidose/fold_config.csv
2361c82b7964039d  results/rq2_table4_decision_multidose/fold_summary.csv
077d7bb83e5a5e34  results/rq2_table4_decision_multidose/model_summary.csv
7a3ae218acde7f17  results/rq2_table5_rgain_strict_audit/REPORT_KO.md
5b9d4a993155c784  results/rq2_table5_rgain_strict_audit/aggregation_A_to_E.csv
a58050a31da98134  results/rq2_table5_rgain_strict_audit/audit.json
7ff6225e7d3ef1c3  results/rq2_table5_rgain_strict_audit/decision_axis_scale_comparison.csv
c8fc713659a9a498  results/rq2_table5_rgain_strict_audit/existing_gain_reproduction.csv
93706eb982bfde02  results/rq2_table5_rgain_strict_audit/qwen_deepseek_fold_discrepancy.csv
5c3db92f4531c2b7  results/rq2_table5_rgain_strict_audit/recomputed_1sd_values.csv
a98e5c1cd9bc00ef  results/rq3_matched_budget_35/PREPARATION_AUDIT.md
1fe2b3576ef07a58  results/rq3_matched_budget_35/analysis_summary.md
cdb462565533cb42  results/rq3_matched_budget_35/audit_report.txt
21bc348eab4ddeea  results/rq3_matched_budget_35/candidate_spaces.csv
53b81485b97623c2  results/rq3_matched_budget_35/comparison_with_original.csv
2c7c68b19228849b  results/rq3_matched_budget_35/fold_level_results.csv
78d5cbe06b23006f  results/rq3_matched_budget_35/mistral_subset_rule.txt
0c2930a94617e974  results/rq3_matched_budget_35/model_summary.csv
829319cfea030816  results/rq3_matched_budget_35/run_config.json
5cc61b5934f704dd  results/rq3_matched_budget_35/selected_rules_by_fold.csv
```
