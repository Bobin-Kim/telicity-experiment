# RQ2 decision-direction controls: frozen design

This directory is non-destructive. It reads the final audited directions and canonical data but never writes into canonical RQ1/RQ2/RQ3 directories.

## Frozen sites and directions

The source of truth is `results/rq2_final_decision_neutralization/final_calibration_assets.npz` plus `final_calibration.csv`. Qwen uses the L27 block-input residual; Mistral uses the fixed zero-based L29 block output (not the legacy L32 final-RMSNorm predictor); DeepSeek uses block-index 28 output; Llama and Gemma use their outer-fold-specific selected block outputs. Every direction is unit L2, margin-positive, and frozen before these controls.

## Output-readout control

The raw output direction is `normalize(W_U[True] - W_U[Unknown])`. Its dimensional coordinates match each residual stream, but every validated intervention site occurs before downstream computation and/or the final normalization. It is therefore labeled `RAW_UNEMBED_EFFECTIVE_SPACE_UNMATCHED`, not treated as an effective matched-site readout.

The primary matched-site baseline is the local gradient `normalize(mean_i(normalize(∂m_i/∂h_site)))`, estimated on the outer-training population only. The gradient, its sign, and its training projection SD are frozen before held-out evaluation. Each decision, raw-unembedding, and gradient direction receives its own `±1` outer-training projection SD dose.

## Prompt and label transfer

The original decision direction, site, sign, and original-training projection SD remain frozen. Prompt controls change only the instruction wording or label listing order. Label controls use two counterbalanced A/B/C mappings; tokenizer validation requires A, B, and C each to be a single token for every model. Semantic margins score the token mapped to True minus the token mapped to Unknown, while canonical True/Unknown logits are recorded separately.

All conditions are evaluated per held-out fold. No held-out row selects a layer, direction, sign, dose, prompt, token mapping, or threshold. Final intervals use 10,000 fold-stratified paired example bootstraps with seed 42.
