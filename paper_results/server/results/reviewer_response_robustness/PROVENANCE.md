# Provenance

- Dataset/folds: `assets/imperfectiveNLI.json`; SHA-256 `ddd01d06b18b9bab24abb5bc92a8eb740bb145123658fa9a10bd86e38b4bdb5a`; `run_telicity_subspace_removal.py::build_verb_folds`, seed 42, five verb-disjoint folds.
- Prompt and label scoring: `run_reproduction.py::build_prompt`; native local tokenizer chat template; first-token logits for True/False/Unknown; margin `z(True)-z(Unknown)` at the last non-padding prompt position.
- Original RQ2 implementation: `telicity_intervention/run_rq2_core_replication.py`. Despite the request's “logistic-probe” label, the actual selected telicity estimator is the original class-balanced dual-ridge solution with alpha `.01`, oriented C-positive. Its fold-specific stored axes and training-projection SDs are reused without refitting.
- Qwen/DeepSeek/Llama/Gemma RQ2 assets: `results/rq2_core_replication/<model>/intervention_assets.npz`.
- Mistral final same-site assets: `results/mistral_l29_same_site_telicity/intervention_assets.npz`; zero-based L29 decoder-block output. Obsolete final-RMSNorm/L32 artifacts are excluded.
- Alternative class-distance and mean-difference directions: fit from outer-training C/D activations only; population SD (`ddof=0`); held-out fold excluded.
- RQ3 Qwen/Mistral/DeepSeek rules: `/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/validation/run_rq24b_standardized.py` and saved `results/rq2_4b_standardized/<model>/folds/fold*/selection.json`.
- RQ3 Llama/Gemma rules: `telicity_intervention/run_rq3_llama_gemma_confirmatory.py` and saved `results/rq3_llama_gemma_confirmatory/<model>/standardized/fold*/selection.json`.
- RQ3 detector, threshold, intervention site, direction, and sign are reconstructed from the original fold-training data and frozen. Only the common dose multiplier changes. Existing Qwen held-out-post-selection and DeepSeek exploratory safety-buffer validity caveats remain.
- Every model/tokenizer is loaded from its recorded local path with `local_files_only=True`; hub/datasets/transformers offline flags are set. No original artifact is overwritten.
