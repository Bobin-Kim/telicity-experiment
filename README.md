# MULTIMODEL TRUE–UNKNOWN PREFERENCE RUNNER

This package runs only the continuous True–Unknown preference protocol derived from canonical Qwen A23 and the A35 full-depth pipeline. It contains no binary probe, binary threshold, or binary mode. The locked estimand is the logit margin at the first answer token: `z(True_first_token) - z(Unknown_first_token)` at the last non-padding prompt position. Every behavioral, cached, predictive, intervention, component, and supplemental stage uses that same scorer.

## Install and authenticate

Use Python 3.10+ in an environment with a CUDA-capable PyTorch build:

```bash
python -m pip install -r requirements.txt
huggingface-cli login  # only for gated/private models; no token is stored here
```

Local Qwen and Mistral mirrors are preferred automatically. Other entries resolve through the existing Hugging Face cache/account. V100 GPUs use configured FP16; edit only model runtime fields if different hardware requires it.

## Preflight, smoke, full run, and resume

`--dry-run` validates the fixed dataset, exact IDs/order/hash, prompt rendering, folds, and tokenizer without loading weights. `--preflight-only` also loads weights and validates every hook. `--max-items` is an engineering smoke fixture and always marks output non-official.

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --dry-run
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --stages behavior,cache --max-items 1 --output-dir smoke/qwen
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --resume
```

Stages run in dependency-checked order: behavior, full component cache, full-depth predictive localization, full-depth residual causal intervention with matched controls, attention/MLP/head analysis plus actual component interventions, generator localization, and aggregation. Add supplemental A37-style upstream localization with `--stages behavior,cache,predictive_localization,causal_intervention,component_analysis,generator_localization,aggregate,upstream_source`. An audit failure preserves diagnostics but sets `official: false` and blocks dependent interpretation.

Outputs are isolated under `results/<model-key>/`. `run_state.json`, `resolved_config.json`, and `preflight.json` lock the model, fixed configuration, smoke scope, and code fingerprint. Resume refuses incompatible state; a nonempty unowned directory is rejected. CSV/JSON writers reject empty official artifacts and NaN/Inf.

## Copy-paste full commands

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model mistral-7b-v03 --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model llama3.1-8b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model yi1.5-9b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model deepseek-7b-chat --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model gemma2-9b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model glm4-9b --gpus 0
```

The full run is intentionally expensive: it covers every layer and all five held-out verb folds. The one-item smoke does not constitute a scientific result.

## Configuration contract and adding a model

Never change `fixed_experiment_config` between models: dataset/hash, A/B/C/D definitions, prompt text/semantics, folds, seed, scoring rule/position, intervention conditions, controls, selection procedure, stage order, and audit criteria are invariants. `model_configs` may contain only loading/tokenizer settings, architecture geometry, adapter/module mapping, dtype/device, and memory options.

To add a compatible decoder, add one model entry with official ID/revision/geometry and module paths. The reusable HF decoder adapter requires layers, attention, MLP, attention output projection input, final norm, and LM head. Run tokenizer dry-run, weight-loading preflight, then the one-item behavior/cache smoke before a full run. Do not insert discovered layer/head numbers.

See `validation/` for exact validation scope and access limitations. The canonical scoring evidence is `run_qwen_rq2bc_continuous.py` lines 35 and 91 (cached and live `logits[:,0]-logits[:,2]`) plus `run_qwen_full_depth_true_signal.py` line 61 (first token of each label).
