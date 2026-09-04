# MULTIMODEL TRUE–UNKNOWN PREFERENCE RUNNER

RQ1 studies where Telicity is represented and how it re-emerges after removal. RQ2 studies how the final decision signal emerges, how the erroneous Group-C True-driving signal is causally routed, and how it can be selectively corrected.

The repository is logically divided into `src/common/`, `src/rq1/`, and `src/rq2/`. The files in `src/common/` are stable re-exports of the existing dataset, model-adapter, scoring, hook, cache, statistics, control, output-schema, and audit implementations. This avoids copying numerical logic while giving new researchers an explicit ownership boundary. Existing files and result directories remain in place so their paths and hashes stay auditable.

The only executable score is the continuous first-answer-token margin

```text
m = z(True_first_token) - z(Unknown_first_token)
```

at the last non-padding prompt position. There is no binary True/Unknown executable stage, classifier, threshold, or alternate scoring mode.

## Research-question entry points

### Fold budgets: use five folds only for final claims

The shared runners expose three explicit fold profiles:

| Profile | Held-out folds | Intended use | Official result? |
|---|---:|---|---|
| `pilot` | 0 (80 held-out items) | debugging, feasibility checks, supplementary diagnostics | no |
| `standard` | 0–2 (240 held-out items) | candidate screening and replication development | no |
| `confirmatory` | 0–4 (400 held-out items) | final confidence intervals and causal/model comparisons | yes, subject to audit |

`confirmatory` remains the default so an omitted flag cannot accidentally produce a
publication claim from partial folds. Expensive intervention stages now honor the
selected fold budget; fold-independent behavior and activation caches are still
computed once over the 400-item dataset. Partial runs are written to separate
`results/<model>-pilot/` or `results/<model>-standard/` directories and are marked
`official: false`. A one-fold result must never be reported as confirmatory evidence.

```bash
# Fast handoff/smoke run for another researcher
scripts/run_rq2.sh qwen2.5-7b rq2.decision_signal_emergence 0 --fold-profile pilot

# Three-fold screening before committing to the final run
scripts/run_rq2.sh qwen2.5-7b rq2.decision_signal_emergence 0 --fold-profile standard

# Final inference and confidence intervals
scripts/run_rq2.sh qwen2.5-7b rq2.decision_signal_emergence 0 --fold-profile confirmatory
```

RQ1 asks where and how Telicity is represented and reconstructed after removal. Its entry point runs only Telicity encoding/cache construction, moving erasure and downstream re-emergence, Attention-versus-MLP recovery, depth-wise aggregation, and source-pathway decomposition. The locked cross-model source paths are `Residual / Context-V / Answer-Q`. `Context-K / Answer-Gate` is an opt-in RQ1 extension; its output is separate and never overwrites or renormalizes the locked three-path result.

```bash
# Locked, final RQ1 protocol
scripts/run_rq1.sh deepseek-7b-chat 0 --fold-profile confirmatory

# Exploratory RQ1 handoff (one held-out fold; non-official)
scripts/run_rq1.sh deepseek-7b-chat 0 --fold-profile pilot

# Locked protocol plus the separately reported K/G extension
scripts/run_rq1.sh deepseek-7b-chat 0 --include-extension

# Routing only: no tokenizer, weights, or GPU work
python -m src.rq1.runner --model deepseek-7b-chat --gpus 0 --plan-only
```

RQ2 asks how the erroneous True-driving signal in Group C is formed, causally routed to the final decision, and selectively corrected. Exactly one of these selectors is required:

| Selector | Question | Current generic behavior |
|---|---|---|
| `rq2.representation_imbalance` | RQ2.1 representation vs readout/routing imbalance | reuse an audited registered artifact, otherwise report planned |
| `rq2.decision_signal_emergence` | RQ2.2 predictive emergence versus causal effectiveness | run/resume the scoped generic stages |
| `rq2.decision_circuit` | RQ2.3 causal decision pathway | run/resume the scoped generic stages |
| `rq2.selective_group_c_correction` | RQ2.4 selective Group-C correction with controls | reuse an audited registered artifact, otherwise report planned |
| `rq2.telicity_decision_relationship` | RQ2.5 shared, overlapping, or separable representations | reuse the completed model backend/artifact |
| `rq2.true_driving_origin` | RQ2.6 causal origin, propagation, and readout | reuse only a causal artifact that passed audit, otherwise report planned |

```bash
# A selected executable generic RQ2 workflow
scripts/run_rq2.sh qwen2.5-7b rq2.decision_signal_emergence 0

# A selected completed-artifact workflow (read-only reuse)
scripts/run_rq2.sh deepseek-7b-chat rq2.telicity_decision_relationship 0

# Routing only
python -m src.rq2.runner --model qwen2.5-7b \
  --subquestion rq2.decision_circuit --gpus 0 --plan-only
```

Omitting `--subquestion` fails closed. RQ2 never runs automatically after RQ1, and selecting one RQ2 subquestion never schedules the other five. The old `upstream_source` correlation summary is explicitly excluded from RQ2.6 because correlation, probe accuracy, Attention dominance, and temporal order do not establish an upstream causal source.

## Shared prerequisites and model aliases

Both entry points resolve the immutable experiment contract from `configs/multimodel_true_unknown_preference.yaml`: fixed dataset and SHA-256, A/B/C/D conditions, prompts, tokenizer/chat rendering, True/Unknown token validation, folds, seeds, continuous scoring, controls, hook utilities, statistics, schemas, and audits. The RQ-specific YAML files contain routes and protocol labels, not effect sizes, p-values, discovered layers, or model conclusions.

Registered aliases are `qwen2.5-7b`, `mistral-7b-v03`, `llama3.1-8b`, `yi1.5-9b`, `deepseek-7b-chat`, `gemma2-9b`, and `glm4-9b`. Add a model only to the shared model registry; do not hard-code its layers or findings in an RQ runner.

Completed stages are reused with `--resume`. The underlying controller verifies ownership, configuration/code fingerprints, completion markers, and required artifacts before reuse; a missing or invalid artifact is not silently treated as complete. Manifest-only RQ2 routes are read-only and either return an audited official artifact or stop as planned/unavailable. No RQ-specific entry point performs a hidden fallback to another model's implementation.

## Artifacts, audit status, and migration map

- `manifests/rq1_artifacts.json` classifies existing RQ1 artifacts, including the locked and extension boundaries.
- `manifests/rq2_artifacts.json` classifies RQ2.1–RQ2.6 artifacts and execution/audit status.
- `manifests/all_artifacts.json` is the combined inventory; `manifests/result_hash_baseline.json` locks every pre-existing file under `results/`.
- `manifests/code_migration_map.json` records old path → new logical ownership without moving old files.
- `validation/RQ_ORGANIZATION_VALIDATION.md` and its JSON companion contain the CPU-only routing and integrity checks; `validation/RQ_ORGANIZATION_FINAL_AUDIT.md` records the no-protocol/no-result-change conclusion.

Completed Qwen, Mistral, and DeepSeek Telicity–decision relationship implementations are classified under RQ2.5; the completed DeepSeek classification remains in its result/report, not executable config. Failed Mistral causal-pathway and upstream-source artifacts, and the failed cross-model upstream summary that depends on them, remain preserved but are classified `invalid` and cannot be selected as official RQ2 output. Unsupported combinations are `planned/unexecuted`; they are not substituted with Qwen code. The legacy generic `upstream_source` correlation-only output is retained for audit history and is never promoted to official RQ2.6 evidence.

`run_multimodel_preference.py` remains the shared low-level compatibility runner. New handoffs should use `scripts/run_rq1.sh` or `scripts/run_rq2.sh`, because their routing makes the research-question boundary explicit.

## Install and authenticate

Use Python 3.10+ in an environment with a CUDA-capable PyTorch build:

```bash
python -m pip install -r requirements.txt
huggingface-cli login  # only for gated/private models; no token is stored here
```

The fixed 400-item ImperfectiveNLI dataset is included at
`assets/imperfectiveNLI.json`; its SHA-256 is checked before every run. Model
weights and generated result directories are intentionally not included.

Local Qwen and Mistral mirrors are preferred automatically. Other entries resolve through the existing Hugging Face cache/account. Mistral uses its original single-device BF16 protocol on this server; other model dtypes remain model-configured. Edit only model runtime fields if different hardware requires it.

## Legacy low-level runner: preflight, smoke, full run, and resume

`--dry-run` validates the fixed dataset, exact IDs/order/hash, prompt rendering, folds, and tokenizer without loading weights. `--preflight-only` also loads weights and validates every hook. `--max-items` is an engineering smoke fixture and always marks output non-official.

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --dry-run
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --stages behavior,cache --max-items 1 --output-dir smoke/qwen
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --resume
```

Stages run in dependency-checked order: behavior, full component cache, full-depth predictive localization, full-depth residual causal intervention with matched controls, attention/MLP/head analysis plus actual component interventions, generator localization, and aggregation. Add supplemental A37-style upstream localization with `--stages behavior,cache,predictive_localization,causal_intervention,component_analysis,generator_localization,aggregate,upstream_source`. An audit failure preserves diagnostics but sets `official: false` and blocks dependent interpretation.

Outputs are isolated under `results/<model-key>/`. `run_state.json`, `resolved_config.json`, and `preflight.json` lock the model, fixed configuration, smoke scope, and code fingerprint. Resume refuses incompatible state; a nonempty unowned directory is rejected. CSV/JSON writers reject empty official artifacts and NaN/Inf.

## Layer-wise re-emergence and Q/K/V/Gate pathways

The complete supplemental chain is:

```bash
python run_multimodel_preference.py \
  --config configs/multimodel_true_unknown_preference.yaml \
  --model deepseek-7b-chat \
  --gpus 0 \
  --resume \
  --stages reemergence_cache,reemergence_sweep,source_pathway,source_pathway_extended,reemergence_aggregate
```

For DeepSeek, `scripts/run_deepseek_reemergence.sh 0` is the equivalent convenience command. Change only `--model` to execute the same model-adapted workflow on another configured architecture.

The stages perform full-sequence caching, inner-CV ordered-INLP rank selection, moving answer-position erasure, immediate Attention/MLP recovery, matched causal/null donor patching, per-layer source attribution, BIC regime segmentation, and joint-fold bootstrapping. The complete canonical 100-C/100-D population is required; `--max-items` is deliberately rejected. For an engineering run, `--layers` denotes one-based re-emergence removal layers and `--folds` limits held-out folds. Restricted runs are not full scientific results.

Two output families are kept separate:

- `locked_layer_fold_pathway_results.csv`: historical Experiment-B `Residual / Context-V / Answer-Q` protocol.
- `extended_layer_fold_pathway_results.csv`: `Residual / Context-V / Context-K / Answer-Q / Answer-Gate` extension.

`Context-K` patches `k_proj` over task-relevant linguistic context positions. `Answer-Gate` patches the MLP gate projection at the answer position; for GLM's fused `gate_up_proj`, only the gate half is patched. Extended five-path shares never renormalize or overwrite locked three-path shares. See `validation/REEMERGENCE_HANDOFF_REPORT.md` and `reemergence/PROTOCOL_BOUNDARY.json` in a completed output directory.

## DeepSeek Telicity versus decision control

The missing DeepSeek causal comparison is packaged as a separate opt-in workflow. It reuses the completed full-depth cache and earlier mechanistic results, then runs only held-out matched-norm interventions on preregistered Telicity L19, continuous decision L30, the exact final readout, and controls. It never reruns behavior, encoding, erasure, re-emergence, Attention/MLP, source pathways, or decision decoding.

```bash
python scripts/run_deepseek_telicity_decision_control.py --gpus 0,1,2,3
```

The launcher is resumable: complete folds and valid prepared assets are reused, while aggregation and the CPU-only handoff audit are regenerated. A single GPU can be supplied with `--gpus 0`. The immutable continuous scorer remains `z(True_first_token) - z(Unknown_first_token)`.

See [the executable protocol](validation/DEEPSEEK_TELICITY_DECISION_CONTROL_PROTOCOL.md) for axis fitting, hook coordinates, doses, controls, seeds, statistical criteria, A–E rules, individual phase commands, and the output schema. The completed reference result and limitation are in `results/deepseek-7b-telicity-rq/telicity_decision_control/REPORT.md`.

## DeepSeek RQ2.3 and RQ2.4

The DeepSeek-specific reproducibility backend packages the completed causal-circuit and selective-correction experiments. It derives its layer/head candidates from outer-training data and the audited RQ2.2 artifacts; discovered layers and effect sizes are not stored in executable YAML. RQ2.3 uses actual-forward component interventions, matched random directions, same-state donor controls, and clean-restoration path mediation. RQ2.4 selects its internal Group-C gate on outer-training folds and reports held-out behavior against random-gate, oracle-C, unconditional, reverse, and exact-restore controls. Both retain the first-token score `z(True) - z(Unknown)`.

```bash
# Resume or reproduce both experiments; complete shards are reused.
python scripts/run_deepseek_rq23_rq24.py --gpus 0,1,2,3

# Resume only one question on one GPU.
python scripts/run_deepseek_rq23_rq24.py --gpus 0 --only rq23
python scripts/run_deepseek_rq23_rq24.py --gpus 0 --only rq24

# Read-only generic routes now reuse the audited official artifacts.
scripts/run_rq2.sh deepseek-7b-chat rq2.decision_circuit 0
scripts/run_rq2.sh deepseek-7b-chat rq2.selective_group_c_correction 0
```

The completed artifacts are under `results/deepseek-7b-rq2/rq2_3/` and `results/deepseek-7b-rq2/rq2_4/`. Each directory contains `summary.json`, `REPORT.md`, `audit.json`, fold shards, and GPU logs. The combined execution and interpretation report is `validation/DEEPSEEK_RQ23_RQ24_REPORT.md`. RQ2.3 establishes the earliest causal generator only within its tested intervention family and does not equate earlier predictivity with causality. RQ2.4's predefined selectivity rule and its statistical uncertainty are reported separately.

## DeepSeek RQ2.5 and RQ2.6

RQ2.5 is the completed 17,200-row Telicity-versus-decision-control experiment. Its launcher performs a read-only fail-closed audit and reuses the completed artifact. RQ2.6 applies the strict Qwen/Mistral A37 causal-origin boundary to DeepSeek: all Attention, MLP, and block-output residual coordinates before the independently validated generator are tested with outer-train C↔D source vectors, bidirectional held-out actual forwards, orthogonal norm-matched controls, head/token localization, and generator restoration mediation.

```bash
# Re-audit RQ2.5 and run/resume only missing RQ2.6 work.
python scripts/run_deepseek_rq25_rq26.py --gpus 0,1,2,3

# Read-only generic reuse after completion.
scripts/run_rq2.sh deepseek-7b-chat rq2.telicity_decision_relationship 0
scripts/run_rq2.sh deepseek-7b-chat rq2.true_driving_origin 0
```

RQ2.6 distinguishes predictor, generator-routed origin, parallel final-margin actuator, downstream propagation, and readout. A component is not called an origin merely because it is predictive or moves only one endpoint. See `validation/DEEPSEEK_RQ26_PROTOCOL.md`, `validation/DEEPSEEK_RQ25_RQ26_REPORT.md`, and `results/deepseek-7b-rq2/rq2_6/`.

## Legacy compatibility commands

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model mistral-7b-v03 --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model llama3.1-8b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model yi1.5-9b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model deepseek-7b-chat --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model gemma2-9b --gpus 0
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model glm4-9b --gpus 0
```

The full run is intentionally expensive: it covers every layer and all five held-out verb folds. The one-item smoke does not constitute a scientific result. For Mistral regression evidence and the boundary between the completed original result and the filtered generic equivalence run, see `validation/MISTRAL_PROTOCOL_REPORT.md`.

## Configuration contract and adding a model

Never change `fixed_experiment_config` between models: dataset/hash, A/B/C/D definitions, prompt text/semantics, folds, seed, scoring rule/position, intervention conditions, controls, selection procedure, stage order, and audit criteria are invariants. `model_configs` may contain only loading/tokenizer settings, architecture geometry, adapter/module mapping, dtype/device, and memory options.

To add a compatible decoder, add one model entry with official ID/revision/geometry and module paths. The reusable HF decoder adapter requires layers, attention, MLP, attention output projection input, input norm, Q/K/V projections, MLP gate projection, final norm, and LM head. A fused gate must declare its output slice, as the GLM entry does. Run tokenizer dry-run, weight-loading preflight, then a behavior/cache smoke before a full run. Do not insert discovered layer/head numbers.

See `validation/` for exact validation scope and access limitations. The canonical scoring evidence is `run_qwen_rq2bc_continuous.py` lines 35 and 91 (cached and live `logits[:,0]-logits[:,2]`) plus `run_qwen_full_depth_true_signal.py` line 61 (first token of each label).
