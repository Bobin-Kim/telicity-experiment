# DeepSeek Small End-to-End Pilot

## Decision

**ENGINEERING PILOT PASS — FULL RQ PIPELINE NOT YET GENERALLY READY**

The selected DeepSeek path completed from locked prompt rendering through shared preference scoring, activation capture, actual-forward intervention, random controls, aggregation, and audit. This pilot is deliberately too small for scientific interpretation.

## Pilot design

- Model: `deepseek-ai/deepseek-llm-7b-chat`, revision `afbda8b347ec881666061fa67447046fc5164ec8`
- Shared scorer: `FirstAnswerTokenMargin`, `m = z_True - z_Unknown`
- Training fixture: 12 items, three from each A/B/C/D group, drawn outside held-out fold 0
- Held-out fixture: 8 items, two from each A/B/C/D group, all in fold 0
- Representative layers: L1, L15, L29
- Components: block-input residual, attention output, MLP output
- Conditions: preference `+1 SD`, preference `−1 SD`, orthogonal-random `+1 SD`, orthogonal-random `−1 SD`
- All directions and scales used only the 12 pilot-training items; held-out items were used only for forward evaluation

L0 was initially considered as the early layer, but its answer-position block input is the same uncontextualized token embedding across these prompts and therefore has zero usable preference-axis variance. L1 was used as the first contextualized early-layer representative. No partial output from the failed L0 attempt was retained.

## Execution result

| Check | Result |
|---|---:|
| Preflight | PASS |
| Activation tensors | 9/9 captured |
| Activation shape | `[20, 165, 4096]` for every layer/component |
| Expected intervention rows | 288 |
| Observed unique intervention rows | 288 |
| Aggregate rows | 45 |
| NaN/Inf | 0 |
| Hooks remaining after removal | 0 |
| Actual forwards | PASS |
| Random controls | PASS |
| Independent raw-to-aggregate recomputation | PASS |
| Full RQ or binary stage started | No |
| Runtime | 21.75 s |
| Peak allocated GPU memory | 15.92 GiB |

The intervention path produced finite, non-identical outputs and the random-control path executed at the same component/layer steering scale. The signs and magnitudes are not interpreted because this is a one-fold, eight-item engineering fixture.

Artifacts:

- `results/deepseek-7b-chat-e2e-pilot/pilot_behavior.csv`
- `results/deepseek-7b-chat-e2e-pilot/pilot_interventions.csv`
- `results/deepseek-7b-chat-e2e-pilot/pilot_aggregation.csv`
- `results/deepseek-7b-chat-e2e-pilot/pilot_audit.json`

## Generalization assessment

### What generalizes successfully

- The shared HF decoder adapter resolves DeepSeek's Llama-compatible layers, attention, MLP, residual stream, final norm, output head, and `o_proj` head-input coordinate without a DeepSeek-specific hook implementation.
- The same hook and intervention implementation worked at early, middle, and late depths and for residual, attention, and MLP coordinates.
- The shared scorer and right-padding answer-position convention worked unchanged.
- Configuration-based geometry checks correctly validated 30 layers, hidden size 4096, and 32 attention heads.
- The adapter contract is plausibly reusable for Qwen2, Mistral, Llama/Yi, DeepSeek, Gemma2, and GLM implementations that expose the configured HF module paths.

### What is not yet generally safe

1. **Degenerate-layer handling:** DeepSeek L0 has zero answer-position residual variance. The full generic `_axis` implementation can reach `np.stack(basis)` with an empty basis, so an exhaustive all-layer causal run can fail even though L1 onward works.
2. **Pilot subset selection:** native `--max-items` takes the first N canonical rows, which produces an A-only fixture rather than a balanced A/B/C/D pilot.
3. **Layer/fold scoping:** the generic predictive, causal, and component paths contain hard-coded `range(5)` and all-layer loops. `--layers` and `--folds` are not consistently honored outside the Mistral compatibility path, so they cannot safely guarantee a small generic run.
4. **Architecture-specific behavior remains:** Mistral already requires a dedicated compatibility profile for residual coordinates, loading, controls, and batching. The package is configuration-driven, but not fully architecture-neutral.
5. **Behavioral readiness gap:** the official generic behavior stage still lacks predicted/correct labels, aggregate metrics, and a defined `Delta_AA`, as recorded in the pre-RQ readiness report.
6. **Unvalidated registry entries:** Llama 3.1, Yi, Gemma2, and GLM have registry entries but no configured local snapshots in the active YAML, so their end-to-end compatibility has not been established. Gemma2/GLM should not be declared supported from static module names alone.
7. **Device assumptions:** controller input placement follows the first parameter device; this is safe for the tested single-GPU 7B run but needs explicit validation for genuinely sharded `device_map: auto` models.
8. **Tokenizer contract:** every new model must independently pass the exact True/Unknown verbalizer and answer-position audit; registry membership alone is insufficient.

## Overall judgment

The code is **successfully generalized at the adapter/scorer/hook level for DeepSeek's Llama-compatible architecture**, and the small actual-forward path is connected correctly. It is **not yet a universally safe full-pipeline implementation across arbitrary registered models**. Before a DeepSeek full RQ pilot or another-model claim, the core runner should add fail-closed degenerate-axis handling, balanced engineering sampling, and consistent requested-layer/requested-fold propagation, then repeat this same small pilot for each architecture family.
