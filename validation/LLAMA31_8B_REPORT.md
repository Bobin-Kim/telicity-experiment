# Llama 3.1 8B validation — COMPLETE, OFFICIAL

Official entry: `meta-llama/Meta-Llama-3.1-8B-Instruct`; reusable HF Llama decoder mapping; 32 layers, hidden size 4096, 32 attention heads, 8 KV heads. Gated access resolved; full run executed on GPU 2 of `nlpgpu7.kaist.ac.kr` under branch `antonio/llama-gemma-run`.

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model llama3.1-8b --gpus 2 --preflight-only
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model llama3.1-8b --gpus 2
```

`RUN_COMPLETE.json`: `"official": true`, all 7 stages complete (`behavior`, `cache`, `predictive_localization`, `causal_intervention`, `component_analysis`, `generator_localization`, `aggregate`), fold profile `confirmatory` (all 5 folds).

## Behavior (continuous True/Unknown margin, `m = z_True - z_Unknown`)

Group A (gold `False`) is not scorable under this margin, since the continuous scorer only compares True vs Unknown logits.

| Group | Gold | n | ACC | True-bias rate | mean margin |
|---|---|---|---|---|---|
| B | True | 100 | 0.910 | 0.910 | +1.297 |
| C | Unknown | 100 | **0.070** | **0.930** | +1.348 |
| D | True | 100 | 0.990 | 0.990 | +1.333 |

Group C failure is severe — more extreme than the Qwen baseline (ACC_C = 0.41 in the reference notebook), closer to the Mistral baseline (ACC_C ≈ 0.01).

## Localization

- **Predictive localization** (decision-predictive R², inner-CV ridge): peak layer 23–28 across folds (R² ≈ 0.99), i.e. ~85% network depth — consistent with the late-layer concentration seen in Qwen/Mistral/DeepSeek.
- **Earliest reliable causal generator** (`generator_selection.json`, actual forward decision contrast vs. matched random contrast, 95% fold-t CI, ≥4/5 positive folds): **MLP at layer 26/32** (81% depth). Mean control-adjusted contrast 0.0746 (95% CI [0.0135, 0.1357], p = 0.027, 5/5 folds positive).
- Component type (MLP) matches Qwen's L20 MLP and DeepSeek's L16 MLP circuits — same causal component class, different absolute layer (expected across architectures).

## Known pipeline bugs hit and fixed on this branch (affect the shared `src/pipeline_controller.py` / `src/output_validation.py`, not Llama-specific)

1. `_axis()` crashed (`ValueError: need at least one array to stack`) on layer 0, whose residual (block-input coordinate) is the embedding of the fixed chat-template suffix token — identical across all items, zero variance. Fixed to return a well-defined zero-effect result instead of crashing.
2. `atomic_json()` crashed on numpy scalar types (`TypeError: Object of type int64 is not JSON serializable`) inside `generator_localization`. Fixed generically via a `default=` handler for any numpy scalar/array.
3. `generator_localization`/`aggregate` t-test computation produced NaN when a layer's fold-means had exactly zero variance (a direct consequence of fix #1 zeroing out degenerate layers), rejected by `write_csv`'s NaN/Inf guard. Fixed by treating a zero standard error as an undefined statistic (reported blank, matching the existing convention for insufficient folds) instead of computing through it.

See commits on `antonio/llama-gemma-run`. These bugs are architecture-agnostic (identically reproduced on Gemma-2-9B) and will recur for any future model run through the generic pipeline unless already merged to `main`.

## Output location

`results/llama3.1-8b/` on `nlpgpu7.kaist.ac.kr:/media/sdd1/antonio_work/telicity-experiment/`, plus CSV/JSON summaries mirrored locally (not the raw `preference_cache.npz`).
