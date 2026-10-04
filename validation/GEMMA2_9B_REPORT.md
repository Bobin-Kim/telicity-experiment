# Gemma 2 9B validation — COMPLETE, OFFICIAL

Official entry: `google/gemma-2-9b-it`; reusable HF decoder mapping; 42 layers, hidden size 3584, 16 attention heads, 8 KV heads, head dimension 256. Gated access resolved; full run executed on GPU 3 of `nlpgpu7.kaist.ac.kr` under branch `antonio/llama-gemma-run`.

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model gemma2-9b --gpus 3 --preflight-only
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model gemma2-9b --gpus 3
```

`RUN_COMPLETE.json`: `"official": true`, all 7 stages complete (`behavior`, `cache`, `predictive_localization`, `causal_intervention`, `component_analysis`, `generator_localization`, `aggregate`), fold profile `confirmatory` (all 5 folds).

## Behavior (continuous True/Unknown margin, `m = z_True - z_Unknown`)

Group A (gold `False`) is not scorable under this margin, since the continuous scorer only compares True vs Unknown logits.

| Group | Gold | n | ACC | True-bias rate | mean margin |
|---|---|---|---|---|---|
| B | True | 100 | 0.970 | 0.970 | +1.115 |
| C | Unknown | 100 | **0.170** | **0.820** | +0.625 |
| D | True | 100 | 0.940 | 0.940 | +0.782 |

Group C failure is severe but somewhat less extreme than Llama-3.1-8B (ACC_C 0.17 vs 0.07) and than Qwen's reference ACC_C = 0.41 is actually higher (less biased) than both new models — Gemma sits between Llama and Qwen in severity.

## Localization

- **Predictive localization**: peak layer 41/42 (final layer, R² ≈ 0.99) across all 5 folds — even later than Llama's peak, consistent with the late-layer concentration pattern.
- **Earliest reliable causal generator**: **Attention at layer 34/42** (81% depth — same depth fraction as Llama's MLP finding, 26/32). Mean control-adjusted contrast 0.2376 (95% CI [0.0916, 0.3835], p = 0.011, 5/5 folds positive) — a larger, cleaner effect than Llama's.
- Component type differs from Llama/Qwen/DeepSeek's MLP-first pattern — Gemma's earliest reliable causal component is **attention**, not MLP. Worth flagging as a genuine cross-architecture difference, not a bug (both component types passed the same audit criterion).

## Known pipeline bugs hit and fixed on this branch (affect the shared `src/pipeline_controller.py` / `src/output_validation.py`, not Gemma-specific)

Identical three bugs as documented in `LLAMA31_8B_REPORT.md` — reproduced independently on this model too, confirming they are architecture-agnostic pipeline issues rather than Llama- or Gemma-specific adapter problems. See that report for details and see commits on `antonio/llama-gemma-run`.

## Output location

`results/gemma2-9b/` on `nlpgpu7.kaist.ac.kr:/media/sdd1/antonio_work/telicity-experiment/`, plus CSV/JSON summaries mirrored locally (not the raw `preference_cache.npz`).
