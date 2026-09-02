# Qwen equivalence report — PASS

Scope: deterministic source-contract equivalence plus live one-item/full-hook smoke; not a completed 400-item scientific rerun.

The fixture `run_qwen_equivalence.py` passed every comparison with zero numerical difference for shared ridge predictions and downstream R² aggregation (`atol=1e-12`). It verified the 400 canonical IDs/order, exact rendered prompt, last non-padding prompt position, True/False/Unknown scoring IDs `[2514, 4049, 13790]`, verb-stratified five folds, seed 42, all 28 layers, canonical Qwen hook paths, six full-depth conditions, and margin formula. The live weight smoke successfully traversed every residual, attention, MLP, and head-output hook and produced finite preference output.

Schema correspondence is explicit: `predictive_full_depth.csv` covers fold × layer; `causal_full_depth.csv` covers example × fold × layer × the six canonical conditions; component/head files cover the complete depth; completion markers record coverage and the run fingerprint. No discovered layer/head is configured.

An additional, non-authoritative comparison to the historical upstream A21 cache was attempted. Its first 16 cached margins did not reproduce under the current weights/runtime (maximum absolute difference 5.625), despite exact prompt/data/token agreement. That cache has no model-weight or runtime fingerprint and predates this package, so it was not accepted as a regression oracle or reused as a protocol-equivalent result. The equivalence PASS is against the current canonical source contract and deterministic fixture, not that unversioned historical artifact.

Commands executed:

```bash
python tests/test_core.py
python validation/run_qwen_equivalence.py
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --dry-run
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --preflight-only --output-dir validation/runtime_qwen_preflight
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0 --stages behavior,cache --max-items 1 --batch-size 1 --output-dir validation/runtime_qwen_smoke
```

Full run:

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus 0
```
