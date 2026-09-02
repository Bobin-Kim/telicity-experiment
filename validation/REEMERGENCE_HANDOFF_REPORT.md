# Re-emergence mechanism handoff validation

Status: **IMPLEMENTED; LIVE DEEPSEEK GPU PREFLIGHT PASS**

## What is now reproducible

The generic runner exposes the complete supplemental workflow as five explicit,
resumable stages:

1. `reemergence_cache`: caches full-sequence hidden states for the canonical C/D
   subset and records linguistic context/answer token coordinates.
2. `reemergence_sweep`: fits training-only ordered INLP bases, selects erasure
   rank by inner verb-disjoint cross-validation, performs real answer-position
   erasure, and records immediate Attention/MLP recovery.
3. `source_pathway`: runs the locked Experiment-B decomposition with Residual,
   Context-V, and Answer-Q causal donors and same-label matched controls.
4. `source_pathway_extended`: separately runs Context-K and Answer-Gate in
   addition to the locked pathways.
5. `reemergence_aggregate`: checks the complete layer/fold/path grid, applies
   the verified-erasure rule, computes layer profiles, BIC contiguous regimes,
   and joint-fold bootstrap support.

The existing first-answer-token preference scorer is used unchanged. This
supplemental workflow studies telicity re-emergence inside the same fixed prompt
and dataset; it does not replace the continuous True–Unknown estimand.

## Scientific boundary

- Locked result: `Residual / Context-V / Answer-Q`. This is the only output that
  may be compared to the historical Qwen/Mistral/DeepSeek Experiment-B result.
- Extension: `Residual / Context-V / Context-K / Answer-Q / Answer-Gate`.
  Context-K patches `k_proj` at linguistic context positions. Answer-Gate
  patches the MLP gate projection at the answer position. The GLM fused
  `gate_up_proj` adapter patches only its gate half.
- The extended five-path shares never renormalize or overwrite the locked
  three-path shares. Extended output has no historical equivalence claim until
  it has been run and audited on GPU.

## Validation performed

- Core deterministic tests: PASS.
- DeepSeek tokenizer/config dry-run with the complete stage chain: PASS.
- Empty-weight architecture resolution for local Qwen, Mistral, and DeepSeek:
  Q, K, V, and gate hook modules all resolve to real projection layers.
- Live DeepSeek full-weight load on GPU 0: PASS. All 30 input norms and all 30
  Q/K/V/Gate projection hooks resolved; GPU memory was released afterward.
- Existing DeepSeek locked result grid: 405/405 rows present
  (27 layers × 5 folds × 3 paths).
- Existing locked raw SHA-256:
  `df35f9e72cf4f097786cc28a221ce8151390de8c1a30e0e2f8004ba677b6e5ed`
  is re-computed by `audit_reemergence_handoff.py`; the JSON audit is the
  authoritative value if this report and artifact ever disagree.

The extended K/G pathway has not yet been executed over every DeepSeek layer;
the code, full checkpoint load, and hook contracts are validated, but no new
K/G scientific result is claimed in this handoff.

## Commands

Complete DeepSeek workflow:

```bash
scripts/run_deepseek_reemergence.sh 0
```

Equivalent generic command for any configured compatible model:

```bash
python run_multimodel_preference.py \
  --config configs/multimodel_true_unknown_preference.yaml \
  --model deepseek-7b-chat \
  --gpus 0 \
  --resume \
  --stages reemergence_cache,reemergence_sweep,source_pathway,source_pathway_extended,reemergence_aggregate
```

The workflow requires the complete 100-C/100-D subset. `--max-items` is
therefore rejected. Engineering validation can restrict one-based removal
layers and held-out folds with `--layers` and `--folds`; such runs are not full
scientific results.
