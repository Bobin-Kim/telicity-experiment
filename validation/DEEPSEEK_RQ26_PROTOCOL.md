# DeepSeek RQ2.6 — causal-origin protocol

The sole target is the first-answer-token margin `m = z(True) - z(Unknown)` with DeepSeek token IDs True `8173` and Unknown `21483`. No binary or full-verbalizer target is executable.

The experiment applies the strict common Qwen/Mistral A37 protocol to the independently audited DeepSeek circuit. RQ2.2 fixes the late L30 decision coordinate and RQ2.3 fixes Attention L15 as the earliest tested causal generator. Those findings are read from audited result files, not copied into executable YAML.

For every held-out verb fold, candidate sources are scanned at Attention, MLP, and block-output residual coordinates in L1–L14. The source vector is the other-four-fold mean state difference `D-C`. C examples receive `D-C`; D examples receive its reverse. Every actual intervention has an outer-train seeded orthogonal random vector with the same L2 norm and BF16 realized-norm calibration.

A component is an upstream causal source only when all of the following hold:

- C→D reduces the validated generator score with a fold-t 95% CI below zero and at least 4/5 directional folds;
- D→C increases that score under the same rule;
- the corresponding random-adjusted final-margin effects pass both directions;
- all forwards are held-out, actual model executions.

Predictive source correlation is reported separately and is never causal evidence. The earliest strict source is ordered by depth, then Attention, MLP, and residual within a block. If nothing qualifies, the strongest generator-changing component is traced only as a diagnostic and the classification remains `NO_CAUSAL_C_SPECIFIC_ORIGIN_IDENTIFIED`.

The selected component receives two outer-train head candidates, five token-source categories, and source→Attention-L15 clean-restoration mediation. Downstream transmission and exact output readout are reused from audited RQ2.3 and RQ2.5 rather than recomputed.

Expected new rows:

- coarse component scan: 17,200;
- head scan: 800;
- token-source scan: 2,000;
- generator mediation: 500.

Run or resume:

```bash
python scripts/run_deepseek_rq25_rq26.py --gpus 0,1,2,3
```

RQ2.5 is first re-audited read-only. Complete RQ2.6 fold shards are reused.
