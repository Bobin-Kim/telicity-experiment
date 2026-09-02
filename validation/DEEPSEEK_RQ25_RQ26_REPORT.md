# DeepSeek RQ2.5 / RQ2.6 execution and interpretation report

Date: 2026-09-01
Model: `deepseek-ai/deepseek-llm-7b-chat`
Status: **COMPLETE — both audits PASS**

## RQ2.5 — Telicity representation versus decision control

The existing 17,200-row artifact was re-audited rather than recomputed. All 43 conditions have 400 rows, all five fold markers are complete, the cache hash matches, and `Δm = ΔTrue - ΔUnknown` holds exactly.

- L19 Telicity axis: Δm `-0.0004`, 95% CI `[-0.0099, +0.0091]`, `p=.9146`; causal success false.
- L30 decision-predictive axis: Δm `+0.0442`, 95% CI `[+0.0384, +0.0501]`, `p=3.04e-5`; 5/5 folds and all causal controls pass.
- Exact final True−Unknown readout: Δm `+2.9116`, 95% CI `[+2.7325, +3.0907]`, `p=1.44e-6`; 5/5 folds.

Classification remains **C. Separable representations**. Telicity is represented and predictive in places, but the preregistered L19 Telicity representation is not the controller of the final decision.

## RQ2.6 — origin of the Group-C True-driving signal

The new GPU run produced 17,200 coarse rows, 800 head rows, 2,000 token-source rows, and 500 generator-mediation rows. Live `nvidia-smi` showed GPUs 0–3 at approximately 15.2 GiB and high utilization during the scan. All folds and controls completed.

Attention L1 was the earliest held-out predictor of the later Attention-L15 generator (`r=.7473`, 95% CI `[.7294, .7652]`), but this is predictive evidence only.

No component in L1–L14 simultaneously passed both the bidirectional Attention-L15 generator criterion and the bidirectional final-margin criterion. Therefore the strict classification is **NO_GENERATOR_ROUTED_C_SPECIFIC_ORIGIN_IDENTIFIED**.

The endpoints separate in two informative ways:

- MLP L9 and L10 were bidirectional final-margin actuators outside the validated L15-generator route. L9 changed C/D by `-.0320` / `+.0352`; L10 by `-.0550` / `+.0564`, with both directional confidence intervals excluding zero. Their L15-generator criteria failed.
- MLP L14 was a reliable L15 router: generator-score changes were negative for C and positive for D with 5/5 folds and confidence intervals excluding zero. Its final-margin effects were `-.0522` / `+.0428`, but both intervals included zero.

The canonical diagnostic mediation therefore traced MLP L14. Its C→D random-adjusted margin effect was `-.0519`, 95% CI `[-.1874, +.0836]`, 3/5 directional folds. Restoring clean Attention L15 mediated `-.0234`, 95% CI `[-.0722, +.0253]`. Because total and mediated intervals include zero, the descriptive 45.2% ratio is not interpretable as established mediation.

Neither of the two selected heads nor any of five token categories passed the combined bidirectional generator-plus-margin rule. The target-predicate category changed the generator in the expected direction but did not reliably change the final margin.

The combined mechanistic interpretation is:

```text
early predictivity: Attention L1 (not causal evidence)
parallel/bypass margin actuation: MLP L9–L10
validated generator routing: MLP L14 → Attention L15
generator-routed Group-C origin: not localized under the strict test
downstream L30 decision-axis mediation: 88.2% (RQ2.3)
exact final readout control: +2.9116 per matched 1 SD (RQ2.5)
```

This supports a distributed/partially parallel origin account rather than a single localized generator-routed source. It does not support saying that L1 is a causal origin or that MLP L14 alone generates the behavioral bias.

## Audit and scoring

Both experiments use only `m = z(True_first_token) - z(Unknown_first_token)`, with True ID `8173` and Unknown ID `21483`. RQ2.6 mean actual/random norm mismatch was `0.0032%`, maximum `0.589%`; both are within the 1%/5% limits. No binary artifact was used.

Artifacts:

- RQ2.5: `results/deepseek-7b-telicity-rq/telicity_decision_control/`
- RQ2.6: `results/deepseek-7b-rq2/rq2_6/`
- protocol: `validation/DEEPSEEK_RQ26_PROTOCOL.md`
