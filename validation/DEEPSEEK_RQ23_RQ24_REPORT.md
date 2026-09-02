# DeepSeek RQ2.3 / RQ2.4 execution and interpretation report

Date: 2026-09-01
Model: `deepseek-ai/deepseek-llm-7b-chat` (registered alias `deepseek-7b-chat`)
Status: **COMPLETE — both audits PASS**

## Execution evidence

The resumable launcher `scripts/run_deepseek_rq23_rq24.py --gpus 0,1,2,3` completed all five folds. Live `nvidia-smi` checks during the component, path, and correction workers showed the Python processes on GPUs 0–3, approximately 15–19 GiB allocated per active V100, and high GPU utilization. Worker completion files report exit success for every fold. After aggregation all GPUs returned to 5 MiB idle allocation and 0% utilization.

The experiment reused the audited DeepSeek full-depth cache at `results/deepseek-7b-telicity-rq/full_depth_cache.npz` (SHA-256 `cb28afb50fbab39262710e10f26b160111ec69215c1c17002ef5d01eb40ab8ce`). The added answer-position head-input cache passed its own audit and reproduces the baseline margin exactly. Its SHA-256 is `cc83295c72da3388d401ae57d70aadca6c10ef6013dd6ec73505b7a287acc95a`.

No binary protocol was used. Both audits record the unchanged first-answer-token score `m = z(True_first_token) - z(Unknown_first_token)`. DeepSeek token IDs were True `8173` and Unknown `21483`.

## RQ2.3 — decision circuit / causal pathway

The earliest tested actual-forward component that passed the held-out causal rule was **Attention L15**: matched-random-adjusted Δm `+0.2203`, fold-t 95% CI `[+0.0479, +0.3926]`, `p=.0239`, positive in 5/5 folds. This follows the earlier RQ2.2 predictive appearance at L14, so the result supports a predictive-before-causal distinction. It does not prove that no distributed causal source exists before L15.

Additional reliable actual-forward effects include Attention L16 (`+0.9570`), Attention L17 (`+1.1009`), MLP L16 (`+0.5656`), L16:H7 (`+0.5022`), L17:H6 (`+0.1657`), and L17:H12 (`+0.2809`); all listed confidence intervals exclude zero. The stable downstream shortlist included Attention L20, L20:H12, and L23:H0.

For the matched Attention-L15 donor path, the unblocked source effect was `+0.1502` with 95% CI `[+0.1093, +0.1911]`. Joint clean restoration of the three downstream components mediated `+0.0263`, or 17.5% of that effect (95% CI `[+0.0050, +0.0476]`). L20:H12 and L23:H0 were individually supported mediators; Attention L20 alone was directionally positive but its individual interval crossed zero. Because these are restoration blocks, the experiment identifies partial mediators but does not by itself establish their serial order.

Restoring the L30 decision axis mediated 88.2% of the source effect (`+0.1325`, 95% CI `[+0.0946, +0.1704]`). Full L30-state restoration mediated 100% by construction. The mechanistic reading is therefore a sparse, statistically supported downstream contribution embedded in a much broader late-layer decision state.

Artifacts: `results/deepseek-7b-rq2/rq2_3/`. The main files are `REPORT.md`, `summary.json`, `component_summary.csv`, `component_interventions.csv`, `path_summary.csv`, `path_mediation.csv`, `audit.json`, and `logs/`.

## RQ2.4 — selective Group-C correction

Baseline accuracy was A `.05`, B `.89`, C `.00`, and D `1.00`. Outer-training selection chose the internal hard gate. On held-out items it changed Group-C accuracy from `.00` to `.02` (two corrected items; Δ `+.02`, item-bootstrap 95% CI `[.00, .05]`) while A, B, and D accuracy were unchanged. Overall accuracy moved from `.485` to `.490`.

This passes the predefined selectivity rule, but it is **not a statistically robust behavioral correction**: the Group-C interval includes zero and the effect is only 2/100 items. The oracle-C control also corrected only 2/100 items at the fixed intervention ceiling, showing that actuator strength / distance from the decision boundary, rather than gate specificity alone, limits the result. The unconditional control achieved the same C gain but harmed A by 4 points and B by 15 points; the random gate harmed A by 1 point and B by 6 points. Exact restore produced zero changes, as required.

The appropriate conclusion is that the internal gate is behaviorally selective under the preregistered criterion, while practical correction efficacy is weak and uncertain. It should not be reported as a robust solution to Group-C errors.

Artifacts: `results/deepseek-7b-rq2/rq2_4/`. The main files are `REPORT.md`, `summary.json`, `metrics_with_ci.csv`, `selective_interventions.csv`, `gate_selections.csv`, `selected_hyperparameters.csv`, `audit.json`, and `logs/`.

## Reproduction and audit

```bash
python scripts/run_deepseek_rq23_rq24.py --gpus 0,1,2,3
python -m unittest tests.test_deepseek_rq23_rq24_contract
python scripts/build_rq_manifests.py --verify-only
```

The launcher reuses complete fold shards and regenerates only missing or invalid work. Generic RQ2 routes return these audited artifacts without starting GPU work:

```bash
scripts/run_rq2.sh deepseek-7b-chat rq2.decision_circuit 0
scripts/run_rq2.sh deepseek-7b-chat rq2.selective_group_c_correction 0
```
