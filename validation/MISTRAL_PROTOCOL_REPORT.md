# Mistral v0.3 original-protocol regression report — PASS

Date: 2026-08-31

## Conclusion

The generic runner's layer-0 failure was a regression relative to the completed original Mistral experiment, not a Mistral model failure. The Mistral compatibility path now reproduces the original behavior, representation coordinate, rank-1 axes, controls, and finite intervention protocol without changing the scoring rule.

The corrected GPU equivalence run completed with exit code 0 at:

`/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/results/mistral-7b-v03-original-equivalence`

This validation run deliberately covered layers 0, 16, 29, and 31 over all five held-out folds. It is marked `official: false` only because `--layers` was used as an engineering equivalence filter. The existing original Mistral experiments 3/4/5 result remains the complete 32-layer scientific result; its audit is PASS with 160/160 finite-layer jobs, zero missing jobs, and zero failed jobs:

`/home/bobin/imperfective_paradox_repro/results/Mistral_RQ2_experiments_3_4_5`

## GPU validation

- Python: `/data/sl_datasets/wlasl/Qwen2.5-1.5B-Instruct/venv/bin/python`
- PyTorch: `2.3.1+cu118`
- `torch.cuda.is_available()`: `True`
- GPU 0: `Tesla V100-PCIE-32GB`
- Live `nvidia-smi` observation: PID 845436 used GPU 0; observed VRAM rose from 14,892 MiB to 22,401 MiB and utilization reached 100%.
- GPUs 1–3 remained at 5 MiB and 0% during the observation.
- Run exit code: 0
- GPU run log: `validation/MISTRAL_ORIGINAL_EQUIVALENCE_RUN.log`
- Pre-regression failure log: `validation/MISTRAL_FULL_RUN_gpu0.log`

## Original implementation and artifacts

- Decision/localization: `/home/bobin/imperfective_paradox_repro/telicity_intervention/run_mistral_rq2_decision.py`
- Representation cache: `/home/bobin/imperfective_paradox_repro/telicity_intervention/run_mistral_rq2_reproducible.py`
- Full-depth finite intervention: `/home/bobin/imperfective_paradox_repro/telicity_intervention/run_mistral_rq2_finite_layer_sweep.py`
- Component screen/validation: `run_mistral_rq2_causal_screen.py`, `run_mistral_rq2_component_validation.py`
- Aggregate: `run_mistral_rq2_experiments345.py`
- Canonical residual cache: `results/mistral7b_rq2_reproducible/interaction/full_answer_layers.npz`
- Canonical margin/directions: `mistral_rq2_decision/rq2a_margin_signal/`
- Canonical behavior: `results/Mistral_RQ1_experiment_5/`
- Canonical interventions: `results/Mistral_RQ2_experiments_3_4_5/finite_layer_sweep/`

## Exact regression cause

The old generic cache registered a forward *pre-hook* on every decoder block. At layer 0 and the answer position this caches the common assistant-token embedding before the first block. It is identical across examples, so every feature has zero variance. The generic multi-component axis builder therefore produced no basis vectors and failed at `np.stack(basis)`.

The original Mistral cache instead uses `hidden_states[l+1]` at `attention_mask.sum(1)-1`. Therefore:

- layers 0–30 are decoder block outputs;
- layer 31 is the post-final-RMSNorm output;
- intervention hooks are forward hooks on decoder block output for layers 0–30 and on `model.model.norm` for layer 31.

The original layer-0 cache is not constant (maximum feature SD about 0.00306), so the standardized Ridge direction is defined. There is no Qwen/random fallback and no zero-variance scoring-rule substitution.

Two additional protocol mismatches were removed: the original loader chooses BF16 on this server and moves the model to one CUDA device, while the generic runner had used FP16 plus `device_map=auto`; and the original decision analysis consumes its frozen first-token margin artifact rather than recomputing that target inside localization.

## Original axis/control protocol retained

- Five outer verb-disjoint folds, 80 held-out items each.
- Standardize each hidden dimension on outer-train data; dimensions with SD `<1e-6` use SD 1.
- Rank-1 `Ridge(alpha=10)`; convert coefficient to activation space with `coef_/sd`; unit-normalize.
- Control RNG seed: `42 + 7300 = 7342`.
- One random direction orthogonal to the decision direction.
- Shared intervention norm: median train-only SD over 16 additional random unit axes, floored at `1e-4`.
- Conditions: `decision_pos`, `decision_neg`, `random_pos`, `random_neg`, packed together with batch size 8.
- Answer position: last non-padding prompt position, `attention_mask.sum(1)-1`.

## Numerical equivalence

| Quantity | Compared | Result | Maximum absolute difference |
|---|---:|---:|---:|
| Behavior True logit | 400 rows | 400/400 exact | 0 |
| Behavior Unknown logit | 400 rows | 400/400 exact | 0 |
| Behavior margin | 400 rows | 400/400 exact | 0 |
| Residual cache | 400×32×4096 | 52,428,800/52,428,800 exact | 0 |
| Frozen localization margin | 400 | 400/400 exact | 0 |
| Decision directions | 5×32×4096 | 655,360/655,360 exact | 0 |
| Random control directions | 5×32×4096 | 655,360/655,360 exact | 0 |
| Control scales | 5×32 | 160/160 exact | 0 |
| Control means | 5×32×4096 | 655,360/655,360 exact | 0 |
| Localization r²/Pearson/Spearman | 32 layers | same values to float64 precision | 1.11e-16 |
| Intervention keys and conditions | 4 layers × 5 folds × 320 rows | 6,400/6,400 exact | 0 |
| Intervention baseline/margin/delta/norm | 6,400 rows | every value exact | 0 |

The compared intervention layers cover the original failure point (0), first causal layer (16), strongest causal layer (29), and the separate final-norm hook (31). Localization reproduced first stable predictive layer 6 and strongest predictive layer 31.

All 32 layers' cached residuals, axes, random controls, scales, and means are bitwise identical. Together with exact GPU forward results at both decoder-output and final-norm hook coordinates, this establishes protocol equivalence for the regression repair. It does not claim that the filtered validation directory is a new full 160-job official result.

## Locked scoring rule

No scoring rule was changed.

- Tokenization: `True=[6878]`, `False=[9018]`, `Unknown=[1703,5485]`.
- Scoring IDs: `True=6878`, `Unknown=1703`.
- Estimand: `m = z(True_first_token) - z(Unknown_first_token)`.
- Unknown token 5485 is retained only in the tokenization audit and is never included in the score.

The scorer selects `ids[0]`, obtains both logits at the last non-padding answer position, and returns `t-u`. Behavior and all live intervention paths call this same scorer.

## Files changed

- `configs/multimodel_true_unknown_preference.yaml`: Mistral original-protocol compatibility profile and canonical artifact locations.
- `run_multimodel_preference.py`: deterministic setup, original single-device BF16 loading, stage-specific batches, and validation-only layer/fold filters.
- `src/pipeline_controller.py`: original residual coordinate, frozen margin source, rank-1 standardized Ridge, exact controls, output/final-norm hooks, and packed four-condition intervention.
- `requirements.txt`: explicit pandas/scikit-learn dependencies used by the protocol.

## Status boundary

- Regression repair: PASS.
- Scoring invariant: PASS.
- Actual GPU 0 execution: PASS.
- Behavior/cache/localization equivalence: PASS.
- Representative finite-intervention equivalence: PASS, exact across 6,400 rows.
- Existing original full Mistral experiments 3/4/5: PASS, complete.
- Corrected generic runner re-execution of every 32×5 intervention cell and every component/supplemental stage: not performed in this validation fixture; no claim is made otherwise.
