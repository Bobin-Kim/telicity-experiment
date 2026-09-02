# DeepSeek Pre-RQ Readiness

## Decision

**NOT READY**

DeepSeek model loading, locked-input validation, shared True–Unknown scoring, the 400-item behavioral forward run, and architecture-hook smoke checks passed. The readiness gate cannot return `READY FOR RQ PILOT` because the finalized generalized pipeline defines no `Delta_AA` formula and its official behavioral stage does not emit predicted/correct labels or an aggregate metric summary. No replacement formula or scientific threshold was introduced.

## Model and architecture

- Model/tokenizer ID: `deepseek-ai/deepseek-llm-7b-chat`
- Resolved local path: `/data/sl_datasets/models/deepseek-ai/deepseek-llm-7b-chat`
- Snapshot revision: `afbda8b347ec881666061fa67447046fc5164ec8`
- Load configuration: FP16, `device_map: auto`, all parameters on `cuda:0`
- Resolved architecture: `LlamaForCausalLM`; 30 layers, hidden size 4096, 32 attention heads, 32 KV heads
- Modules: `model.layers`, `self_attn`, `mlp`, block-input residual, `model.norm`, and `lm_head`; no Qwen- or Mistral-specific module path was active

## Tokenizer and shared scorer

| Alternative | Exact form | Token string | Token ID | Tokens |
|---|---|---|---:|---:|
| True | `True` | `True` | 8173 | 1 |
| Unknown | `Unknown` | `Unknown` | 21483 | 1 |

- Neither alternative has a leading-space token; capitalization is exact title case.
- Representative rendered prompt: 165 tokens; scoring position 164, the last non-padding prompt token immediately after `Assistant:`.
- Exact shared rule: `FirstAnswerTokenMargin`, `m = z_True - z_Unknown`.
- Behavioral execution, hook smoke, and RQ controller all resolve the same shared scorer; no DeepSeek-specific formula was added.

## Locked inputs

- Dataset: `/home/bobin/imperfective_paradox_repro/assets/imperfectiveNLI.json`
- SHA-256: `ddd01d06b18b9bab24abb5bc92a8eb740bb145123658fa9a10bd86e38b4bdb5a`
- IDs: exactly 400 unique ordered IDs, `A_001` through `D_100`
- Groups: A/B/C/D = 100/100/100/100; assignments and order unchanged
- Prompt-contract SHA-256 passed; system/user strings and tokenizer chat-template rendering were unchanged
- Seed 42 and the fixed five verb folds were unchanged; fold counts were 80 each

## Behavioral baseline

The values below use the scorer's locked single-token label logits and the pre-existing group-accuracy/TBR formulas. `Delta_AA` is unavailable because no formula exists in the finalized generalized or canonical continuous-preference path.

| Metric | Value |
|---|---:|
| Acc_A | 0.050 |
| Acc_B | 0.890 |
| Acc_C | 0.000 |
| Acc_D | 1.000 |
| TBR_C | 1.000 |
| Delta_AA | **Not defined** |

Raw output: `results/deepseek-7b-chat-pre-rq-readiness/behavior.csv`
Aggregate validation: `results/deepseek-7b-chat-pre-rq-readiness/behavior_summary.json`

## Group C preference

| Statistic | Value |
|---|---:|
| n | 100 |
| Mean margin | 4.9200 |
| Median margin | 4.8516 |
| Sample SD | 0.9391 |
| Minimum | 1.8125 |
| Maximum | 8.1875 |
| Fraction m > 0 | 1.000 |
| Fraction m < 0 | 0.000 |
| Exact fraction m = 0 | 0.000 |

The pipeline defines no nonzero numerical tolerance or formal RQ-eligibility threshold. Exact equality is reported for the zero fraction, and scientific eligibility therefore remains **review required** rather than being assigned a post-hoc threshold. Descriptively, Group C has a strong and universal positive True-driving preference.

## Smoke and output validation

- One-item layer-0 hook shapes: residual, attention output, MLP output, and head-output input were all `[1, 165, 4096]`.
- Final norm and output head resolved; hooks were removed cleanly with zero registered smoke handles remaining.
- One forward pass and all behavioral outputs were finite; no NaN/Inf occurred.
- All 400 IDs occurred exactly once and preserved A/B/C/D membership.
- Recomputed aggregate values matched the saved validation summary.
- Same-batch deterministic reproduction on 8 items had maximum absolute margin error `0.0`.
- The official generalized behavior stage completed, but its native schema contains only index/ID/group/fold/True logit/Unknown logit/margin. Required prediction, gold-label, correctness, and aggregate fields had to be produced by the readiness validation; this schema gap blocks an official READY decision.
- No binary experiment or binary artifact was invoked. No Qwen/Mistral result was used as an expected value. Only the `behavior` stage ran; no cache, localization, intervention, component, generator, aggregate-RQ, or A37 stage started.

## Engineering change and resources

- Modified `configs/multimodel_true_unknown_preference.yaml` only: added the existing DeepSeek local path and pinned the downloaded snapshot revision. No scientific setting changed.
- Official behavior-stage wall time: 30.69 s. Additional scorer/hook/output validation: 24.35 s; deterministic subset recheck: approximately 10 s.
- Peak GPU memory allocated: 15,218,654,720 bytes (14.17 GiB); peak reserved: 16,229,859,328 bytes (15.12 GiB).

To become ready, the finalized generalized package must supply the pre-registered `Delta_AA` definition and official behavioral prediction/summary schema without changing the locked scorer, prompt, data, or scientific thresholds.
