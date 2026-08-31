# Mistral protocol report — PASS (adapter/protocol smoke)

This is a smoke validation, not completed scientific execution. Local `mistralai/Mistral-7B-Instruct-v0.3` weights loaded in FP16 on a V100. Live preflight resolved all 32 transformer blocks, all attention and MLP modules, final norm, and LM head. A one-item run successfully executed behavior and cache collection through every residual, attention, MLP, and head-output hook with finite output and compatible schemas.

Tokenization is `True=[6878]`, `False=[9018]`, `Unknown=[1703,5485]`. The locked canonical estimand compares IDs 6878 and 1703 at the first answer position. They are distinct. The full `Unknown` sequence is retained in the tokenization audit, but no stage substitutes sequence likelihood: all stages receive the same shared first-token scorer. Historical mixed-scoring Mistral results were not reused.

The smoke validates complete layer coverage and hook/output compatibility. Full fold controls, causal sweeps, component audits, and aggregation are implemented but were not executed because they require the expensive 400-item full-depth run.

Full run:

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model mistral-7b-v03 --gpus 0
```
