# Gemma 2 9B validation — BLOCKED BY GATED ACCESS

Official entry: `google/gemma-2-9b-it`; reusable HF decoder mapping; 42 layers, hidden size 3584, 16 attention heads, 8 KV heads, head dimension 256. Dry-run stopped fail-closed with HTTP 401. Accept Google's license and authenticate, then run:

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model gemma2-9b --gpus 0 --preflight-only
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model gemma2-9b --gpus 0
```
