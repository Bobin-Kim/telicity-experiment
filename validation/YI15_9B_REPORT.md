# Yi 1.5 9B validation — PASS (config/tokenizer dry-run)

Official `01-ai/Yi-1.5-9B-Chat` tokenizer resolved. Exact prompt rendered; dataset/folds passed; verbalizers are `True=[11279]`, `False=[14761]`, `Unknown=[44854]`, with distinct scoring tokens. Official Llama-compatible geometry (48 layers, hidden 4096, 32 heads, 4 KV heads) is configured. Weights are not local, so no weight/hook smoke was performed.

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model yi1.5-9b --gpus 0
```
