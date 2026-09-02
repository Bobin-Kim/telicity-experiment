# DeepSeek 7B Chat validation — PASS (config/tokenizer dry-run)

Official `deepseek-ai/deepseek-llm-7b-chat` tokenizer resolved. Dataset, exact prompt, and folds passed; verbalizers are `True=[8173]`, `False=[13813]`, `Unknown=[21483]`, with distinct alternatives. Official Llama-compatible geometry (30 layers, hidden 4096, 32 attention/KV heads) is configured with the model's required remote code permission. Weights are not local, so no weight/hook smoke was performed.

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model deepseek-7b-chat --gpus 0
```
