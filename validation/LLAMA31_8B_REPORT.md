# Llama 3.1 8B validation — BLOCKED BY GATED ACCESS

Official entry: `meta-llama/Meta-Llama-3.1-8B-Instruct`; reusable HF Llama decoder mapping; 32 layers, hidden size 4096, 32 attention heads, 8 KV heads. The dry-run stopped fail-closed at tokenizer access with HTTP 401. No credential is embedded. After the user accepts the model license and runs `huggingface-cli login`, execute:

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model llama3.1-8b --gpus 0 --preflight-only
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model llama3.1-8b --gpus 0
```
