# GLM 4 9B HF validation — PASS (config/tokenizer dry-run)

Official current HF-converted entry `zai-org/glm-4-9b-chat-hf` resolved at the pinned configuration revision. Dataset, exact prompt, and folds passed; verbalizers are `True=[2514]`, `False=[4049]`, `Unknown=[13781]`, with distinct alternatives. Official geometry is 40 layers, hidden size 4096, 32 attention heads, and 2 KV heads. Weights are not local, so no live hook smoke was performed.

```bash
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model glm4-9b --gpus 0
```
