#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model="${1:?model key required: llama3.1-8b|yi1.5-9b|deepseek-7b-chat|gemma2-9b|glm4-9b}"
gpu="${2:-0}"
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model "$model" --gpus "$gpu"
