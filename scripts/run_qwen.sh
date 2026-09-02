#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model qwen2.5-7b --gpus "${1:-0}"
