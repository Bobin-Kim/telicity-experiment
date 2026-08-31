#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python run_multimodel_preference.py --config configs/multimodel_true_unknown_preference.yaml --model mistral-7b-v03 --gpus "${1:-0}"
