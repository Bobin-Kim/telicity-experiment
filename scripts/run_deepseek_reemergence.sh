#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Complete moving-erasure plus locked and extended source-path analysis.
# The locked Q/V/Residual result and the new Q/K/V/Gate/Residual extension are
# written to different artifacts and are never normalized together.
python run_multimodel_preference.py \
  --config configs/multimodel_true_unknown_preference.yaml \
  --model deepseek-7b-chat \
  --gpus "${1:-0}" \
  --resume \
  --stages reemergence_cache,reemergence_sweep,source_pathway,source_pathway_extended,reemergence_aggregate
