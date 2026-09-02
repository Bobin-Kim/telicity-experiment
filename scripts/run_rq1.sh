#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_rq1.sh MODEL_ALIAS [GPU_IDS] [--include-extension]" >&2
  exit 2
fi
model=$1
gpus=${2:-0}
shift $(( $# >= 2 ? 2 : 1 ))
python -m src.rq1.runner --model "$model" --gpus "$gpus" "$@"
