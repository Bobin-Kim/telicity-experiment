#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]]; then
  echo "usage: scripts/run_rq2.sh MODEL_ALIAS RQ2_SUBQUESTION [GPU_IDS] [--fold-profile pilot|standard|confirmatory] [extra args]" >&2
  exit 2
fi
model=$1
subquestion=$2
gpus=${3:-0}
shift $(( $# >= 3 ? 3 : 2 ))
python -m src.rq2.runner --model "$model" --subquestion "$subquestion" --gpus "$gpus" "$@"
