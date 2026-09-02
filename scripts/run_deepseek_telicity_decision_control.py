#!/usr/bin/env python3
"""Resumable launcher for the DeepSeek Telicity/decision-control experiment.

Examples
--------
Four GPUs (fold 0 and 4 share GPU 0):
    python scripts/run_deepseek_telicity_decision_control.py --gpus 0,1,2,3

One GPU:
    python scripts/run_deepseek_telicity_decision_control.py --gpus 0

Completed preparation, folds, and the cache-reproduction audit are reused.
Every invocation re-aggregates and validates the saved scientific outputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "validation/run_deepseek_telicity_decision_control.py"
OUT = ROOT / "results/deepseek-7b-telicity-rq/telicity_decision_control"


def command(*args: object) -> list[str]:
    return [sys.executable, str(EXPERIMENT), *(str(value) for value in args)]


def checked(*args: object) -> None:
    subprocess.run(command(*args), cwd=ROOT, check=True)


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must be comma-separated integer IDs") from exc
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("--gpus must contain one or more unique nonnegative IDs")
    return gpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--condition-chunk", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.condition_chunk < 1:
        parser.error("batch and condition chunk sizes must be positive")

    # This is deliberately printed before any model load or CUDA work.
    checked("plan")
    checked("prepare", "--gpu", args.gpus[0])

    assignments = {gpu: [] for gpu in args.gpus}
    for fold in range(5):
        assignments[args.gpus[fold % len(args.gpus)]].append(fold)
    OUT.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    try:
        for gpu, fold_ids in assignments.items():
            if not fold_ids:
                continue
            log_path = OUT / f"gpu{gpu}.log"
            handle = log_path.open("ab")
            invocation = command("run", "--gpu", gpu, "--folds", *fold_ids,
                                 "--batch-size", args.batch_size,
                                 "--condition-chunk", args.condition_chunk)
            processes.append((gpu, subprocess.Popen(invocation, cwd=ROOT, stdout=handle,
                                                     stderr=subprocess.STDOUT), handle))
        failures = []
        for gpu, process, handle in processes:
            code = process.wait()
            handle.close()
            if code:
                failures.append({"gpu": gpu, "exit_code": code, "log": str(OUT / f"gpu{gpu}.log")})
        if failures:
            raise RuntimeError(f"fold worker failure: {json.dumps(failures)}")
    except BaseException:
        for _, process, handle in processes:
            if process.poll() is None:
                process.terminate()
            if not handle.closed:
                handle.close()
        raise

    checked("audit-baseline", "--gpu", args.gpus[0], "--batch-size", args.batch_size)
    checked("aggregate")
    checked("validate")
    print(json.dumps({"status": "complete", "audit": "PASS",
                      "report": str(OUT / "REPORT.md"),
                      "summary": str(OUT / "summary.json"),
                      "handoff_validation": str(OUT / "handoff_validation.json")}, indent=2))


if __name__ == "__main__":
    main()
