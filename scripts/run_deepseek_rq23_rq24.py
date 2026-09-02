#!/usr/bin/env python3
"""Resume the packaged DeepSeek RQ2.3/RQ2.4 experiment on available GPUs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = "src.rq2.deepseek_rq23_rq24"


def call(phase: str, extra: list[str] | None = None) -> None:
    command = [sys.executable, "-m", MODULE, "--phase", phase, *(extra or [])]
    subprocess.run(command, cwd=ROOT, check=True)


def parallel(phase: str, gpus: list[int], batch_size: int) -> None:
    assignments = [[] for _ in gpus]
    for fold in range(5): assignments[fold % len(gpus)].append(fold)
    processes = []
    for gpu, folds in zip(gpus, assignments):
        if not folds: continue
        command = [sys.executable, "-m", MODULE, "--phase", phase, "--gpu", str(gpu),
                   "--batch-size", str(batch_size), "--folds", *map(str, folds)]
        processes.append((command, subprocess.Popen(command, cwd=ROOT)))
    failures = []
    for command, process in processes:
        status = process.wait()
        if status: failures.append((command, status))
    if failures: raise RuntimeError(f"Worker failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--only", choices=("both", "rq23", "rq24"), default="both")
    args = parser.parse_args(); gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus: parser.error("at least one GPU is required")
    call("plan"); call("cache-head", ["--gpu", str(gpus[0]), "--batch-size", "8"]); call("prepare")
    if args.only in ("both", "rq23"):
        parallel("rq23-worker", gpus, 4); call("rq23-aggregate")
        parallel("rq23-path-worker", gpus, 2); call("rq23-path-aggregate")
    if args.only in ("both", "rq24"):
        call("rq24-select"); parallel("rq24-worker", gpus, 8); call("rq24-aggregate")


if __name__ == "__main__": main()
