#!/usr/bin/env python3
"""Re-audit completed DeepSeek RQ2.5 and resume the missing RQ2.6 GPU work."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = "src.rq2.deepseek_rq26"
OUT = ROOT / "results/deepseek-7b-rq2/rq2_6"


def call(phase: str, extra: list[str] | None = None) -> None:
    subprocess.run([sys.executable, "-m", MODULE, "--phase", phase, *(extra or [])], cwd=ROOT, check=True)


def parallel(phase: str, gpus: list[int], batch_size: int) -> None:
    assignments = [[] for _ in gpus]
    for fold in range(5): assignments[fold % len(gpus)].append(fold)
    processes = []
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for gpu, folds in zip(gpus, assignments):
        if not folds: continue
        log_path = OUT / "logs" / f"gpu{gpu}_{phase}.log"
        handle = log_path.open("a", encoding="utf-8")
        command = [sys.executable, "-m", MODULE, "--phase", phase, "--gpu", str(gpu),
                   "--batch-size", str(batch_size), "--folds", *map(str, folds)]
        processes.append((command, subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT), handle))
    failures = []
    for command, process, handle in processes:
        status = process.wait(); handle.close()
        if status: failures.append((command, status))
    if failures: raise RuntimeError(f"Worker failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args(); gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus: parser.error("at least one GPU is required")
    subprocess.run([sys.executable, "validation/run_deepseek_telicity_decision_control.py", "validate"], cwd=ROOT, check=True)
    call("plan"); call("prepare"); call("smoke", ["--gpu", str(gpus[0])])
    parallel("scan-worker", gpus, 8); call("scan-aggregate"); call("shortlist-heads")
    parallel("head-worker", gpus, 8); call("token-cache", ["--gpu", str(gpus[0]), "--batch-size", "8"])
    parallel("token-worker", gpus, 8); parallel("mediation-worker", gpus, 4); call("aggregate")


if __name__ == "__main__": main()
