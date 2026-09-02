"""Subprocess routing shared by the separate RQ entry points."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def preference_runner_command(base_config: Path, model: str, gpus: str, stages: list[str],
                              output_dir: Path | None = None, dry_run: bool = False,
                              batch_size: int | None = None) -> list[str]:
    if not stages:
        raise ValueError("Fail-closed: refusing an empty stage route")
    command = [sys.executable, str(ROOT / "run_multimodel_preference.py"),
               "--config", str(base_config), "--model", model, "--gpus", gpus,
               "--stages", ",".join(stages), "--resume"]
    if output_dir is not None:
        command += ["--output-dir", str(output_dir.resolve())]
    if dry_run:
        command.append("--dry-run")
    if batch_size is not None:
        command += ["--batch-size", str(batch_size)]
    return command


def run_command(command: list[str], plan_only: bool) -> int:
    print(" ".join(command))
    if plan_only:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode
