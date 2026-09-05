#!/usr/bin/env python3
"""RQ1-only entry point for Telicity representation and re-emergence."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..common.config import load_rq_config, registered_aliases
from ..common.fold_profiles import FOLD_PROFILES
from ..common.routing import preference_runner_command, run_command
from .routing import rq1_stages

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/rq1_telicity_reemergence.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--include-extension", action="store_true",
                        help="also run Context-K/Answer-Gate without changing the locked three-path result")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--fold-profile", choices=tuple(FOLD_PROFILES), default="confirmatory",
                        help="pilot=1 fold, standard=3 folds, confirmatory=5 folds (official)")
    parser.add_argument("--plan-only", action="store_true",
                        help="print the resolved command without tokenizer/model access")
    parser.add_argument("--dry-run", action="store_true",
                        help="execute the underlying non-weight tokenizer/config dry run")
    args = parser.parse_args()
    _, base = load_rq_config(args.config, "rq1")
    aliases = registered_aliases(base)
    if args.model not in aliases:
        parser.error(f"unknown model alias {args.model!r}; choose from {', '.join(aliases)}")
    stages = rq1_stages(args.include_extension)
    command = preference_runner_command(base, args.model, args.gpus, stages, args.output_dir,
                                        args.dry_run, args.batch_size, args.fold_profile)
    raise SystemExit(run_command(command, args.plan_only))


if __name__ == "__main__":
    main()
