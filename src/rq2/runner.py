#!/usr/bin/env python3
"""Explicit-subquestion entry point for RQ2; no implicit RQ1 or RQ2 chain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.config import load_rq_config, registered_aliases
from ..common.contracts import RQ2_SUBQUESTIONS
from ..common.routing import preference_runner_command, run_command
from .artifacts import reusable_official, select_artifacts
from .routing import resolve_subquestion

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/rq2_decision_mechanism.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subquestion", required=True, choices=tuple(RQ2_SUBQUESTIONS))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="execute the underlying non-weight tokenizer/config dry run")
    args = parser.parse_args()
    config, base = load_rq_config(args.config, "rq2")
    aliases = registered_aliases(base)
    if args.model not in aliases:
        parser.error(f"unknown model alias {args.model!r}; choose from {', '.join(aliases)}")
    route = resolve_subquestion(config, args.subquestion)
    manifest = Path(config["artifact_manifest"])
    if not manifest.is_absolute():
        manifest = (args.config.resolve().parent / manifest).resolve()
    records = select_artifacts(manifest, args.model, args.subquestion)
    official = reusable_official(records)
    if official:
        print(json.dumps({"status": "reuse", "research_question": route["id"],
                          "selector": args.subquestion, "model": args.model,
                          "implementation": route["implementation"],
                          "official_artifacts": official, "manifest": str(manifest),
                          "gpu_work_started": False}, indent=2))
        return
    if route["implementation"] == "generic_pipeline":
        command = preference_runner_command(base, args.model, args.gpus, route["generic_stages"],
                                            args.output_dir, args.dry_run, args.batch_size)
        raise SystemExit(run_command(command, args.plan_only))

    result = {"status": "reuse" if official else "planned_or_unavailable",
              "research_question": route["id"], "selector": args.subquestion,
              "model": args.model, "implementation": route["implementation"],
              "official_artifacts": official, "manifest": str(manifest),
              "gpu_work_started": False}
    print(json.dumps(result, indent=2))
    if not official and not args.plan_only:
        raise SystemExit(f"No audited completed implementation is registered for {args.model} / {args.subquestion}")


if __name__ == "__main__":
    main()
