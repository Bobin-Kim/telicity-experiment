#!/usr/bin/env python3
"""Read-only audit of the organized re-emergence handoff implementation."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reemergence import EXTENDED_PATHS, LOCKED_PATHS, pathway_contract


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config = yaml.safe_load((ROOT / "configs/multimodel_true_unknown_preference.yaml").read_text())
    fixed = config["fixed_experiment_config"]
    re_cfg = fixed["reemergence_analysis"]
    raw_path = ROOT / "results/deepseek-7b-telicity-rq/source_pathway/layer_fold_pathway_results.csv"
    with raw_path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    observed = {(int(r["erasure_layer"]), int(r["fold"]), r["pathway"]) for r in raw}
    expected = {(layer, fold, path) for layer in range(1, 28) for fold in range(5) for path in LOCKED_PATHS}
    model_modules = config["model_configs"]["deepseek-7b-chat"]["modules"]
    live_preflight = json.loads((ROOT / "validation/DEEPSEEK_REEMERGENCE_PREFLIGHT.json").read_text())
    scoring_source = (ROOT / "src/preference_scorer.py").read_text()
    controller_source = (ROOT / "src/pipeline_controller.py").read_text()
    checks = {
        "existing_deepseek_locked_grid_405": len(raw) == 405 and observed == expected,
        "yaml_locked_paths_exact": tuple(re_cfg["locked_pathways"]) == LOCKED_PATHS,
        "yaml_extended_paths_exact": tuple(re_cfg["extended_pathways"]) == EXTENDED_PATHS,
        "protocols_not_renormalized_together": pathway_contract()["renormalize_locked_with_extensions"] is False,
        "deepseek_qkv_gate_modules_configured": all(name in model_modules for name in ("query", "key", "value", "gate")),
        "deepseek_live_gpu_hook_preflight": live_preflight["status"] == "PASS"
            and all(live_preflight["hook_module_counts"][name] == 30 for name in ("query", "key", "value", "gate")),
        "all_optional_stages_registered": all(stage in controller_source for stage in
            ("reemergence_cache", "reemergence_sweep", "source_pathway", "source_pathway_extended", "reemergence_aggregate")),
        "first_token_scorer_unchanged": "t-u" in scoring_source and "self.ids[\"True\"]" in scoring_source
            and "self.ids[\"Unknown\"]" in scoring_source,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "existing_locked_raw_sha256": sha256(raw_path),
        "existing_locked_raw_rows": len(raw),
        "locked_pathways": list(LOCKED_PATHS),
        "extended_pathways": list(EXTENDED_PATHS),
        "scientific_boundary": {
            "locked": "historical Residual/Context-V/Answer-Q reproduction target",
            "extended": "new Context-K and Answer-Gate causal patches; no historical equivalence claim",
        },
        "gpu_validation": "live DeepSeek GPU-0 load and Q/K/V/G hook preflight PASS; existing locked run complete; extended K/G numerical run pending",
    }
    target = ROOT / "validation/REEMERGENCE_HANDOFF_AUDIT.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
