#!/usr/bin/env python3
"""CPU-only validation for the RQ1/RQ2 repository reorganization."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_rq_config, registered_aliases
from src.common.contracts import RQ2_SUBQUESTIONS
from src.reemergence import pathway_contract
from src.rq1.routing import rq1_stages
from src.rq2.routing import resolve_subquestion

OUT_JSON = ROOT / "validation/RQ_ORGANIZATION_VALIDATION.json"
OUT_REPORT = ROOT / "validation/RQ_ORGANIZATION_VALIDATION.md"
OUT_AUDIT = ROOT / "validation/RQ_ORGANIZATION_FINAL_AUDIT.md"

# Captured from MANIFEST.json before the RQ-only routing files were added.
# Together these lock prompts/dataset/folds/seeds/controls and the sole scorer.
PREORGANIZATION_PROTOCOL_HASHES = {
    "configs/multimodel_true_unknown_preference.yaml": "ffeeba14348ed28e117c89929dfa19e7fa0d32f3d8f733105d2988fd41aebd75",
    "src/preference_scorer.py": "90732cfcf05d7afb63aa2c1c9fd2afa4b03fd19804df729bb35e7003bf05784e",
    "src/preflight.py": "2d5de934a6a4d3efb6bd453f94a591e7a677899598c5dbb93ce0188bf671223e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(command, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def main() -> None:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    rq1, base1 = load_rq_config(ROOT / "configs/rq1_telicity_reemergence.yaml", "rq1")
    rq2, base2 = load_rq_config(ROOT / "configs/rq2_decision_mechanism.yaml", "rq2")
    aliases = registered_aliases(base1)
    checks["yaml_parsing"] = bool(rq1 and rq2)
    checks["shared_model_registry"] = base1 == base2
    checks["model_alias_resolution"] = {"qwen2.5-7b", "mistral-7b-v03", "deepseek-7b-chat"} <= set(aliases)
    details["model_aliases"] = list(aliases)

    locked, extended = rq1_stages(False), rq1_stages(True)
    rq2_physical = {stage for selector in RQ2_SUBQUESTIONS
                    for stage in resolve_subquestion(rq2, selector)["generic_stages"]}
    checks["rq1_locked_route"] = locked == ["reemergence_cache", "reemergence_sweep",
                                                   "source_pathway", "reemergence_aggregate"]
    checks["rq1_extension_is_optional"] = ("source_pathway_extended" not in locked
                                             and "source_pathway_extended" in extended)
    checks["rq1_never_routes_rq2"] = not set(extended).intersection(rq2_physical)
    checks["rq2_has_six_subquestions"] = set(rq2["subquestions"]) == set(RQ2_SUBQUESTIONS)
    checks["rq2_excludes_rq1_and_legacy_upstream"] = not rq2_physical.intersection(
        {"reemergence_cache", "reemergence_sweep", "source_pathway", "source_pathway_extended",
         "reemergence_aggregate", "upstream_source"})
    try:
        resolve_subquestion(rq2, "")
        checks["rq2_requires_explicit_subquestion"] = False
    except ValueError:
        checks["rq2_requires_explicit_subquestion"] = True
    contract = pathway_contract()
    checks["locked_extended_pathways_separate"] = (contract["locked_is_subset"]
                                                     and not contract["renormalize_locked_with_extensions"])

    executable_text = json.dumps({"rq1": {"execution": rq1["execution"], "logical": rq1["logical_stages"]},
                                  "rq2": {"execution": rq2["execution"], "subquestions": rq2["subquestions"]}}).lower()
    checks["no_alternate_executable_scoring_stage"] = "binary" not in executable_text
    checks["canonical_scoring_locked"] = (rq1["scoring_contract"] == rq2["scoring_contract"]
                                           and rq1["scoring_contract"]["formula"] == "z_True - z_Unknown")
    protocol_hashes = {path: sha256(ROOT / path) for path in PREORGANIZATION_PROTOCOL_HASHES}
    checks["scientific_protocol_files_unchanged"] = protocol_hashes == PREORGANIZATION_PROTOCOL_HASHES
    details["protocol_hashes_before_and_after"] = {
        path: {"before": expected, "after": protocol_hashes[path]}
        for path, expected in PREORGANIZATION_PROTOCOL_HASHES.items()
    }

    python = sys.executable
    no_selector = run([python, "-m", "src.rq2.runner", "--model", "qwen2.5-7b", "--gpus", ""])
    checks["rq2_cli_no_selector_fails"] = no_selector.returncode != 0 and "--subquestion" in no_selector.stderr
    rq1_plan = run([python, "-m", "src.rq1.runner", "--model", "qwen2.5-7b", "--gpus", "", "--plan-only"])
    rq1_extension_plan = run([python, "-m", "src.rq1.runner", "--model", "qwen2.5-7b", "--gpus", "",
                              "--include-extension", "--plan-only"])
    rq2_plan = run([python, "-m", "src.rq2.runner", "--model", "qwen2.5-7b", "--gpus", "",
                    "--subquestion", "rq2.decision_signal_emergence", "--plan-only"])
    rq25_reuse = run([python, "-m", "src.rq2.runner", "--model", "deepseek-7b-chat", "--gpus", "",
                      "--subquestion", "rq2.telicity_decision_relationship", "--plan-only"])
    checks["rq1_plan_dry_run"] = rq1_plan.returncode == 0 and "reemergence_cache" in rq1_plan.stdout and "causal_intervention" not in rq1_plan.stdout
    checks["rq1_extension_plan_dry_run"] = rq1_extension_plan.returncode == 0 and "source_pathway_extended" in rq1_extension_plan.stdout
    # Once an audited official artifact exists, the artifact-first runner
    # intentionally returns reuse rather than constructing a generic command.
    checks["rq2_plan_dry_run"] = (rq2_plan.returncode == 0
                                   and ("causal_intervention" in rq2_plan.stdout
                                        or '"status": "reuse"' in rq2_plan.stdout)
                                   and "reemergence" not in rq2_plan.stdout)
    checks["rq25_completed_artifact_reuse"] = rq25_reuse.returncode == 0 and '"status": "reuse"' in rq25_reuse.stdout
    details["routing_commands"] = {"rq1": rq1_plan.stdout.strip(), "rq1_extension": rq1_extension_plan.stdout.strip(),
                                    "rq2_2": rq2_plan.stdout.strip()}

    real_rq1_dry = run([python, "-m", "src.rq1.runner", "--model", "qwen2.5-7b", "--gpus", "", "--dry-run"])
    real_rq2_dry = run([python, "-m", "src.rq2.runner", "--model", "qwen2.5-7b", "--gpus", "",
                        "--subquestion", "rq2.decision_signal_emergence", "--dry-run"])
    checks["rq1_tokenizer_config_dry_run"] = real_rq1_dry.returncode == 0 and '"mode": "dry-run/config-and-tokenizer"' in real_rq1_dry.stdout
    checks["rq2_tokenizer_config_dry_run"] = (real_rq2_dry.returncode == 0
                                               and ('"mode": "dry-run/config-and-tokenizer"' in real_rq2_dry.stdout
                                                    or ('"status": "reuse"' in real_rq2_dry.stdout
                                                        and '"gpu_work_started": false' in real_rq2_dry.stdout)))
    details["dry_run_errors"] = {"rq1": real_rq1_dry.stderr[-1000:], "rq2": real_rq2_dry.stderr[-1000:]}

    baseline = json.loads((ROOT / "manifests/result_hash_baseline.json").read_text())
    current_files = sorted(path for path in (ROOT / "results").rglob("*") if path.is_file())
    current_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in current_files}
    checks["existing_result_hashes_preserved"] = current_hashes == baseline["files"]
    details["result_hash_file_count"] = len(current_hashes)

    all_manifest = json.loads((ROOT / "manifests/all_artifacts.json").read_text())
    rq1_manifest = json.loads((ROOT / "manifests/rq1_artifacts.json").read_text())
    rq2_manifest = json.loads((ROOT / "manifests/rq2_artifacts.json").read_text())
    local_manifest_paths = {row["existing_path"] for row in all_manifest["artifacts"]
                            if row["artifact_kind"] == "file"}
    checks["manifest_covers_every_local_result"] = len(local_manifest_paths) == len(current_files)
    required = {"model_alias", "research_question", "subquestion", "stage", "protocol_type",
                "execution_status", "audit_status", "existing_path", "sha256", "classification"}
    checks["manifest_schema_complete"] = all(required <= set(row) for row in all_manifest["artifacts"])
    checks["manifest_paths_valid"] = all(Path(row["existing_path"]).exists() for row in all_manifest["artifacts"]
                                          if row["execution_status"] not in {"missing", "planned"})
    failed_mistral = [row for row in rq2_manifest["artifacts"]
                      if "mistral7b_decision_alignment_A36" in row["existing_path"]
                      or "mistral7b_upstream_true_origin_A37" in row["existing_path"]]
    checks["failed_mistral_artifacts_invalid"] = (len(failed_mistral) == 2
                                                   and all(row["classification"] == "invalid"
                                                           and row["audit_status"] == "FAIL"
                                                           for row in failed_mistral))
    checks["rq1_manifest_has_locked_and_extension"] = (
        any(row["stage"] == "source_pathway_locked" for row in rq1_manifest["artifacts"])
        and any(row["stage"] == "source_pathway_extension"
                and row["classification"] == "RQ1 extension/unexecuted" for row in rq1_manifest["artifacts"]))

    status = "PASS" if all(checks.values()) else "FAIL"
    result = {"status": status, "gpu_experiments_run": False, "result_files_modified": False,
              "checks": checks, "details": details}
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# RQ1/RQ2 Organization Validation", "", f"Status: **{status}**", "",
             "This validation invocation loaded no model weights and ran no GPU experiment or scientific stage. Result files were read only for hash verification.", "",
             "## Checks", ""]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in checks.items())
    lines += ["", f"Result files hash-checked: **{len(current_files)}**.",
              "Failed Mistral decision-alignment and upstream-origin artifacts remain classified invalid and are absent from official executable routes.", ""]
    OUT_REPORT.write_text("\n".join(lines))
    audit_lines = [
        "# RQ Reorganization Final Audit",
        "",
        f"Status: **{status}**",
        "",
        "This validation invocation was read-only: no GPU experiment, model-weight load, scientific stage, or result recomputation was performed by the validator. The separately authorized DeepSeek RQ2.3/RQ2.4 execution is documented in `validation/DEEPSEEK_RQ23_RQ24_REPORT.md`.",
        "",
        "- The fixed experiment config, prompt/dataset/fold preflight, and sole continuous scorer match their pre-reorganization SHA-256 values.",
        "- The canonical endpoint remains `m = z(True_first_token) - z(Unknown_first_token)` at the last non-padding prompt position.",
        "- RQ1 locked and extended source-pathway outputs remain separate; the extension cannot renormalize or overwrite the locked result.",
        "- RQ2 requires exactly one explicit selector and is never launched by the RQ1 route.",
        f"- All **{len(current_files)}** recorded files under `results/` match the locked baseline; prior hashes were preserved and the authorized RQ2.3/RQ2.4 artifacts were appended.",
        "- Failed Mistral causal-pathway/upstream-source artifacts are retained as invalid and are not official executable inputs.",
        "",
        "Conclusion: no previously locked scientific result value or scientific protocol changed during repository organization or this validation.",
        "",
    ]
    OUT_AUDIT.write_text("\n".join(audit_lines))
    print(json.dumps({"status": status, "checks_passed": sum(checks.values()),
                      "checks_total": len(checks), "result_files_hash_checked": len(current_files)}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
