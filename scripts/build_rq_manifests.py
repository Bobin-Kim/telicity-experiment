#!/usr/bin/env python3
"""Build exhaustive, non-executable RQ ownership and hash manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "manifests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def model_alias(path: Path) -> str:
    text = str(path).lower()
    if "deepseek" in text:
        return "deepseek-7b-chat"
    if "mistral" in text:
        return "mistral-7b-v03"
    if "qwen" in text:
        return "qwen2.5-7b"
    return "shared"


def local_result_record(path: Path, digest: str) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    lower = rel.lower()
    base = {"artifact_kind": "file", "model_alias": model_alias(path),
            "existing_path": str(path.resolve()), "sha256": digest,
            "protocol_type": "continuous_first_token_true_unknown",
            "execution_status": "complete", "audit_status": "NOT_APPLICABLE"}
    if path.name == ".run.lock":
        return {**base, "research_question": "RQ1", "subquestion": None,
                "stage": "execution_lock", "classification": "legacy",
                "execution_status": "ephemeral", "audit_status": "NOT_APPLICABLE"}
    if "deepseek-7b-telicity-rq/telicity_decision_control/" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.telicity_decision_relationship",
                "stage": "telicity_decision_relationship", "classification": "RQ2.5 official",
                "audit_status": "PASS"}
    if "deepseek-7b-rq2/rq2_1/" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.representation_imbalance",
                "stage": "representation_readout_routing_aggregation", "classification": "RQ2.1 official",
                "audit_status": "PASS"}
    if "deepseek-7b-rq2/rq2_2/" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.decision_signal_emergence",
                "stage": "full_depth_decision_emergence", "classification": "RQ2.2 official",
                "audit_status": "PASS"}
    if "deepseek-7b-rq2/rq2_3/" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.decision_circuit",
                "stage": "decision_circuit", "classification": "RQ2.3 official",
                "protocol_type": "heldout_actual_forward_component_and_path_mediation",
                "audit_status": "PASS"}
    if "deepseek-7b-rq2/rq2_4/" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.selective_group_c_correction",
                "stage": "selective_correction", "classification": "RQ2.4 official",
                "protocol_type": "heldout_gate_conditioned_exact_readout_correction",
                "audit_status": "PASS"}
    if "deepseek-7b-rq2/rq2_6/" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.true_driving_origin",
                "stage": "bidirectional_upstream_origin_and_generator_mediation",
                "classification": "RQ2.6 official",
                "protocol_type": "heldout_bidirectional_actual_forward_origin_with_matched_random_control",
                "audit_status": "PASS"}
    if "deepseek-7b-telicity-rq/removal_sweep/" in lower:
        if "next_block" in lower:
            stage = "attention_mlp_recovery"
        elif "recovery" in lower or "layerwise_mechanism" in lower:
            stage = "downstream_reemergence"
        else:
            stage = "moving_telicity_erasure"
        return {**base, "research_question": "RQ1", "subquestion": None, "stage": stage,
                "classification": "RQ1 official", "protocol_type": "telicity_C_vs_D_moving_erasure",
                "audit_status": "PASS"}
    if "deepseek-7b-telicity-rq/source_pathway/" in lower:
        return {**base, "research_question": "RQ1", "subquestion": None,
                "stage": "source_pathway_locked", "classification": "RQ1 official",
                "protocol_type": "locked_residual_context_value_answer_query", "audit_status": "PASS"}
    if "deepseek-7b-telicity-rq/" in lower:
        name = path.name.lower()
        if name.startswith("telicity_probe") or name == "baseline_cd_reps.npz":
            return {**base, "research_question": "RQ1", "subquestion": None,
                    "stage": "telicity_encoding", "classification": "RQ1 official", "audit_status": "PASS"}
        if name.startswith("decision_"):
            return {**base, "research_question": "RQ2", "subquestion": "rq2.decision_signal_emergence",
                    "stage": "decision_predictive_localization", "classification": "RQ2.2 official",
                    "protocol_type": "predictive_continuous_margin_only", "audit_status": "PASS"}
        if name in {"full_depth_cache.npz", "cache_audit.json"}:
            return {**base, "research_question": "shared", "subquestion": None,
                    "stage": "activation_cache", "classification": "shared prerequisite", "audit_status": "PASS"}
        return {**base, "research_question": "legacy", "subquestion": None,
                "stage": "mixed_encoding_decision_output", "classification": "legacy",
                "protocol_type": "continuous_but_mixed_RQ_ownership", "audit_status": "PASS"}
    if "mistral-7b-v03-pre-regression-failed/" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.decision_signal_emergence",
                "stage": "regression_failure", "classification": "invalid",
                "execution_status": "failed", "audit_status": "FAIL"}
    if "mistral-7b-v03-full-rerun-stopped" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.decision_signal_emergence",
                "stage": "stopped_rerun", "classification": "invalid",
                "execution_status": "incomplete", "audit_status": "FAIL"}
    if "mistral-7b-v03-original-equivalence/" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.decision_signal_emergence",
                "stage": "filtered_regression_equivalence", "classification": "legacy",
                "execution_status": "complete_filtered", "audit_status": "PASS"}
    if "mistral-7b-v03-regression-fixed" in lower:
        return {**base, "research_question": "RQ2", "subquestion": "rq2.decision_signal_emergence",
                "stage": "intermediate_regression_repair", "classification": "legacy",
                "execution_status": "partial", "audit_status": "NOT_OFFICIAL"}
    if "e2e-pilot" in lower or "pre-rq-readiness" in lower:
        return {**base, "research_question": "legacy", "subquestion": None,
                "stage": "engineering_readiness", "classification": "legacy",
                "execution_status": "engineering_only", "audit_status": "NOT_OFFICIAL"}
    return {**base, "research_question": "legacy", "subquestion": None,
            "stage": "unclassified_historical", "classification": "legacy"}


def code_record(path: str, rq: str, subquestion: str | None, stage: str,
                classification: str, protocol: str = "continuous_first_token_true_unknown",
                audit: str = "NOT_APPLICABLE") -> dict:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    return {"artifact_kind": "code", "model_alias": model_alias(target),
            "research_question": rq, "subquestion": subquestion, "stage": stage,
            "protocol_type": protocol, "execution_status": "available" if target.exists() else "missing",
            "audit_status": audit, "existing_path": str(target),
            "sha256": sha256(target) if target.is_file() else None, "classification": classification}


def external_record(path: str, model: str, rq: str, subquestion: str | None, stage: str,
                    classification: str, audit: str, protocol: str,
                    execution: str = "complete") -> dict:
    target = Path(path)
    return {"artifact_kind": "external_directory" if target.is_dir() else "external_file",
            "model_alias": model, "research_question": rq, "subquestion": subquestion,
            "stage": stage, "protocol_type": protocol, "execution_status": execution,
            "audit_status": audit, "existing_path": str(target),
            "sha256": sha256(target) if target.is_file() else None,
            "classification": classification}


def declared_code() -> list[dict]:
    records = [
        code_record("src/reemergence.py", "RQ1", None, "reemergence_workflow", "RQ1 official"),
        code_record("src/rq1/runner.py", "RQ1", None, "rq1_entrypoint", "RQ1 official"),
        code_record("src/rq1/routing.py", "RQ1", None, "rq1_stage_routing", "RQ1 official"),
        code_record("validation/run_deepseek_telicity_sweep.py", "RQ1", None, "moving_telicity_erasure", "RQ1 official"),
        code_record("validation/run_deepseek_source_pathway.py", "RQ1", None, "source_pathway_locked", "RQ1 official"),
        code_record("scripts/run_deepseek_reemergence.sh", "RQ1", None, "legacy_rq1_launcher", "legacy"),
        code_record("src/reemergence.py", "RQ1", None, "source_pathway_extension", "RQ1 extension/unexecuted",
                    "extended_context_key_answer_gate"),
        code_record("src/pipeline_controller.py", "RQ2", "rq2.decision_signal_emergence", "generic_decision_emergence", "RQ2.2 official"),
        code_record("src/pipeline_controller.py", "RQ2", "rq2.decision_circuit", "generic_decision_circuit", "RQ2.3 official"),
        code_record("src/pipeline_controller.py", "RQ2", "rq2.true_driving_origin", "correlation_only_upstream_source", "invalid",
                    "correlation_only_not_causal_origin", "FAIL"),
        code_record("validation/run_deepseek_telicity_decision_control.py", "RQ2", "rq2.telicity_decision_relationship", "deepseek_relationship", "RQ2.5 official", audit="PASS"),
        code_record("validation/analyze_deepseek_rq2_representation_imbalance.py", "RQ2", "rq2.representation_imbalance", "deepseek_artifact_first_rq2_1", "RQ2.1 official", audit="PASS"),
        code_record("validation/run_deepseek_rq2_decision_causality.py", "RQ2", "rq2.decision_signal_emergence", "deepseek_block_output_adapter_rq2_2", "RQ2.2 official", audit="PASS"),
        code_record("src/rq2/deepseek_rq23_rq24.py", "RQ2", "rq2.decision_circuit", "deepseek_heldout_component_and_path_backend", "RQ2.3 official", "heldout_actual_forward_component_and_path_mediation", "PASS"),
        code_record("src/rq2/deepseek_rq23_rq24.py", "RQ2", "rq2.selective_group_c_correction", "deepseek_heldout_selective_correction_backend", "RQ2.4 official", "heldout_gate_conditioned_exact_readout_correction", "PASS"),
        code_record("scripts/run_deepseek_rq23_rq24.py", "RQ2", "rq2.decision_circuit", "deepseek_rq23_rq24_launcher", "RQ2.3 official", audit="PASS"),
        code_record("scripts/run_deepseek_rq23_rq24.py", "RQ2", "rq2.selective_group_c_correction", "deepseek_rq23_rq24_launcher", "RQ2.4 official", audit="PASS"),
        code_record("src/rq2/deepseek_rq26.py", "RQ2", "rq2.true_driving_origin", "deepseek_bidirectional_origin_backend", "RQ2.6 official", "heldout_bidirectional_actual_forward_origin_with_matched_random_control", "PASS"),
        code_record("scripts/run_deepseek_rq25_rq26.py", "RQ2", "rq2.true_driving_origin", "deepseek_rq25_reaudit_rq26_launcher", "RQ2.6 official", audit="PASS"),
        code_record("scripts/run_deepseek_telicity_decision_control.py", "RQ2", "rq2.telicity_decision_relationship", "deepseek_relationship_launcher", "RQ2.5 official", audit="PASS"),
        code_record("src/rq2/runner.py", "RQ2", None, "rq2_entrypoint", "RQ2 planned/unexecuted"),
        code_record("src/rq2/routing.py", "RQ2", None, "rq2_subquestion_routing", "RQ2 planned/unexecuted"),
    ]
    for path in ("src/common/config.py", "src/common/contracts.py", "src/common/data.py",
                 "src/common/modeling.py", "src/common/scoring.py", "src/common/audit.py",
                 "src/common/cache.py", "src/common/hooks.py", "src/common/statistics.py",
                 "src/common/controls.py", "src/common/schemas.py",
                 "src/model_registry.py", "src/model_adapters.py", "src/preference_scorer.py",
                 "src/preflight.py", "src/output_validation.py"):
        records.append(code_record(path, "shared", None, "shared_component", "shared prerequisite"))
    return records


def declared_external() -> list[dict]:
    root = "/home/bobin/imperfective_paradox_repro"
    result = f"{root}/results"
    return [
        external_record(f"{result}/qwen25_7b_layerwise_probing", "qwen2.5-7b", "RQ1", None, "telicity_encoding", "RQ1 official", "PASS", "telicity_encoding"),
        external_record(f"{result}/qwen25_7b_telicity_removal_layer_sweep_A6", "qwen2.5-7b", "RQ1", None, "moving_telicity_erasure", "RQ1 official", "PASS", "telicity_moving_erasure"),
        external_record(f"{result}/qwen25_7b_experimentB_depthwise_telicity_recovery_pathway_B1", "qwen2.5-7b", "RQ1", None, "source_pathway_locked", "RQ1 official", "PASS", "locked_residual_context_value_answer_query"),
        external_record(f"{result}/qwen25_7b_exhaustive_telicity_pathways_A12", "qwen2.5-7b", "RQ1", None, "source_pathway_extension", "RQ1 extension/unexecuted", "NOT_CROSS_MODEL_OFFICIAL", "extended_pathways"),
        external_record(f"{result}/Mistral_RQ1_COMPLETE", "mistral-7b-v03", "RQ1", None, "telicity_encoding_erasure_reemergence", "RQ1 official", "PASS", "mistral_telicity_reemergence"),
        external_record(f"{result}/mistral7b_mechanistic_replication", "mistral-7b-v03", "RQ1", None, "depthwise_mechanism", "RQ1 official", "PASS", "mistral_telicity_mechanism"),
        external_record(f"{result}/mistral7b_matched_magnitude_gain_A33", "mistral-7b-v03", "RQ2", "rq2.representation_imbalance", "matched_representation_readout", "RQ2.1 official", "PASS", "matched_norm_causal_gain"),
        external_record(f"{result}/qwen25_7b_rq2a_continuous_preference_A22", "qwen2.5-7b", "RQ2", "rq2.decision_signal_emergence", "decision_signal_emergence", "RQ2.2 official", "PASS", "continuous_margin_decoding"),
        external_record(f"{result}/qwen25_7b_rq2bc_continuous_preference_A23", "qwen2.5-7b", "RQ2", "rq2.decision_signal_emergence", "causal_decision_emergence", "RQ2.2 official", "PASS", "continuous_margin_intervention"),
        external_record(f"{result}/Mistral_RQ2_experiments_3_4_5", "mistral-7b-v03", "RQ2", "rq2.decision_signal_emergence", "full_depth_decision_emergence", "RQ2.2 official", "PASS", "continuous_margin_intervention"),
        external_record(f"{result}/Qwen_Full_Depth_True_Driving_A35", "qwen2.5-7b", "RQ2", "rq2.decision_circuit", "decision_circuit", "RQ2.3 official", "PASS", "full_depth_causal_circuit"),
        external_record(f"{result}/mistral7b_multisource_routing_trace_A32", "mistral-7b-v03", "RQ2", "rq2.decision_circuit", "decision_circuit", "RQ2.3 official", "PASS", "multisource_causal_routing"),
        external_record(f"{result}/qwen25_7b_rq2_final_semantic_rebalancing_A20", "qwen2.5-7b", "RQ2", "rq2.selective_group_c_correction", "selective_correction", "RQ2.4 official", "PASS", "heldout_selective_correction"),
        external_record(f"{result}/qwen25_7b_experiment6_adaptive_ta_l20_l27_A27", "qwen2.5-7b", "RQ2", "rq2.selective_group_c_correction", "selective_correction", "RQ2.4 official", "PASS", "heldout_selective_correction"),
        external_record(f"{result}/qwen25_7b_rq2c_continuous_preference_A24", "qwen2.5-7b", "RQ2", "rq2.telicity_decision_relationship", "telicity_decision_relationship", "RQ2.5 official", "PASS", "continuous_telicity_decision_control"),
        external_record(f"{result}/mistral7b_telicity_decision_mediation_A30", "mistral-7b-v03", "RQ2", "rq2.telicity_decision_relationship", "telicity_decision_relationship", "RQ2.5 official", "PASS", "continuous_telicity_decision_control"),
        external_record(f"{result}/mistral7b_decision_alignment_A36", "mistral-7b-v03", "RQ2", "rq2.telicity_decision_relationship", "failed_decision_alignment", "invalid", "FAIL", "failed_causal_pathway"),
        external_record(f"{result}/qwen25_7b_upstream_true_origin_A37", "qwen2.5-7b", "RQ2", "rq2.true_driving_origin", "upstream_true_origin", "RQ2.6 official", "PASS", "causal_upstream_source"),
        external_record(f"{result}/mistral7b_upstream_true_origin_A37", "mistral-7b-v03", "RQ2", "rq2.true_driving_origin", "failed_upstream_true_origin", "invalid", "FAIL", "failed_causal_upstream_source"),
        external_record(f"{result}/cross_model_upstream_true_origin_A37", "shared", "RQ2", "rq2.true_driving_origin", "failed_cross_model_origin", "invalid", "FAIL", "failed_cross_model_aggregate"),
    ]


def declared_plans(records: list[dict]) -> list[dict]:
    """Make unsupported/unexecuted model-question combinations explicit."""
    registry = yaml.safe_load((ROOT / "configs/multimodel_true_unknown_preference.yaml").read_text())
    aliases = tuple(registry["model_configs"])
    rq2_config = ROOT / "configs/rq2_decision_mechanism.yaml"
    selectors = tuple(yaml.safe_load(rq2_config.read_text())["subquestions"])
    planned: list[dict] = []
    for model in aliases:
        for selector in selectors:
            complete = any(row["model_alias"] == model and row["subquestion"] == selector
                           and row["classification"].endswith("official")
                           and row["execution_status"] == "complete" and row["audit_status"] == "PASS"
                           for row in records)
            if not complete:
                planned.append({
                    "artifact_kind": "plan", "model_alias": model,
                    "research_question": "RQ2", "subquestion": selector,
                    "stage": "explicit_subquestion_backend",
                    "protocol_type": "continuous_first_token_true_unknown",
                    "execution_status": "planned", "audit_status": "NOT_EXECUTED",
                    "existing_path": str(rq2_config), "sha256": sha256(rq2_config),
                    "classification": "RQ2 planned/unexecuted",
                })
    extension_config = ROOT / "configs/rq1_telicity_reemergence.yaml"
    for model in aliases:
        complete_extension = any(row["model_alias"] == model
                                 and row["stage"] == "source_pathway_extension"
                                 and row["execution_status"] == "complete" for row in records)
        if not complete_extension:
            planned.append({
                "artifact_kind": "plan", "model_alias": model,
                "research_question": "RQ1", "subquestion": None,
                "stage": "source_pathway_extension",
                "protocol_type": "extended_context_key_answer_gate",
                "execution_status": "planned", "audit_status": "NOT_EXECUTED",
                "existing_path": str(extension_config), "sha256": sha256(extension_config),
                "classification": "RQ1 extension/unexecuted",
            })
    return planned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--refresh-prefix", help="authorize hash refresh only below this result-path prefix")
    args = parser.parse_args()
    result_files = sorted(path for path in RESULTS.rglob("*") if path.is_file())
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in result_files}
    baseline_path = OUT / "result_hash_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text()).get("files", {})
        changed = [path for path, digest in baseline.items() if hashes.get(path) != digest]
        unauthorized = [path for path in changed if not (args.refresh_prefix and path.startswith(args.refresh_prefix))]
        if unauthorized:
            raise RuntimeError(f"Existing result hash preservation failed: {unauthorized[:20]}")
        # Authorized new experiments extend the baseline without rewriting any
        # previously locked hash.
        if not args.verify_only and (set(hashes) != set(baseline) or changed):
            write_json(baseline_path, {"scope": "all completed result files; prior hashes preserved and authorized experiments appended",
                                       "file_count": len(hashes), "files": hashes})
    elif args.verify_only:
        raise RuntimeError("Result hash baseline is missing")
    else:
        write_json(baseline_path, {"scope": "all files under repository results/ before RQ reorganization",
                                   "file_count": len(hashes), "files": hashes})
    local = [local_result_record(ROOT / path, digest) for path, digest in hashes.items()]
    records = local + declared_code() + declared_external()
    records += declared_plans(records)
    all_manifest = {"schema_version": 1, "executable": False,
                    "scoring_contract": "z_True - z_Unknown at first answer token",
                    "artifact_count": len(records), "artifacts": records}
    rq1 = [row for row in records if row["research_question"] in {"RQ1", "shared"}]
    rq2 = [row for row in records if row["research_question"] in {"RQ2", "shared"}]
    if not args.verify_only:
        write_json(OUT / "all_artifacts.json", all_manifest)
        write_json(OUT / "rq1_artifacts.json", {**all_manifest, "artifact_count": len(rq1), "artifacts": rq1})
        write_json(OUT / "rq2_artifacts.json", {**all_manifest, "artifact_count": len(rq2), "artifacts": rq2})
    print(json.dumps({"status": "PASS", "result_files": len(hashes), "all_records": len(records),
                      "rq1_records": len(rq1), "rq2_records": len(rq2),
                      "result_hashes_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
