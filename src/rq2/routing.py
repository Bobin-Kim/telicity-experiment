"""Fail-closed resolution of the six RQ2 subquestions."""
from __future__ import annotations

from ..common.contracts import RQ2_SUBQUESTIONS


def resolve_subquestion(config: dict, selector: str) -> dict:
    if not selector:
        raise ValueError("An explicit RQ2 subquestion is required")
    if selector not in RQ2_SUBQUESTIONS:
        raise ValueError(f"Unknown RQ2 subquestion {selector!r}")
    spec = config.get("subquestions", {}).get(selector)
    if not spec or spec.get("id") != RQ2_SUBQUESTIONS[selector]:
        raise ValueError(f"Incomplete route for {selector}")
    implementation = spec.get("implementation")
    stages = list(spec.get("generic_stages", []))
    if implementation == "generic_pipeline" and not stages:
        raise ValueError(f"Generic route for {selector} has no stages")
    if implementation != "generic_pipeline" and stages:
        raise ValueError(f"Artifact/backend route for {selector} must not expose generic stages")
    forbidden = {"reemergence_cache", "reemergence_sweep", "source_pathway",
                 "source_pathway_extended", "reemergence_aggregate", "upstream_source"}
    overlap = forbidden.intersection(stages)
    if overlap:
        raise ValueError(f"RQ2 route includes RQ1 or unsupported legacy stages: {sorted(overlap)}")
    return {**spec, "selector": selector}
