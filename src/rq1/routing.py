"""RQ1-only physical stage routing."""
from __future__ import annotations

from ..common.contracts import RQ1_EXTENSION_STAGE, RQ1_LOCKED_PIPELINE_STAGES


def rq1_stages(include_extension: bool) -> list[str]:
    stages = list(RQ1_LOCKED_PIPELINE_STAGES)
    if include_extension:
        stages.insert(stages.index("reemergence_aggregate"), RQ1_EXTENSION_STAGE)
    forbidden = {"behavior", "predictive_localization", "causal_intervention",
                 "component_analysis", "generator_localization", "aggregate", "upstream_source"}
    if forbidden.intersection(stages):
        raise RuntimeError("RQ1 route crossed into an RQ2 stage")
    return stages
