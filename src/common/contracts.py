"""Immutable contracts shared by RQ1 and RQ2 entry points."""
from __future__ import annotations

CANONICAL_SCORING_MODE = "first_answer_token_logit_margin"
CANONICAL_FORMULA = "z_True - z_Unknown"
CANONICAL_POSITION = "last_non_padding_prompt_token"

RQ1_LOCKED_PIPELINE_STAGES = (
    "reemergence_cache",
    "reemergence_sweep",
    "source_pathway",
    "reemergence_aggregate",
)
RQ1_EXTENSION_STAGE = "source_pathway_extended"

RQ2_SUBQUESTIONS = {
    "rq2.representation_imbalance": "RQ2.1",
    "rq2.decision_signal_emergence": "RQ2.2",
    "rq2.decision_circuit": "RQ2.3",
    "rq2.selective_group_c_correction": "RQ2.4",
    "rq2.telicity_decision_relationship": "RQ2.5",
    "rq2.true_driving_origin": "RQ2.6",
}


def assert_continuous_contract(value: dict) -> None:
    """Reject any orchestration config that changes the canonical endpoint."""
    scoring = value.get("scoring_contract", {})
    expected = {"mode": CANONICAL_SCORING_MODE, "formula": CANONICAL_FORMULA,
                "position": CANONICAL_POSITION}
    if scoring != expected:
        raise ValueError(f"Scoring contract mismatch: expected {expected}, found {scoring}")
    executable = {"execution": value.get("execution"), "subquestions": value.get("subquestions")}
    if "binary" in str(executable).lower():
        raise ValueError("Fail-closed: disallowed alternate scoring protocol in executable routing")
