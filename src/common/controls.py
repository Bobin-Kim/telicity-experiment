"""Read-only access to the single shared control specification."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/multimodel_true_unknown_preference.yaml"


def control_contract(registry: Path = DEFAULT_REGISTRY) -> dict:
    """Return the fixed controls without maintaining an RQ-specific copy."""
    fixed = yaml.safe_load(registry.read_text())["fixed_experiment_config"]
    return deepcopy({
        "random_seed": fixed["random_seed"],
        "folds": fixed["folds"],
        "preference_basis": fixed["preference_basis"],
        "intervention": fixed["intervention"],
        "reemergence_analysis": fixed["reemergence_analysis"],
    })


__all__ = ["control_contract"]
