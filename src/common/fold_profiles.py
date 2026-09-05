"""Named held-out-fold budgets for reproducible staged experimentation."""
from __future__ import annotations

FOLD_PROFILES = {
    "pilot": (0,),
    "standard": (0, 1, 2),
    "confirmatory": (0, 1, 2, 3, 4),
}


def folds_for_profile(profile: str) -> list[int]:
    try:
        return list(FOLD_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"Unknown fold profile {profile!r}; choose from {tuple(FOLD_PROFILES)}") from exc


def is_confirmatory(folds: list[int]) -> bool:
    return folds == list(FOLD_PROFILES["confirmatory"])
