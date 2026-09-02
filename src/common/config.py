"""Load small RQ routing configs while retaining the single model registry."""
from __future__ import annotations

from pathlib import Path
import yaml

from .contracts import assert_continuous_contract

ROOT = Path(__file__).resolve().parents[2]


def load_rq_config(path: Path, expected_rq: str) -> tuple[dict, Path]:
    path = path.resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("research_question") != expected_rq:
        raise ValueError(f"Expected {expected_rq} routing config: {path}")
    assert_continuous_contract(value)
    base = Path(value["base_model_registry"])
    if not base.is_absolute():
        base = (path.parent / base).resolve()
    if not base.is_file():
        raise FileNotFoundError(f"Base model registry does not exist: {base}")
    return value, base


def registered_aliases(base_registry: Path) -> tuple[str, ...]:
    raw = yaml.safe_load(base_registry.read_text())
    aliases = raw.get("model_configs", {})
    if not aliases:
        raise ValueError("Base registry has no model aliases")
    return tuple(aliases)
