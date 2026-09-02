"""Logical RQ2.5 ownership for model-specific completed implementations.

The DeepSeek implementation remains at its audited historical path
`validation/run_deepseek_telicity_decision_control.py`; this module is the
stable logical owner and deliberately contains no selected layer or result.
Qwen and Mistral canonical implementations are referenced by the RQ2
artifact manifest rather than copied.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK_IMPLEMENTATION = ROOT / "validation/run_deepseek_telicity_decision_control.py"


def implementation_path() -> Path:
    return DEEPSEEK_IMPLEMENTATION
