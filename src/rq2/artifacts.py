"""Read-only reuse of completed or planned RQ2 artifact records."""
from __future__ import annotations

import json
from pathlib import Path


def select_artifacts(manifest: Path, model: str, selector: str) -> list[dict]:
    value = json.loads(manifest.read_text())
    return [row for row in value.get("artifacts", [])
            if row.get("model_alias") == model and row.get("subquestion") == selector]


def reusable_official(records: list[dict]) -> list[dict]:
    return [row for row in records
            if row.get("classification", "").endswith("official")
            and row.get("execution_status") == "complete"
            and row.get("audit_status") == "PASS"
            and row.get("existing_path")]
