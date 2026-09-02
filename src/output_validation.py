from __future__ import annotations
import csv, json, math, os
from pathlib import Path

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows: raise ValueError(f"Refusing empty official artifact: {path}")
    fields = []
    for row in rows:
        fields.extend(x for x in row if x not in fields)
        for v in row.values():
            if isinstance(v, float) and not math.isfinite(v): raise ValueError(f"NaN/Inf in {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def validate_stage(path: Path, stage: str, fingerprint: str):
    marker = path / f"{stage}.complete.json"
    if not marker.exists(): return False
    try: value = json.loads(marker.read_text())
    except Exception: return False
    return value.get("status") == "complete" and value.get("run_fingerprint") == fingerprint

def ensure_clean_or_resume(out: Path, fingerprint: str, resume: bool):
    state = out / "run_state.json"
    if not state.exists():
        if out.exists() and any(out.iterdir()): raise RuntimeError(f"Output-directory collision: {out} is nonempty and has no compatible run state")
        out.mkdir(parents=True, exist_ok=True); atomic_json(state, {"run_fingerprint": fingerprint})
    else:
        old = json.loads(state.read_text()).get("run_fingerprint")
        if old != fingerprint: raise RuntimeError("Incompatible resume state (model/config/code fingerprint differs)")
        if not resume: raise RuntimeError("Output already exists; use --resume or select a new --output-dir")
