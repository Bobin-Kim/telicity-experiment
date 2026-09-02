from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import yaml

def load_registry(path: Path, key: str):
    path = path.resolve()
    raw = yaml.safe_load(path.read_text())
    if set(raw) != {"package_version", "fixed_experiment_config", "model_configs"}:
        raise ValueError("Configuration must contain only package_version, fixed_experiment_config, model_configs")
    if key not in raw["model_configs"]:
        raise KeyError(f"Unknown model {key!r}; choose from {', '.join(raw['model_configs'])}")
    fixed, model = copy.deepcopy(raw["fixed_experiment_config"]), copy.deepcopy(raw["model_configs"][key])
    dataset = Path(fixed["dataset_path"])
    if not dataset.is_absolute():
        fixed["dataset_path"] = str((path.parent / dataset).resolve())
    if "binary" in json.dumps({"fixed": fixed, "model": model}).lower():
        raise ValueError("Fail-closed: binary-only artifact/configuration detected")
    return raw, fixed, model

def resolved_source(model: dict) -> str:
    local = model.get("local_path")
    return local if local and Path(local).is_dir() else model["model_id"]

def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
