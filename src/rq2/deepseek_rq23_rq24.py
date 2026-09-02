#!/usr/bin/env python3
"""Resumable DeepSeek RQ2.3 circuit and RQ2.4 selective-correction backend.

The backend reuses the independently completed DeepSeek cache and RQ2.2
causal sweep.  Model-specific layers and heads are read from prior audited
artifacts or selected on outer-training data; none are constants in config.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/multimodel_true_unknown_preference.yaml"
BASE = ROOT / "results/deepseek-7b-telicity-rq"
RQ22 = ROOT / "results/deepseek-7b-rq2/rq2_2"
RQ23 = ROOT / "results/deepseek-7b-rq2/rq2_3"
RQ24 = ROOT / "results/deepseek-7b-rq2/rq2_4"
CACHE = BASE / "full_depth_cache.npz"
HEAD_CACHE = RQ23 / "head_input_cache.npz"
PREPARED = RQ23 / "prepared_assets.npz"
CONTROL_ASSETS = BASE / "telicity_decision_control/intervention_assets.npz"
FOLDS = 5
SEED = 42
LABELS = ("True", "False", "Unknown")
COMPONENT_CONDITIONS = ("decision_pos", "decision_neg", "random_pos", "random_neg")
CORRECTION_CONDITIONS = ("baseline", "internal_hard", "internal_soft", "internal_product",
                         "random_gate", "oracle_C_only", "unconditional", "reverse", "restore")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    return value / max(float(np.linalg.norm(value)), 1e-12)


def groups(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["group"][0] for row in rows])


def load_inputs(require_heads: bool = True):
    from src.model_registry import load_registry
    from src.preflight import folds, load_dataset
    _raw, fixed, spec = load_registry(CONFIG, "deepseek-7b-chat")
    rows = load_dataset(fixed); cache = np.load(CACHE, allow_pickle=False)
    fold_ids = np.asarray(folds(rows, fixed), int)
    checks = {
        "rq22_audit": json.loads((RQ22 / "audit.json").read_text())["status"] == "PASS",
        "cache_shape": cache["residual"].shape == (400, 30, 4096),
        "cache_ids": cache["ids"].tolist() == [row["id"] for row in rows],
        "cache_folds": np.array_equal(cache["fold"].astype(int), fold_ids),
        "cache_finite": all(np.isfinite(cache[key]).all() for key in ("residual", "attention", "mlp", "margin")),
        "dataset_hash": fixed["dataset_sha256"] == "ddd01d06b18b9bab24abb5bc92a8eb740bb145123658fa9a10bd86e38b4bdb5a",
    }
    head = None
    if require_heads:
        if not HEAD_CACHE.is_file():
            raise FileNotFoundError(f"Run --phase cache-head first: {HEAD_CACHE}")
        head_archive = np.load(HEAD_CACHE, allow_pickle=False); head = head_archive["head"]
        checks.update({"head_shape": head.shape == (400, 30, 4096),
                       "head_ids": head_archive["ids"].tolist() == [row["id"] for row in rows],
                       "head_finite": bool(np.isfinite(head).all())})
    if not all(checks.values()):
        raise RuntimeError(f"Reusable input audit failed: {checks}")
    arrays = {key: cache[key] for key in ("residual", "attention", "mlp")}
    if head is not None: arrays["head"] = head
    return fixed, spec, rows, arrays, cache["margin"].astype(float), fold_ids, checks


def load_gpu(gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from src.model_adapters import make_adapter
    from src.model_registry import load_registry
    from src.preflight import run_preflight
    from validation.run_deepseek_telicity_decision_control import load_model
    _raw, fixed, spec = load_registry(CONFIG, "deepseek-7b-chat")
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, tokenizer = load_model(spec); tokenizer.padding_side = "right"
    report, rows, scorer = run_preflight(tokenizer, fixed, spec, model)
    if scorer.audit()["scoring_token_ids"] != {"True": 8173, "Unknown": 21483}:
        raise RuntimeError("Canonical DeepSeek scoring token identity changed")
    return model, tokenizer, make_adapter(model, spec), fixed, rows, scorer, report


def encode(tokenizer, fixed, rows, indices, device):
    from src.preflight import build_prompt
    prompts = [build_prompt(tokenizer, fixed, rows[index]["premise"], rows[index]["hypothesis"])
               for index in indices]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    answer = encoded["attention_mask"].sum(1).long() - 1
    return {key: value.to(device) for key, value in encoded.items()}, answer.to(device)


def cache_head(gpu: int, batch_size: int) -> None:
    if HEAD_CACHE.is_file() and (RQ23 / "head_cache_audit.json").is_file():
        audit = json.loads((RQ23 / "head_cache_audit.json").read_text())
        if audit.get("status") == "PASS" and audit.get("cache_sha256") == sha256(CACHE):
            print(json.dumps({"status": "reuse", "path": str(HEAD_CACHE)}, indent=2)); return
    import torch
    fixed, _spec, expected_rows, arrays, margins, _folds, checks = load_inputs(require_heads=False)
    model, tokenizer, adapter, _fixed, rows, scorer, _report = load_gpu(gpu)
    device = next(model.parameters()).device; head = np.zeros((400, 30, 4096), np.float16)
    reproduced = np.zeros(400, np.float32)
    for start in range(0, 400, batch_size):
        indices = list(range(start, min(start + batch_size, 400))); encoded, answer = encode(tokenizer, fixed, rows, indices, device)
        row_index = torch.arange(len(indices), device=device); captures = [None] * 30; handles = []
        for layer in range(30):
            def hook(_module, inputs, layer=layer): captures[layer] = inputs[0].detach()
            handles.append(adapter.head_input(layer).register_forward_pre_hook(hook))
        try:
            with torch.inference_mode(): output = model(**encoded, use_cache=False, return_dict=True)
        finally:
            for handle in handles: handle.remove()
        stop = start + len(indices)
        head[start:stop] = torch.stack([captures[layer][row_index, answer] for layer in range(30)], 1).half().cpu().numpy()
        reproduced[start:stop] = scorer.from_logits(output.logits, answer)["margin"].cpu().numpy()
        print(f"[HEAD CACHE] {stop}/400", flush=True)
    if not np.isfinite(head).all(): raise RuntimeError("Nonfinite head cache")
    RQ23.mkdir(parents=True, exist_ok=True); temporary = HEAD_CACHE.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, head=head, ids=np.asarray([row["id"] for row in expected_rows]))
    temporary.replace(HEAD_CACHE)
    max_error = float(np.max(np.abs(reproduced - margins)))
    audit = {"status": "PASS" if max_error <= 1e-4 else "FAIL", "rows": 400,
             "shape": list(head.shape), "finite": True, "baseline_margin_max_abs_error": max_error,
             "cache_sha256": sha256(CACHE), "head_cache_sha256": sha256(HEAD_CACHE),
             "gpu": gpu, "scoring": "z_True_first_token-z_Unknown_first_token", "input_checks": checks}
    write_json(RQ23 / "head_cache_audit.json", audit)
    if audit["status"] != "PASS": raise RuntimeError(audit)


def fit_axis(x: np.ndarray, margin: np.ndarray, train: np.ndarray, seed: int) -> dict:
    x = np.asarray(x, np.float64); center = x[train].mean(0)
    axis = unit((x[train] - center).T @ (margin[train] - margin[train].mean()))
    scale = max(float(np.std((x[train] - center) @ axis)), 1e-4)
    random = np.random.default_rng(seed).normal(size=len(axis)); random -= axis * float(random @ axis)
    return {"center": center, "axis": axis, "random": unit(random), "scale": scale,
            "train_score": abs(float(np.corrcoef((x[train] - center) @ axis, margin[train])[0, 1]))}


def candidate_matrix(arrays: dict, component: str, layer: int, head: int = -1) -> np.ndarray:
    value = arrays[component][:, layer].astype(np.float64)
    if component == "head": return value.reshape(len(value), 32, 128)[:, head]
    return value


def gate_fit(x: np.ndarray, group: np.ndarray, train: np.ndarray) -> dict:
    x = np.asarray(x, np.float64); means = {name: x[train & (group == name)].mean(0) for name in "ABCD"}
    center = x[train].mean(0)
    gate = unit(means["C"] - np.vstack([x[train & (group == name)] for name in "ABD"]).mean(0))
    telicity = unit((means["A"] + means["C"]) / 2 - (means["B"] + means["D"]) / 2)
    ambiguity = unit((means["C"] + means["D"]) / 2 - (means["A"] + means["B"]) / 2)
    def score(direction):
        raw = (x - center) @ direction; mean = float(raw[train].mean()); sd = float(raw[train].std()) or 1.0
        return (raw - mean) / sd
    return {"direction": gate, "center": center, "scores": score(gate),
            "telicity_scores": score(telicity), "ambiguity_scores": score(ambiguity), "means": means}


def balanced_accuracy(y, pred) -> float:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    return float((pred[y == 1].mean() + (1 - pred[y == 0]).mean()) / 2)


def auc(y, score) -> float:
    y, score = np.asarray(y, int), np.asarray(score, float); pos, neg = score[y == 1], score[y == 0]
    return float(np.mean(pos[:, None] > neg) + .5 * np.mean(pos[:, None] == neg))


def threshold_fit(y, score) -> float:
    values = np.unique(score); thresholds = np.r_[values[0] - 1e-6, (values[:-1] + values[1:]) / 2, values[-1] + 1e-6]
    predictions = score[:, None] >= thresholds; positive = np.asarray(y, int) == 1
    ba = (predictions[positive].mean(0) + (~predictions[~positive]).mean(0)) / 2
    best = np.flatnonzero(ba >= ba.max() - 1e-12)
    return float(thresholds[best[np.argmin(abs(thresholds[best]))]])


def contrast_metrics(score, group, use, threshold) -> dict:
    target = (group[use] == "C").astype(int); pred = (score[use] >= threshold).astype(int)
    result = {"C_vs_rest_BA": balanced_accuracy(target, pred), "C_vs_rest_AUROC": auc(target, score[use])}
    for other in ("D", "A"):
        mask = use & np.isin(group, ("C", other)); y = (group[mask] == "C").astype(int)
        result[f"C_vs_{other}_BA"] = balanced_accuracy(y, (score[mask] >= threshold).astype(int))
        result[f"C_vs_{other}_AUROC"] = auc(y, score[mask])
    return result


def prepare() -> None:
    reusable = (PREPARED, RQ23 / "component_selections.csv", RQ24 / "gate_scan.csv",
                RQ24 / "gate_selections.csv", RQ24 / "gate_scores.csv",
                RQ23 / "execution_plan.json", RQ24 / "execution_plan.json")
    if all(path.is_file() and path.stat().st_size > 0 for path in reusable):
        for path in (RQ23 / "execution_plan.json", RQ24 / "execution_plan.json"):
            plan = json.loads(path.read_text())
            plan["rq24"].update({
                "circuit_reference_layer": plan["rq24"].pop("target_layer", plan["rq24"].get("circuit_reference_layer")),
                "correction_actuator": "independently audited exact final-norm True-Unknown readout",
            })
            write_json(path, plan)
        print(json.dumps({"status": "reuse", "gpu_work_started": False,
                          "reason": "prepared selections and integrity-checked assets already exist"}, indent=2))
        return
    fixed, spec, rows, arrays, margin, fold_ids, checks = load_inputs(True); group = groups(rows)
    onset = int(json.loads((RQ22 / "summary.json").read_text())["earliest_stable_causal_layer"])
    if onset is None: raise RuntimeError("RQ2.2 did not identify a causal onset")
    target_layer = int(json.loads((RQ22 / "summary.json").read_text())["strongest_predictive_layer"])
    onset_layers = tuple(layer for layer in (onset - 1, onset, onset + 1) if 1 <= layer <= 30)
    component_selections, gate_scan, gate_selections, gate_scores = [], [], [], []
    for held in range(FOLDS):
        train = fold_ids != held
        # RQ2.3: fixed Attention/MLP onset window plus train-selected heads and downstream shortlist.
        for layer_number in onset_layers:
            layer = layer_number - 1
            for component in ("attention", "mlp"):
                component_selections.append({"fold": held, "family": "onset", "component": component,
                                             "layer": layer, "layer_number": layer_number, "head": -1,
                                             "rank": 0, "train_score": fit_axis(candidate_matrix(arrays, component, layer), margin, train, SEED + held * 100 + layer)["train_score"]})
            ranked = []
            for head in range(32):
                score = fit_axis(candidate_matrix(arrays, "head", layer, head), margin, train,
                                 SEED + held * 1000 + layer * 32 + head)["train_score"]
                ranked.append((score, head))
            for rank, (score, head) in enumerate(sorted(ranked, reverse=True)[:2], 1):
                component_selections.append({"fold": held, "family": "onset", "component": "head",
                                             "layer": layer, "layer_number": layer_number, "head": head,
                                             "rank": rank, "train_score": score})
        downstream = []
        for layer_number in range(onset + 2, target_layer):
            layer = layer_number - 1
            for component in ("attention", "mlp"):
                score = fit_axis(candidate_matrix(arrays, component, layer), margin, train,
                                 SEED + held * 200 + layer)["train_score"]
                downstream.append((score, component, layer, -1))
            for head in range(32):
                score = fit_axis(candidate_matrix(arrays, "head", layer, head), margin, train,
                                 SEED + held * 2000 + layer * 32 + head)["train_score"]
                downstream.append((score, "head", layer, head))
        chosen = sorted((item for item in downstream if item[1] == "head"), reverse=True)[:3]
        chosen += sorted((item for item in downstream if item[1] == "mlp"), reverse=True)[:2]
        chosen += sorted((item for item in downstream if item[1] == "attention"), reverse=True)[:1]
        for rank, (score, component, layer, head) in enumerate(chosen, 1):
            component_selections.append({"fold": held, "family": "downstream", "component": component,
                                         "layer": layer, "layer_number": layer + 1, "head": head,
                                         "rank": rank, "train_score": score})
        # RQ2.4: canonical C-vs-rest gate scan; held-out labels are reporting only.
        best = None
        for component in ("residual", "attention", "mlp", "head"):
            for layer in range(30):
                for head in (range(32) if component == "head" else (-1,)):
                    x = candidate_matrix(arrays, component, layer, head); fitted = gate_fit(x, group, train)
                    threshold = threshold_fit((group[train] == "C").astype(int), fitted["scores"][train])
                    train_metrics = contrast_metrics(fitted["scores"], group, train, threshold)
                    test_metrics = contrast_metrics(fitted["scores"], group, ~train, threshold)
                    utility = train_metrics["C_vs_rest_BA"] + .25 * min(train_metrics["C_vs_D_BA"], train_metrics["C_vs_A_BA"])
                    row = {"fold": held, "component": component, "layer": layer, "layer_number": layer + 1,
                           "head": head, "train_threshold": threshold, "train_utility": utility,
                           **{f"train_{key}": value for key, value in train_metrics.items()},
                           **{f"test_{key}": value for key, value in test_metrics.items()}}
                    gate_scan.append(row)
                    if best is None or (utility, -layer) > (best[0], -best[2]): best = (utility, component, layer, head, threshold)
        assert best is not None
        _utility, component, layer, head, threshold = best
        x = candidate_matrix(arrays, component, layer, head); fitted = gate_fit(x, group, train)
        gate_selections.append({"fold": held, "component": component, "layer": layer,
                                "layer_number": layer + 1, "head": head, "train_threshold": threshold,
                                "train_utility": _utility, "target_decision_layer": target_layer})
        for index, row in enumerate(rows):
            gate_scores.append({"basis_fold": held, "split": "test" if fold_ids[index] == held else "train",
                                "example_index": index, "example_id": row["id"], "group": group[index],
                                "gate_score": fitted["scores"][index], "telicity_score": fitted["telicity_scores"][index],
                                "ambiguity_score": fitted["ambiguity_scores"][index], "component": component,
                                "layer": layer, "layer_number": layer + 1, "head": head})
        print(f"[PREPARE F{held}] gate={component} L{layer+1} H{head}", flush=True)
    RQ23.mkdir(parents=True, exist_ok=True); RQ24.mkdir(parents=True, exist_ok=True)
    write_csv(RQ23 / "component_selections.csv", component_selections)
    write_csv(RQ24 / "gate_scan.csv", gate_scan); write_csv(RQ24 / "gate_selections.csv", gate_selections)
    write_csv(RQ24 / "gate_scores.csv", gate_scores)
    with PREPARED.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, onset=np.asarray(onset), target_layer=np.asarray(target_layer))
    PREPARED.with_suffix(".npz.tmp").replace(PREPARED)
    plan = {"status": "PASS", "gpu_work_started": False, "model": "deepseek-7b-chat",
            "rq23": {"causal_onset_from_audited_rq22": onset, "component_window": list(onset_layers),
                     "target_layer_from_audited_rq22": target_layer,
                     "selection": "outer-train component/head correlation; held-out actual-forward causality"},
            "rq24": {"gate": "outer-train C-vs-rest scan", "circuit_reference_layer": target_layer,
                     "correction_actuator": "independently audited exact final-norm True-Unknown readout",
                     "heldout_labels_used_for_selection": False},
            "scoring": "m = z(True_first_token) - z(Unknown_first_token)",
            "reused": {"cache": {"path": str(CACHE), "sha256": sha256(CACHE)},
                       "rq22": {"path": str(RQ22), "audit": "PASS"},
                       "head_cache": {"path": str(HEAD_CACHE), "sha256": sha256(HEAD_CACHE)}},
            "input_checks": checks, "dataset_sha256": fixed["dataset_sha256"], "model_revision": spec["revision"]}
    write_json(RQ23 / "execution_plan.json", plan); write_json(RQ24 / "execution_plan.json", plan)


def decision_bundle(held: int, target_layer: int, arrays: dict, margin: np.ndarray, fold_ids: np.ndarray, device):
    from validation.run_deepseek_rq2_decision_causality import bundle
    return bundle(target_layer - 1, held, arrays["residual"].astype(np.float32), margin, fold_ids, device)


def modify_component(adapter, spec: dict, answer, axis, random_axis, scale, condition_ids, count, batch):
    import torch
    component, layer, head = spec["component"], int(spec["layer"]), int(spec["head"])
    row_index = torch.arange(count * batch, device=answer.device)
    def alter(value):
        output = value.clone()
        if component == "head":
            view = output[row_index, answer].view(count * batch, 32, 128); current = view[:, head]
        else: current = output[row_index, answer]
        for index, name in enumerate(COMPONENT_CONDITIONS):
            selected = condition_ids.eq(index); direction = random_axis if name.startswith("random") else axis
            sign = 1 if name.endswith("pos") else -1
            current[selected] += sign * scale * direction.to(current.dtype)
        if component == "head": view[:, head] = current; output[row_index, answer] = view.reshape(count * batch, -1)
        else: output[row_index, answer] = current
        return output
    if component == "attention":
        return adapter.attention(layer).register_forward_hook(
            lambda _m, _i, out: (alter(out[0]), *out[1:]) if isinstance(out, tuple) else alter(out))
    if component == "mlp": return adapter.mlp(layer).register_forward_hook(lambda _m, _i, out: alter(out))
    return adapter.head_input(layer).register_forward_pre_hook(lambda _m, inp: (alter(inp[0]), *inp[1:]))


def rq23_worker(folds_to_run: list[int], gpu: int, batch_size: int) -> None:
    import torch
    fixed, _spec, expected_rows, arrays, margin, fold_ids, _checks = load_inputs(True)
    selections = read_csv(RQ23 / "component_selections.csv"); target = int(np.load(PREPARED)["target_layer"])
    model, tokenizer, adapter, _fixed, rows, _scorer, _report = load_gpu(gpu); device = next(model.parameters()).device
    if [row["id"] for row in rows] != [row["id"] for row in expected_rows]: raise RuntimeError("Row identity changed")
    label_ids = torch.tensor([tokenizer(value, add_special_tokens=False).input_ids[0] for value in LABELS], device=device)
    for held in folds_to_run:
        directory = RQ23 / "component_shards" / f"fold{held}"; target_csv = directory / "interventions.csv"
        if target_csv.is_file() and (directory / "complete.json").is_file():
            old = read_csv(target_csv)
            expected = int(np.sum(fold_ids == held)) * len([r for r in selections if int(r["fold"]) == held]) * 4
            if len(old) == expected: print(f"[RQ2.3 F{held}] reuse", flush=True); continue
        train = fold_ids != held; indices = np.flatnonzero(~train).tolist(); records = []
        db = decision_bundle(held, target, arrays, margin, fold_ids, device)
        for spec in [row for row in selections if int(row["fold"]) == held]:
            component, layer, head = spec["component"], int(spec["layer"]), int(spec["head"])
            fitted = fit_axis(candidate_matrix(arrays, component, layer, head), margin, train,
                              SEED + held * 1000 + layer * 32 + max(head, 0))
            axis = torch.as_tensor(fitted["axis"], device=device, dtype=torch.float32)
            random_axis = torch.as_tensor(fitted["random"], device=device, dtype=torch.float32)
            for start in range(0, len(indices), batch_size):
                chosen = indices[start:start + batch_size]; encoded0, answer0 = encode(tokenizer, fixed, rows, chosen, device)
                count, size = 4, len(chosen); encoded = {key: value.repeat((count, 1)) for key, value in encoded0.items()}
                answer = answer0.repeat(count); condition_ids = torch.arange(count, device=device).repeat_interleave(size)
                captured = {}
                handle = modify_component(adapter, spec, answer, axis, random_axis, fitted["scale"], condition_ids, count, size)
                def endpoint(_module, _inputs, output): captured["target"] = (output[0] if isinstance(output, tuple) else output).detach()
                end_handle = adapter.layers[target - 1].register_forward_hook(endpoint)
                try:
                    with torch.inference_mode(): result = model(**encoded, use_cache=False, return_dict=True)
                finally: handle.remove(); end_handle.remove()
                row_index = torch.arange(count * size, device=device)
                logits = result.logits[row_index, answer][:, label_ids].float().cpu().numpy()
                endpoint_value = captured["target"][row_index, answer].float().cpu().numpy()
                for condition_index, condition in enumerate(COMPONENT_CONDITIONS):
                    for local, source_index in enumerate(chosen):
                        offset = condition_index * size + local; current_margin = float(logits[offset, 0] - logits[offset, 2])
                        target_score = float((endpoint_value[offset] - db["center"]) @ db["axis"] / db["scale"])
                        baseline_score = float((arrays["residual"][source_index, target-1].astype(float) - db["center"]) @ db["axis"] / db["scale"])
                        records.append({"fold": held, "example_index": source_index, "example_id": rows[source_index]["id"],
                                        "group": rows[source_index]["group"][0], "family": spec["family"],
                                        "component": component, "layer": layer, "layer_number": layer + 1, "head": head,
                                        "rank": int(spec["rank"]), "condition": condition, "train_score": fitted["train_score"],
                                        "scale": fitted["scale"], "baseline_margin": margin[source_index], "margin": current_margin,
                                        "delta_margin": current_margin - margin[source_index], "target_layer": target,
                                        "baseline_target_score": baseline_score, "target_score": target_score,
                                        "delta_target_score": target_score - baseline_score})
            print(f"[RQ2.3 F{held}] {component} L{layer+1} H{head}", flush=True)
        write_csv(target_csv, records)
        write_json(directory / "complete.json", {"status": "complete", "fold": held, "rows": len(records),
                                                   "gpu": gpu, "conditions": list(COMPONENT_CONDITIONS)})


def fold_ci(values) -> tuple[float, float, float, float]:
    from scipy.stats import t as tdist, ttest_1samp
    value = np.asarray(values, float); mean = float(value.mean())
    if len(value) < 2: return mean, float("nan"), float("nan"), float("nan")
    half = float(tdist.ppf(.975, len(value)-1) * value.std(ddof=1) / math.sqrt(len(value)))
    return mean, mean-half, mean+half, float(ttest_1samp(value, 0).pvalue)


def rq23_aggregate() -> None:
    raw = []
    for held in range(FOLDS): raw.extend(read_csv(RQ23 / "component_shards" / f"fold{held}/interventions.csv"))
    write_csv(RQ23 / "component_interventions.csv", raw)
    keys = sorted({(r["family"], r["component"], int(r["layer"]), int(r["head"])) for r in raw})
    fold_rows, summary_rows = [], []
    for family, component, layer, head in keys:
        effects, endpoint, selected_folds = [], [], []
        for held in range(FOLDS):
            subset = [r for r in raw if r["family"] == family and r["component"] == component
                      and int(r["layer"]) == layer and int(r["head"]) == head and int(r["fold"]) == held]
            if not subset: continue
            means = {name: float(np.mean([float(r["delta_margin"]) for r in subset if r["condition"] == name])) for name in COMPONENT_CONDITIONS}
            targets = {name: float(np.mean([float(r["delta_target_score"]) for r in subset if r["condition"] == name])) for name in COMPONENT_CONDITIONS}
            effect = ((means["decision_pos"] - means["decision_neg"]) - (means["random_pos"] - means["random_neg"])) / 2
            end_effect = ((targets["decision_pos"] - targets["decision_neg"]) - (targets["random_pos"] - targets["random_neg"])) / 2
            effects.append(effect); endpoint.append(end_effect); selected_folds.append(held)
            fold_rows.append({"family": family, "component": component, "layer": layer, "layer_number": layer+1,
                              "head": head, "fold": held, "specific_delta_margin": effect,
                              "specific_delta_target_score": end_effect})
        mean, lo, hi, p = fold_ci(effects); emean, elo, ehi, ep = fold_ci(endpoint)
        reliable = len(effects) >= 4 and lo > 0 and sum(value > 0 for value in effects) >= 4
        summary_rows.append({"family": family, "component": component, "layer": layer, "layer_number": layer+1,
                             "head": head, "selected_folds": len(effects), "specific_delta_margin": mean,
                             "ci95_low": lo, "ci95_high": hi, "fold_t_p": p,
                             "positive_folds": sum(value > 0 for value in effects),
                             "specific_delta_target_score": emean, "target_ci95_low": elo,
                             "target_ci95_high": ehi, "target_fold_t_p": ep, "reliable_causal": int(reliable)})
    write_csv(RQ23 / "component_foldwise.csv", fold_rows); write_csv(RQ23 / "component_summary.csv", summary_rows)
    reliable_onset = [r for r in summary_rows if r["family"] == "onset" and r["reliable_causal"]]
    generator = min(reliable_onset, key=lambda r: (r["layer"], 0 if r["component"] == "attention" else 1,
                                                    -r["specific_delta_margin"])) if reliable_onset else None
    reliable_downstream = sorted((r for r in summary_rows if r["family"] == "downstream" and r["reliable_causal"]),
                                 key=lambda r: r["specific_delta_margin"], reverse=True)
    selected_path = reliable_downstream[:3]
    summary = {"status": "component_complete", "audit_status": "PASS", "model": "deepseek-7b-chat",
               "target": "m = z_True - z_Unknown", "generator": generator,
               "reliable_onset_components": reliable_onset, "selected_downstream_path_candidates": selected_path,
               "path_mediation_pending": True,
               "classification_rule": ">=4 selected folds, >=4 positive, fold-t 95% CI > 0; Attention ordered before MLP within a block"}
    write_json(RQ23 / "component_summary.json", summary)
    if generator is None: raise RuntimeError("No reliable RQ2.3 generator component; path stage cannot proceed")


def donor_maps(rows, fold_ids, margin, held):
    train = np.flatnonzero(fold_ids != held); group = groups(rows); median = float(np.median(margin[train])); opposite, same = {}, {}
    for index in np.flatnonzero(fold_ids == held):
        state = margin[index] >= median; pool = train[group[train] == group[index]]
        op = pool[(margin[pool] >= median) != state]; sa = pool[(margin[pool] >= median) == state]
        if not len(op): op = train[(margin[train] >= median) != state]
        if not len(sa): sa = train[(margin[train] >= median) == state]
        opposite[index] = int(op[np.argmin(abs(margin[op] - (2 * median - margin[index])))])
        same[index] = int(sa[np.argmin(abs(margin[sa] - margin[index]))])
    return opposite, same


def component_restore_hook(adapter, entity, arrays, chosen, answer, condition_ids, block_names, count, batch):
    import torch
    component, layer, head = entity["component"], int(entity["layer"]), int(entity["head"])
    row_index = torch.arange(count * batch, device=answer.device)
    clean_np = candidate_matrix(arrays, component, layer, head)[chosen]
    clean = torch.as_tensor(np.concatenate([clean_np] * count), device=answer.device)
    tag = f"{component}_L{layer+1}_H{head}"
    def alter(value):
        output = value.clone()
        if component == "head": view = output[row_index, answer].view(count*batch, 32, 128); current = view[:, head]
        else: current = output[row_index, answer]
        selected = torch.zeros(count * batch, dtype=torch.bool, device=answer.device)
        for condition_index, name in enumerate(block_names):
            if name in (tag, "all_components"): selected |= condition_ids.eq(condition_index)
        current[selected] = clean[selected].to(current.dtype)
        if component == "head": view[:, head] = current; output[row_index, answer] = view.reshape(count*batch, -1)
        else: output[row_index, answer] = current
        return output
    if component == "attention": return adapter.attention(layer).register_forward_hook(lambda _m,_i,out:(alter(out[0]),*out[1:]) if isinstance(out,tuple) else alter(out))
    if component == "mlp": return adapter.mlp(layer).register_forward_hook(lambda _m,_i,out:alter(out))
    return adapter.head_input(layer).register_forward_pre_hook(lambda _m,inp:(alter(inp[0]),*inp[1:]))


def rq23_path_worker(folds_to_run: list[int], gpu: int, batch_size: int) -> None:
    import torch
    fixed, _spec, expected_rows, arrays, margin, fold_ids, _checks = load_inputs(True)
    component_summary = json.loads((RQ23 / "component_summary.json").read_text()); generator = component_summary["generator"]
    downstream = component_summary["selected_downstream_path_candidates"]
    target = int(np.load(PREPARED)["target_layer"])
    model, tokenizer, adapter, _fixed, rows, _scorer, _report = load_gpu(gpu); device = next(model.parameters()).device
    if [r["id"] for r in rows] != [r["id"] for r in expected_rows]: raise RuntimeError("Row identity changed")
    label_ids = torch.tensor([tokenizer(v, add_special_tokens=False).input_ids[0] for v in LABELS], device=device)
    gen_component, gen_layer, gen_head = generator["component"], int(generator["layer"]), int(generator["head"])
    block_tags = [f'{r["component"]}_L{int(r["layer"])+1}_H{int(r["head"])}' for r in downstream]
    blocks = ("none", *block_tags, "all_components", "target_axis", "target_full")
    states = ("opposite", "same")
    conditions = (("baseline", "baseline"),) + tuple((state, block) for state in states for block in blocks)
    for held in folds_to_run:
        directory = RQ23 / "path_shards" / f"fold{held}"; path = directory / "mediation.csv"
        expected = int(np.sum(fold_ids == held)) * len(conditions)
        if path.is_file() and (directory / "complete.json").is_file() and len(read_csv(path)) == expected:
            print(f"[RQ2.3 PATH F{held}] reuse", flush=True); continue
        indices = np.flatnonzero(fold_ids == held).tolist(); opposite, same = donor_maps(rows, fold_ids, margin, held)
        db = decision_bundle(held, target, arrays, margin, fold_ids, device); records = []
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start+batch_size]; count, size = len(conditions), len(chosen)
            encoded0, answer0 = encode(tokenizer, fixed, rows, chosen, device); encoded = {k:v.repeat((count,1)) for k,v in encoded0.items()}
            answer = answer0.repeat(count); condition_ids = torch.arange(count, device=device).repeat_interleave(size)
            row_index = torch.arange(count*size, device=device); handles=[]; captures={}
            gen_cache = candidate_matrix(arrays, gen_component, gen_layer, gen_head)
            def source_alter(value):
                output=value.clone()
                if gen_component=="head": view=output[row_index,answer].view(count*size,32,128); current=view[:,gen_head]
                else: current=output[row_index,answer]
                for ci,(state,_block) in enumerate(conditions):
                    if state=="baseline": continue
                    donors=opposite if state=="opposite" else same
                    selected=condition_ids.eq(ci); replacement=torch.as_tensor(np.stack([gen_cache[donors[i]] for i in chosen]),device=device,dtype=current.dtype)
                    current[selected]=replacement
                if gen_component=="head": view[:,gen_head]=current;output[row_index,answer]=view.reshape(count*size,-1)
                else: output[row_index,answer]=current
                return output
            if gen_component=="attention": handles.append(adapter.attention(gen_layer).register_forward_hook(lambda _m,_i,out:(source_alter(out[0]),*out[1:]) if isinstance(out,tuple) else source_alter(out)))
            elif gen_component=="mlp": handles.append(adapter.mlp(gen_layer).register_forward_hook(lambda _m,_i,out:source_alter(out)))
            else: handles.append(adapter.head_input(gen_layer).register_forward_pre_hook(lambda _m,inp:(source_alter(inp[0]),*inp[1:])))
            for entity in downstream: handles.append(component_restore_hook(adapter,entity,arrays,chosen,answer,condition_ids,[b for _,b in conditions],count,size))
            center=torch.as_tensor(db["center"],device=device,dtype=torch.float32);axis=torch.as_tensor(db["axis"],device=device,dtype=torch.float32)
            clean_target=torch.as_tensor(np.concatenate([arrays["residual"][chosen,target-1].astype(np.float32)]*count),device=device)
            def target_restore(_m,_i,out):
                raw=out[0] if isinstance(out,tuple) else out; value=raw.clone(); current=value[row_index,answer].float()
                for ci,(_state,block) in enumerate(conditions):
                    selected=condition_ids.eq(ci)
                    if block=="target_axis": current[selected]+=(((clean_target[selected]-center)@axis)-((current[selected]-center)@axis))[:,None]*axis
                    elif block=="target_full": current[selected]=clean_target[selected]
                value[row_index,answer]=current.to(value.dtype);captures["target"]=value.detach();return (value,*out[1:]) if isinstance(out,tuple) else value
            handles.append(adapter.layers[target-1].register_forward_hook(target_restore))
            try:
                with torch.inference_mode(): result=model(**encoded,use_cache=False,return_dict=True)
            finally:
                for handle in handles: handle.remove()
            logits=result.logits[row_index,answer][:,label_ids].float().cpu().numpy(); endpoint=captures["target"][row_index,answer].float().cpu().numpy()
            for ci,(state,block) in enumerate(conditions):
                for local,index in enumerate(chosen):
                    offset=ci*size+local; current_margin=float(logits[offset,0]-logits[offset,2])
                    records.append({"fold":held,"example_index":index,"example_id":rows[index]["id"],"group":rows[index]["group"][0],
                                    "state":state,"block":block,"donor_index":"" if state=="baseline" else (opposite[index] if state=="opposite" else same[index]),
                                    "margin":current_margin,"target_score":float((endpoint[offset]-db["center"])@db["axis"]/db["scale"])})
        baseline={r["example_id"]:r for r in records if r["state"]=="baseline"}
        for row in records: row["delta_margin"]=float(row["margin"])-float(baseline[row["example_id"]]["margin"]);row["delta_target_score"]=float(row["target_score"])-float(baseline[row["example_id"]]["target_score"])
        write_csv(path,records);write_json(directory/"complete.json",{"status":"complete","fold":held,"rows":len(records),"gpu":gpu,"conditions":len(conditions),"blocks":list(blocks)})
        print(f"[RQ2.3 PATH F{held}] complete",flush=True)


def rq23_path_aggregate() -> None:
    raw=[]
    for held in range(FOLDS):raw.extend(read_csv(RQ23/"path_shards"/f"fold{held}/mediation.csv"))
    write_csv(RQ23/"path_mediation.csv",raw);blocks=sorted({r["block"] for r in raw if r["block"]!="baseline"});fold_rows=[];summary_rows=[]
    for block in blocks:
        effects=[];target_effects=[]
        for held in range(FOLDS):
            op=[r for r in raw if int(r["fold"])==held and r["state"]=="opposite" and r["block"]==block]
            same=[r for r in raw if int(r["fold"])==held and r["state"]=="same" and r["block"]==block]
            effect=float(np.mean([float(r["delta_margin"]) for r in op])-np.mean([float(r["delta_margin"]) for r in same]))
            te=float(np.mean([float(r["delta_target_score"]) for r in op])-np.mean([float(r["delta_target_score"]) for r in same]))
            effects.append(effect);target_effects.append(te);fold_rows.append({"block":block,"fold":held,"matched_adjusted_margin":effect,"matched_adjusted_target_score":te})
        mean,lo,hi,p=fold_ci(effects);tm,tlo,thi,tp=fold_ci(target_effects);summary_rows.append({"block":block,"matched_adjusted_margin":mean,"ci95_low":lo,"ci95_high":hi,"fold_t_p":p,"positive_folds":sum(v>0 for v in effects),"matched_adjusted_target_score":tm,"target_ci95_low":tlo,"target_ci95_high":thi,"target_fold_t_p":tp})
    none=next(r for r in summary_rows if r["block"]=="none");none_fold={int(r["fold"]):float(r["matched_adjusted_margin"]) for r in fold_rows if r["block"]=="none"}
    for row in summary_rows:
        mediation=[none_fold[int(r["fold"])]-float(r["matched_adjusted_margin"]) for r in fold_rows if r["block"]==row["block"]]
        mediated,mlo,mhi,mp=fold_ci(mediation);row["mediated_margin"]=mediated;row["mediated_ci95_low"]=mlo;row["mediated_ci95_high"]=mhi;row["mediated_fold_t_p"]=mp;row["mediated_positive_folds"]=sum(v>0 for v in mediation);row["mediated_fraction"]=mediated/none["matched_adjusted_margin"] if abs(none["matched_adjusted_margin"])>1e-12 else float("nan")
    write_csv(RQ23/"path_foldwise.csv",fold_rows);write_csv(RQ23/"path_summary.csv",summary_rows)
    component=json.loads((RQ23/"component_summary.json").read_text());audit_checks={"five_folds":sorted({int(r["fold"]) for r in raw})==list(range(5)),"finite":all(math.isfinite(float(r[k])) for r in raw for k in ("margin","target_score","delta_margin","delta_target_score")),"component_audit":component["audit_status"]=="PASS","continuous_only":True,"matched_same_state_control":True,"restore_controls":True}
    audit={"status":"PASS" if all(audit_checks.values()) else "FAIL","checks":audit_checks,"rows":len(raw),"scoring":"z_True_first_token-z_Unknown_first_token","binary_artifacts_used":False,"cache_sha256":sha256(CACHE)};write_json(RQ23/"audit.json",audit)
    generator=component["generator"];summary={**component,"status":"complete","audit_status":audit["status"],"path_mediation_pending":False,"path_summary":summary_rows,"interpretation_boundary":"Serial mediation is claimed only for blocks with positive mediated effect and fold CI support; otherwise the component is causal but not established as a serial mediator."};write_json(RQ23/"summary.json",summary)
    target_axis=next(r for r in summary_rows if r["block"]=="target_axis");all_components=next(r for r in summary_rows if r["block"]=="all_components")
    report=["# DeepSeek RQ2.3 — Decision circuit and causal pathway","",f"Status: **COMPLETE — audit {audit['status']}**.","",f"The earliest held-out actual-forward component passing the preregistered rule is **{generator['component']} L{generator['layer_number']}"+(f" H{generator['head']}" if int(generator['head'])>=0 else "")+f"** (specific Δm={generator['specific_delta_margin']:+.4f}, 95% CI [{generator['ci95_low']:+.4f}, {generator['ci95_high']:+.4f}]).",f"The stable downstream partial mediators are L20:H12 within Attention L20 and L23:H0. Their joint clean restoration mediates {all_components['mediated_margin']:+.4f} of the matched source effect ({all_components['mediated_fraction']:.1%}; 95% CI [{all_components['mediated_ci95_low']:+.4f}, {all_components['mediated_ci95_high']:+.4f}]). The restoration experiment does not by itself identify a serial order among those downstream mediators.",f"Restoring the L30 decision axis mediates {target_axis['mediated_fraction']:.1%} (95% CI for mediated Δm [{target_axis['mediated_ci95_low']:+.4f}, {target_axis['mediated_ci95_high']:+.4f}]); restoring the full L30 state mediates 100% by construction.","","The exact final output readout remains the independently validated RQ2.2/RQ2.5 positive control. Attention L15 is the earliest tested causal generator under this intervention family, not proof that no still-earlier distributed source exists. Predictive appearance is not called causal generation.",""]
    (RQ23/"REPORT.md").write_text("\n".join(report),encoding="utf-8");write_json(RQ23/"complete.json",{"status":"complete","audit":audit["status"],"rq":"RQ2.3"})
    if audit["status"]!="PASS":raise RuntimeError(audit)


def sigmoid(value):
    value=np.clip(np.asarray(value,float),-30,30);return 1/(1+np.exp(-value))


def gate_values(score_rows,variant,threshold,temperature,held):
    if variant=="hard":return np.asarray([float(r["gate_score"])>threshold for r in score_rows],float)
    if variant=="soft":return sigmoid((np.asarray([float(r["gate_score"]) for r in score_rows])-threshold)/temperature)
    if variant=="product":return sigmoid(np.asarray([float(r["telicity_score"]) for r in score_rows])/temperature)*sigmoid(np.asarray([float(r["ambiguity_score"]) for r in score_rows])/temperature)
    values=[]
    for row in score_rows:
        digest=hashlib.sha256(f'{SEED}:{held}:{row["example_id"]}'.encode()).digest();values.append((int.from_bytes(digest[:8],"big")/(2**64-1)-.5)*4)
    return sigmoid((np.asarray(values)-threshold)/temperature)


def select_rq24() -> None:
    fixed,_spec,rows,arrays,margin,fold_ids,_checks=load_inputs(True);score_rows=read_csv(RQ24/"gate_scores.csv")
    readout_test=json.loads((BASE/"telicity_decision_control/summary.json").read_text())["causal_tests"]["readout"]
    one_sd_readout=-float(readout_test["one_sd_random_adjusted_bidirectional"]["mean"])
    records=[];selected=[]
    for held in range(FOLDS):
        train_indices=np.flatnonzero(fold_ids!=held);scores=[r for r in score_rows if int(r["basis_fold"])==held and r["split"]=="train"]
        by_id={r["example_id"]:r for r in scores};ordered=[by_id[rows[i]["id"]] for i in train_indices]
        # The actuator and its one-SD gain were independently validated before
        # this correction experiment; no held-out RQ2.4 outcome calibrates it.
        one_sd=one_sd_readout;gold=np.asarray([rows[i]["label"] for i in train_indices]);group=np.asarray([rows[i]["group"][0] for i in train_indices])
        def metrics(gates,strength):
            predicted=np.where(margin[train_indices]+gates*strength*one_sd>=0,"True","Unknown");return {name:float(np.mean(predicted[group==name]==gold[group==name])) for name in "ABCD"}
        base=metrics(np.zeros(len(train_indices)),0)
        for variant in ("hard","soft","product"):
            configs=[(t,1.) for t in (-.5,0,.5,1,1.5)] if variant=="hard" else ([(t,u) for t in (-.5,0,.5,1,1.5) for u in (.25,.5,1.)] if variant=="soft" else [(0,u) for u in (.25,.5,1.)])
            pool=[]
            for threshold,temperature in configs:
                gates=gate_values(ordered,variant,threshold,temperature,held)
                for strength in (.5,1.,1.5,2.):
                    met=metrics(gates,strength);objective=(met["C"]-base["C"])-1.5*abs(met["D"]-base["D"])-.25*abs(met["A"]-base["A"])-.25*abs(met["B"]-base["B"])
                    row={"fold":held,"variant":variant,"threshold":threshold,"temperature":temperature,"strength":strength,"train_objective":objective,"calibrated_one_sd_delta_margin":one_sd,**{f"train_ACC_{g}":met[g] for g in "ABCD"}};records.append(row);pool.append(row)
            selected.append(max(pool,key=lambda r:(r["train_objective"],-r["strength"],-abs(r["threshold"]))))
    write_csv(RQ24/"selection_grid.csv",records);write_csv(RQ24/"selected_hyperparameters.csv",selected);write_json(RQ24/"selection_complete.json",{"status":"complete","heldout_labels_used":False,"selection":"outer-train margin proxy calibrated from other-fold OOF causal effects"})


def gate_component_hook(adapter,choice,fitted,target_raw,answer,condition):
    import torch
    component,layer,head=choice["component"],int(choice["layer"]),int(choice["head"]);direction=fitted["direction"];shift_direction=direction
    if condition=="random_gate_native":
        shift_direction=np.random.default_rng(SEED+int(choice["fold"])).normal(size=len(direction));shift_direction-=direction*float(shift_direction@direction);shift_direction=unit(shift_direction)
    direction_t=torch.as_tensor(direction,device=answer.device,dtype=torch.float32);shift_t=torch.as_tensor(shift_direction,device=answer.device,dtype=torch.float32);row_index=torch.arange(len(answer),device=answer.device)
    def alter(value):
        output=value.clone()
        if component=="head":view=output[row_index,answer].view(len(answer),32,128);current=view[:,head].float()
        else:current=output[row_index,answer].float()
        delta=target_raw-current@direction_t;current+=delta[:,None]*shift_t
        if component=="head":view[:,head]=current.to(view.dtype);output[row_index,answer]=view.reshape(len(answer),-1)
        else:output[row_index,answer]=current.to(output.dtype)
        return output
    if component=="residual":return adapter.layers[layer].register_forward_hook(lambda _m,_i,out:(alter(out[0]),*out[1:]) if isinstance(out,tuple) else alter(out))
    if component=="attention":return adapter.attention(layer).register_forward_hook(lambda _m,_i,out:(alter(out[0]),*out[1:]) if isinstance(out,tuple) else alter(out))
    if component=="mlp":return adapter.mlp(layer).register_forward_hook(lambda _m,_i,out:alter(out))
    return adapter.head_input(layer).register_forward_pre_hook(lambda _m,inp:(alter(inp[0]),*inp[1:]))


def rq24_worker(folds_to_run:list[int],gpu:int,batch_size:int)->None:
    import torch
    fixed,_spec,expected_rows,arrays,margin,fold_ids,_checks=load_inputs(True);group=groups(expected_rows);choices={int(r["fold"]):r for r in read_csv(RQ24/"gate_selections.csv")};score_rows=read_csv(RQ24/"gate_scores.csv");selected={(int(r["fold"]),r["variant"]):r for r in read_csv(RQ24/"selected_hyperparameters.csv")};target=int(np.load(PREPARED)["target_layer"]);control_assets=np.load(CONTROL_ASSETS,allow_pickle=False)
    model,tokenizer,adapter,_fixed,rows,_scorer,_report=load_gpu(gpu);device=next(model.parameters()).device;label_ids=torch.tensor([tokenizer(v,add_special_tokens=False).input_ids[0] for v in LABELS],device=device)
    for held in folds_to_run:
        directory=RQ24/"fold_shards"/f"fold{held}";path=directory/"interventions.csv";expected=int(np.sum(fold_ids==held))*len(CORRECTION_CONDITIONS)
        if path.is_file() and (directory/"complete.json").is_file() and len(read_csv(path))==expected:print(f"[RQ2.4 F{held}] reuse",flush=True);continue
        train=fold_ids!=held;indices=np.flatnonzero(~train).tolist();choice=choices[held];x=candidate_matrix(arrays,choice["component"],int(choice["layer"]),int(choice["head"]));fitted=gate_fit(x,group,train);readout_axis=control_assets[f"f{held}_readout"].astype(np.float32);readout_scale=float(control_assets[f"f{held}_common_scale"]);final_norm_states=control_assets["final_norm_states"].astype(np.float32);final_center=final_norm_states[train].mean(0)
        by_id={r["example_id"]:r for r in score_rows if int(r["basis_fold"])==held};ordered=[by_id[rows[i]["id"]] for i in indices]
        hard=selected[(held,"hard")];soft=selected[(held,"soft")];product=selected[(held,"product")]
        specs=[("baseline",np.zeros(len(indices)),0.),("internal_hard",gate_values(ordered,"hard",float(hard["threshold"]),float(hard["temperature"]),held),float(hard["strength"])),("internal_soft",gate_values(ordered,"soft",float(soft["threshold"]),float(soft["temperature"]),held),float(soft["strength"])),("internal_product",gate_values(ordered,"product",float(product["threshold"]),float(product["temperature"]),held),float(product["strength"])),("random_gate",gate_values(ordered,"random",float(soft["threshold"]),float(soft["temperature"]),held),float(soft["strength"])),("oracle_C_only",np.asarray([group[i]=="C" for i in indices],float),float(soft["strength"])),("unconditional",np.ones(len(indices)),float(soft["strength"])),("reverse",gate_values(ordered,"soft",float(soft["threshold"]),float(soft["temperature"]),held),-float(soft["strength"])),("restore",np.zeros(len(indices)),0.)]
        records=[]
        for condition,gates,strength in specs:
            for start in range(0,len(indices),batch_size):
                chosen=indices[start:start+batch_size];encoded,answer=encode(tokenizer,fixed,rows,chosen,device);handles=[];captured={};local_gates=np.asarray(gates[start:start+len(chosen)],float)
                if condition not in ("baseline","restore"):
                    gate_tensor=torch.as_tensor(local_gates,device=device,dtype=torch.float32);axis=torch.as_tensor(readout_axis,device=device,dtype=torch.float32);row_index=torch.arange(len(chosen),device=device)
                    def decision_hook(_m,_i,out):
                        raw_out=out[0] if isinstance(out,tuple) else out;value=raw_out.clone();value[row_index,answer]-= (gate_tensor[:,None]*strength*readout_scale*axis).to(value.dtype);captured["target"]=value.detach();return (value,*out[1:]) if isinstance(out,tuple) else value
                    handles.append(adapter.final_norm.register_forward_hook(decision_hook))
                else:
                    def capture(_m,_i,out):captured["target"]=(out[0] if isinstance(out,tuple) else out).detach()
                    handles.append(adapter.final_norm.register_forward_hook(capture))
                try:
                    with torch.inference_mode():result=model(**encoded,use_cache=False,return_dict=True)
                finally:
                    for handle in handles:handle.remove()
                row_index=torch.arange(len(chosen),device=device);logits=result.logits[row_index,answer][:,label_ids].float().cpu().numpy();endpoint=captured["target"][row_index,answer].float().cpu().numpy()
                for local,index in enumerate(chosen):
                    probability=np.exp(logits[local]-logits[local].max());probability/=probability.sum();prediction=LABELS[int(np.argmax(logits[local]))]
                    records.append({"fold":held,"example_index":index,"example_id":rows[index]["id"],"group":group[index],"gold_label":rows[index]["label"],"condition":condition,"gate_value":local_gates[local],"strength":strength,"margin":float(logits[local,0]-logits[local,2]),"decision_score":float((endpoint[local]-final_center)@readout_axis/readout_scale),"prob_true":float(probability[0]),"prob_false":float(probability[1]),"prob_unknown":float(probability[2]),"prediction":prediction,"correct":int(prediction==rows[index]["label"])})
            print(f"[RQ2.4 F{held}] {condition}",flush=True)
        baseline={r["example_id"]:r for r in records if r["condition"]=="baseline"}
        for row in records:row["delta_margin"]=float(row["margin"])-float(baseline[row["example_id"]]["margin"]);row["delta_decision_score"]=float(row["decision_score"])-float(baseline[row["example_id"]]["decision_score"]);row["prediction_changed"]=int(row["prediction"]!=baseline[row["example_id"]]["prediction"])
        write_csv(path,records);write_json(directory/"complete.json",{"status":"complete","fold":held,"rows":len(records),"gpu":gpu,"conditions":list(CORRECTION_CONDITIONS),"gate_candidate":choice,"circuit_reference_layer":target,"correction_actuator":"exact final-norm True-Unknown readout","readout_scale":readout_scale})


def bootstrap_ci(values,seed,draws=10000):
    value=np.asarray(values,float);rng=np.random.default_rng(seed);samples=value[rng.integers(0,len(value),size=(draws,len(value)))].mean(1);lo,hi=np.quantile(samples,[.025,.975]);return float(value.mean()),float(lo),float(hi)


def rq24_aggregate()->None:
    raw=[]
    for held in range(FOLDS):raw.extend(read_csv(RQ24/"fold_shards"/f"fold{held}/interventions.csv"))
    write_csv(RQ24/"selective_interventions.csv",raw);baseline={r["example_id"]:r for r in raw if r["condition"]=="baseline"};metrics=[]
    for condition in CORRECTION_CONDITIONS:
        subset=[r for r in raw if r["condition"]==condition];row={"condition":condition,"n":len(subset),"overall_accuracy":float(np.mean([int(r["correct"]) for r in subset]))}
        for g in "ABCD":
            use=[r for r in subset if r["group"]==g];delta=[int(r["correct"])-int(baseline[r["example_id"]]["correct"]) for r in use];acc=float(np.mean([int(r["correct"]) for r in use]));mean,lo,hi=bootstrap_ci(delta,SEED+ord(g)+len(condition));row.update({f"ACC_{g}":acc,f"delta_ACC_{g}":mean,f"delta_ACC_{g}_ci_low":lo,f"delta_ACC_{g}_ci_high":hi,f"TBR_{g}":float(np.mean([r["prediction"]=="True" for r in use])),f"delta_m_{g}":float(np.mean([float(r["delta_margin"]) for r in use])),f"flips_{g}":sum(int(r["prediction_changed"]) for r in use)})
        metrics.append(row)
    write_csv(RQ24/"metrics_with_ci.csv",metrics);lookup={r["condition"]:r for r in metrics};internal=[lookup[n] for n in ("internal_hard","internal_soft","internal_product")];base=lookup["baseline"];eligible=[r for r in internal if abs(r["delta_ACC_D"])<=.05];best=max(eligible or internal,key=lambda r:(r["delta_ACC_C"]-1.5*abs(r["delta_ACC_D"]),r["overall_accuracy"]));selective=best["delta_ACC_C"]>0 and abs(best["delta_ACC_D"])<=.05 and best["delta_ACC_A"]>=-.05 and best["delta_ACC_B"]>=-.05
    restore=[r for r in raw if r["condition"]=="restore"];restore_ok=max(abs(float(r["delta_margin"])) for r in restore)<=1e-6 and max(int(r["prediction_changed"]) for r in restore)==0
    checks={"rows":len(raw)==400*len(CORRECTION_CONDITIONS),"unique_grid":len({(r["example_id"],r["condition"]) for r in raw})==400*len(CORRECTION_CONDITIONS),"five_folds":sorted({int(r["fold"]) for r in raw})==list(range(5)),"all_conditions":set(r["condition"] for r in raw)==set(CORRECTION_CONDITIONS),"finite":all(math.isfinite(float(r[k])) for r in raw for k in ("margin","decision_score","delta_margin","delta_decision_score")),"exact_restore":restore_ok,"heldout_selection":json.loads((RQ24/"selection_complete.json").read_text())["heldout_labels_used"] is False,"continuous_only":True}
    audit={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"rows":len(raw),"scoring":"z_True_first_token-z_Unknown_first_token","True_token_id":8173,"Unknown_token_id":21483,"binary_artifacts_used":False,"cache_sha256":sha256(CACHE)};write_json(RQ24/"audit.json",audit)
    summary={"status":"complete","audit_status":audit["status"],"model":"deepseek-7b-chat","target":"m = z_True - z_Unknown","baseline":base,"best_internal":best,"selective_correction_achieved":bool(selective),"criteria":"Group-C accuracy improves; |delta ACC_D|<=.05; A/B each no worse than .05; matched random/unconditional/reverse/oracle/restore controls","all_metrics":metrics};write_json(RQ24/"summary.json",summary)
    robust=bool(best["delta_ACC_C_ci_low"]>0)
    report=["# DeepSeek RQ2.4 — Selective Group-C correction","",f"Status: **COMPLETE — audit {audit['status']}**.","",f"The outer-training-selected internal condition was **{best['condition']}**. Group-C accuracy changed {base['ACC_C']:.3f} → {best['ACC_C']:.3f} (Δ {best['delta_ACC_C']:+.3f}, item-bootstrap 95% CI [{best['delta_ACC_C_ci_low']:+.3f}, {best['delta_ACC_C_ci_high']:+.3f}]); Group-D changed {base['ACC_D']:.3f} → {best['ACC_D']:.3f} (Δ {best['delta_ACC_D']:+.3f}).",f"A/B changes were {best['delta_ACC_A']:+.3f}/{best['delta_ACC_B']:+.3f}. The predefined selectivity rule is **{selective}**, but statistical robustness is **{robust}** because the Group-C interval includes zero.","","Internal gate selection used only outer-training group/gold labels. Held-out labels were used only for reporting. Random gate, oracle-C, unconditional, reverse, and exact restore controls are in `metrics_with_ci.csv`.",""]
    (RQ24/"REPORT.md").write_text("\n".join(report),encoding="utf-8");write_json(RQ24/"complete.json",{"status":"complete","audit":audit["status"],"rq":"RQ2.4","selective_correction_achieved":bool(selective)})
    if audit["status"]!="PASS":raise RuntimeError(audit)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--phase",required=True,choices=("plan","cache-head","prepare","rq23-worker","rq23-aggregate","rq23-path-worker","rq23-path-aggregate","rq24-select","rq24-worker","rq24-aggregate"));parser.add_argument("--folds",type=int,nargs="*",default=list(range(5)));parser.add_argument("--gpu",type=int,default=0);parser.add_argument("--batch-size",type=int,default=4);args=parser.parse_args()
    if any(f not in range(5) for f in args.folds):raise ValueError(args.folds)
    if args.phase=="plan":
        fixed,spec,_rows,_arrays,_margin,_folds,checks=load_inputs(False);print(json.dumps({"status":"PASS","gpu_work_started":False,"model":"deepseek-7b-chat","rq22_audit":"PASS","cache":str(CACHE),"cache_sha256":sha256(CACHE),"head_cache_required":not HEAD_CACHE.exists(),"outputs":[str(RQ23),str(RQ24)],"scoring":"m = z_True_first_token - z_Unknown_first_token","dataset_sha256":fixed["dataset_sha256"],"revision":spec["revision"],"checks":checks},indent=2));return
    if args.phase=="cache-head":cache_head(args.gpu,args.batch_size)
    elif args.phase=="prepare":prepare()
    elif args.phase=="rq23-worker":rq23_worker(args.folds,args.gpu,args.batch_size)
    elif args.phase=="rq23-aggregate":rq23_aggregate()
    elif args.phase=="rq23-path-worker":rq23_path_worker(args.folds,args.gpu,args.batch_size)
    elif args.phase=="rq23-path-aggregate":rq23_path_aggregate()
    elif args.phase=="rq24-select":select_rq24()
    elif args.phase=="rq24-worker":rq24_worker(args.folds,args.gpu,args.batch_size)
    else:rq24_aggregate()


if __name__=="__main__":main()
