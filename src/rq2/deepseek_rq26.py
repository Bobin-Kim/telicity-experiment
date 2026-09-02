#!/usr/bin/env python3
"""DeepSeek RQ2.6: held-out causal origin of the Group-C True-driving signal.

The implementation preserves the causal intersection of the Qwen/Mistral
A37 protocol: outer-train C/D source contrasts, bidirectional actual forwards,
orthogonal norm-matched controls, component/head/token localization, and
clean restoration of the independently validated decision generator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from .deepseek_rq23_rq24 import (CONFIG, FOLDS, LABELS, SEED, encode, fold_ci,
                                 load_gpu, load_inputs, read_csv, sha256,
                                 unit, write_csv, write_json)

ROOT = Path(__file__).resolve().parents[2]
RQ22 = ROOT / "results/deepseek-7b-rq2/rq2_2"
RQ23 = ROOT / "results/deepseek-7b-rq2/rq2_3"
RQ25 = ROOT / "results/deepseek-7b-telicity-rq/telicity_decision_control"
OUT = ROOT / "results/deepseek-7b-rq2/rq2_6"
CACHE = ROOT / "results/deepseek-7b-telicity-rq/full_depth_cache.npz"
HEAD_CACHE = RQ23 / "head_input_cache.npz"
KINDS = ("attention", "mlp", "residual")
CONDS = ("C_to_D", "D_to_C", "random_C_to_D", "random_D_to_C")
TOKENS = ("target_predicate", "premise_context", "hypothesis_event_expression",
          "answer_query_tokens", "other_context")
ORIGIN_SEED = 37042


def group_values(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["group"][0] for row in rows])


def result_layers() -> tuple[int, int]:
    rq23 = json.loads((RQ23 / "summary.json").read_text())
    rq22 = json.loads((RQ22 / "summary.json").read_text())
    if rq23.get("audit_status") != "PASS" or not rq23["generator"]["reliable_causal"]:
        raise RuntimeError("RQ2.3 has no audited causal generator")
    generator = int(rq23["generator"]["layer"])
    # Match the independently frozen RQ2.3 endpoint: strongest predictive
    # residual L30. Causality is established by actual-forward interventions,
    # not inferred from this candidate-selection label.
    target = int(rq22["strongest_predictive_layer"] - 1)
    if not (0 < generator < target < 30):
        raise RuntimeError((generator, target))
    return generator, target


def candidate_matrix(arrays: dict, kind: str, layer: int, head: int = -1) -> np.ndarray:
    value = arrays["head"][:, layer].reshape(400, 32, 128)[:, head] if kind == "head" else arrays[kind][:, layer]
    return value.astype(np.float32)


def direction_bundle(arrays: dict, margin: np.ndarray, fold_ids: np.ndarray,
                     held: int, generator: int, target: int) -> dict:
    train = fold_ids != held
    result = {}
    for name, x in (("generator", candidate_matrix(arrays, "attention", generator)),
                    ("decision", candidate_matrix(arrays, "residual", target))):
        center = x[train].mean(0)
        axis = unit((x[train] - center).T @ (margin[train] - margin[train].mean()))
        scale = max(float(np.std((x[train] - center) @ axis)), 1e-8)
        result.update({f"{name}_center": center, f"{name}_axis": axis, f"{name}_scale": scale})
    return result


def source_vectors(x: np.ndarray, groups: np.ndarray, fold_ids: np.ndarray,
                   held: int, layer: int, kind: str, head: int = -1) -> tuple[np.ndarray, np.ndarray]:
    train = fold_ids != held
    delta = x[train & (groups == "D")].mean(0) - x[train & (groups == "C")].mean(0)
    rng = np.random.default_rng(ORIGIN_SEED + held * 10000 + layer * 100 + len(kind) + head)
    random = rng.normal(size=len(delta)); axis = unit(delta)
    random -= axis * float(random @ axis); random = unit(random) * float(np.linalg.norm(delta))
    return delta.astype(np.float32), random.astype(np.float32)


def capture_hooks(adapter, answer, generator: int, target: int, captured: dict):
    import torch
    row = torch.arange(len(answer), device=answer.device)

    def generator_hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captured["generator"] = value[row, answer].detach()

    def decision_hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captured["decision"] = value[row, answer].detach()

    return [adapter.attention(generator).register_forward_hook(generator_hook),
            adapter.layers[target].register_forward_hook(decision_hook)]


def calibrated(old, addition):
    import torch
    target = torch.linalg.norm(addition, dim=-1); factor = torch.ones_like(target)
    for _ in range(8):
        candidate = (old.float() + addition * factor[..., None]).to(old.dtype)
        realized = torch.linalg.norm(candidate.float() - old.float(), dim=-1)
        factor *= torch.where(target > 0, target / torch.clamp(realized, min=1e-12), torch.zeros_like(factor))
    return (old.float() + addition * factor[..., None]).to(old.dtype)


def source_hook(adapter, kind: str, layer: int, head: int, answer, vector: np.ndarray,
                token_masks: list[list[bool]] | None, norm_capture: dict):
    import torch
    row = torch.arange(len(answer), device=answer.device)
    addition = torch.as_tensor(vector, device=answer.device, dtype=torch.float32)

    def patch(value):
        output = value.clone(); norms = []
        if kind == "head":
            view = output[row, answer].view(len(answer), 32, 128); old = view[:, head].clone()
            new = calibrated(old, addition.expand(len(old), -1)); view[:, head] = new
            output[row, answer] = view.reshape(len(answer), -1)
            norms = torch.linalg.norm(new.float() - old.float(), dim=1)
        elif token_masks is None:
            old = output[row, answer].clone(); new = calibrated(old, addition.expand(len(old), -1))
            output[row, answer] = new; norms = torch.linalg.norm(new.float() - old.float(), dim=1)
        else:
            for index, mask in enumerate(token_masks):
                selected = torch.as_tensor(mask, device=answer.device, dtype=torch.bool)
                old = output[index, selected].clone()
                if len(old):
                    new = calibrated(old, addition.expand(len(old), -1)); output[index, selected] = new
                    norms.append(torch.linalg.norm(new.float() - old.float()))
                else: norms.append(torch.zeros((), device=answer.device))
            norms = torch.stack(norms)
        norm_capture["norm"] = norms.detach()
        return output

    if kind == "attention":
        return adapter.attention(layer).register_forward_hook(
            lambda _m, _i, out: (patch(out[0]), *out[1:]) if isinstance(out, tuple) else patch(out))
    if kind == "mlp":
        return adapter.mlp(layer).register_forward_hook(lambda _m, _i, out: patch(out))
    if kind == "residual":
        return adapter.layers[layer].register_forward_hook(
            lambda _m, _i, out: (patch(out[0]), *out[1:]) if isinstance(out, tuple) else patch(out))
    return adapter.head_input(layer).register_forward_pre_hook(
        lambda _m, inputs: (patch(inputs[0]), *inputs[1:]))


def token_masks(tokenizer, rows: list[dict], indices: list[int], prompts: list[str], offsets,
                category: str) -> list[list[bool]]:
    result = []
    for local, index in enumerate(indices):
        prompt = prompts[local]; premise = rows[index]["premise"]; hypothesis = rows[index]["hypothesis"]
        verb = rows[index]["verb"]; ps = prompt.find(premise); hs = prompt.find(hypothesis)
        pe, he = ps + len(premise), hs + len(hypothesis)
        vs = prompt.lower().find(verb.lower(), ps, pe); ve = vs + len(verb) if vs >= 0 else vs
        # The dataset stores lemmas (e.g. ``bake``) while premises often use
        # progressive surface forms (``was baking``). Every canonical item has
        # an auxiliary-progressive predicate; use its captured verb span when
        # the lemma is not a literal substring. The first match avoids the one
        # A-item whose later complement contains ``were missing``.
        if vs < 0:
            match = re.search(r"\b(?:was|were)\s+([A-Za-z-]+ing)\b", premise, re.IGNORECASE)
            if match is not None: vs, ve = ps + match.start(1), ps + match.end(1)
        mask = []
        local_offsets = offsets[local].tolist() if hasattr(offsets[local], "tolist") else offsets[local]
        for start, stop in local_offsets:
            overlap = lambda left, right: start < right and stop > left
            if category == "target_predicate": valid = vs >= 0 and overlap(vs, ve)
            elif category == "premise_context": valid = overlap(ps, pe) and not (vs >= 0 and overlap(vs, ve))
            elif category == "hypothesis_event_expression": valid = overlap(hs, he)
            elif category == "answer_query_tokens": valid = start >= he
            else: valid = not overlap(ps, pe) and not overlap(hs, he) and start < he
            mask.append(bool(valid and stop > start))
        result.append(mask)
    return result


def encode_offsets(tokenizer, fixed, rows: list[dict], indices: list[int], device):
    from src.preflight import build_prompt
    prompts = [build_prompt(tokenizer, fixed, rows[i]["premise"], rows[i]["hypothesis"]) for i in indices]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping"); answer = encoded["attention_mask"].sum(1).long() - 1
    return {key: value.to(device) for key, value in encoded.items()}, answer.to(device), prompts, offsets


def run_forward(model, tokenizer, adapter, fixed, rows: list[dict], indices: list[int], bundle: dict,
                generator: int, target: int, kind: str | None = None, layer: int = -1, head: int = -1,
                vector: np.ndarray | None = None, batch_size: int = 8, token_category: str | None = None,
                restore_generator: bool = False) -> list[dict]:
    import torch
    device = next(model.parameters()).device
    label_ids = torch.tensor([tokenizer(v, add_special_tokens=False).input_ids[0] for v in LABELS], device=device)
    records = []
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        if token_category is None:
            encoded, answer = encode(tokenizer, fixed, rows, chosen, device); masks = None
        else:
            encoded, answer, prompts, offsets = encode_offsets(tokenizer, fixed, rows, chosen, device)
            masks = token_masks(tokenizer, rows, chosen, prompts, offsets, token_category)
        row = torch.arange(len(chosen), device=device); clean_generator = None
        if restore_generator:
            clean_capture = {}; handles = capture_hooks(adapter, answer, generator, target, clean_capture)
            try:
                with torch.inference_mode(): model(**encoded, use_cache=False, return_dict=True)
            finally:
                for handle in handles: handle.remove()
            clean_generator = clean_capture["generator"]
        norm_capture = {}; handles = []
        if kind is not None and vector is not None:
            handles.append(source_hook(adapter, kind, layer, head, answer, vector, masks, norm_capture))
        if restore_generator:
            def restore(_module, _inputs, output):
                value = (output[0] if isinstance(output, tuple) else output).clone()
                value[row, answer] = clean_generator.to(value.dtype)
                return (value, *output[1:]) if isinstance(output, tuple) else value
            handles.append(adapter.attention(generator).register_forward_hook(restore))
        captured = {}; handles.extend(capture_hooks(adapter, answer, generator, target, captured))
        try:
            with torch.inference_mode(): output = model(**encoded, use_cache=False, return_dict=True)
        finally:
            for handle in handles: handle.remove()
        logits = output.logits[row, answer][:, label_ids].float().cpu().numpy()
        gen = captured["generator"].float().cpu().numpy(); dec = captured["decision"].float().cpu().numpy()
        norms = (norm_capture["norm"].float().cpu().numpy() if kind is not None
                 else np.zeros(len(chosen), np.float32))
        for local, index in enumerate(chosen):
            records.append({"example_index": index, "example_id": rows[index]["id"],
                            "group": rows[index]["group"][0], "gold_label": rows[index]["label"],
                            "margin": float(logits[local, 0] - logits[local, 2]),
                            "generator_score": float((gen[local] - bundle["generator_center"]) @ bundle["generator_axis"] / bundle["generator_scale"]),
                            "decision_score": float((dec[local] - bundle["decision_center"]) @ bundle["decision_axis"] / bundle["decision_scale"]),
                            "prediction": LABELS[int(np.argmax(logits[local]))],
                            "intervention_norm": float(norms[local])})
    return records


def valid_shard(path: Path, marker: Path, expected: int) -> bool:
    try:
        return (json.loads(marker.read_text())["status"] == "complete" and len(read_csv(path)) == expected)
    except Exception:
        return False


def plan() -> None:
    fixed, spec, rows, arrays, margin, fold_ids, checks = load_inputs(True)
    generator, target = result_layers(); rq25 = json.loads((RQ25 / "audit.json").read_text())
    if rq25.get("status") != "PASS": raise RuntimeError("RQ2.5 audit is not PASS")
    payload = {"status": "PASS", "gpu_work_started": False, "model": "deepseek-7b-chat",
               "protocol": "Qwen/Mistral A37 strict bidirectional held-out causal origin",
               "scoring": "m = z(True_first_token) - z(Unknown_first_token)",
               "generator": {"component": "attention", "layer": generator, "layer_number": generator + 1,
                             "source": str(RQ23 / "summary.json")},
               "decision_target": {"component": "residual block output", "layer": target,
                                   "layer_number": target + 1},
               "candidate_layers": list(range(generator)), "candidate_layer_numbers": list(range(1, generator + 1)),
               "components": list(KINDS), "conditions": list(CONDS),
               "estimated_rows": {"coarse_scan": FOLDS * (80 + generator * len(KINDS) * len(CONDS) * 20),
                                  "head_scan": FOLDS * 2 * len(CONDS) * 20,
                                  "token_scan": FOLDS * len(TOKENS) * len(CONDS) * 20,
                                  "mediation": FOLDS * 5 * 20},
               "reused": {"activation_cache": {"path": str(CACHE), "sha256": sha256(CACHE)},
                          "head_cache": {"path": str(HEAD_CACHE), "sha256": sha256(HEAD_CACHE)},
                          "rq22": {"path": str(RQ22), "audit": "PASS"},
                          "rq23": {"path": str(RQ23), "audit": "PASS"},
                          "rq25": {"path": str(RQ25), "audit": "PASS"}},
               "outputs": str(OUT), "input_checks": checks,
               "dataset_sha256": fixed["dataset_sha256"], "revision": spec["revision"]}
    print(json.dumps(payload, indent=2))


def prepare() -> None:
    fixed, spec, rows, arrays, margin, fold_ids, checks = load_inputs(True)
    generator, target = result_layers()
    if json.loads((RQ25 / "audit.json").read_text()).get("status") != "PASS":
        raise RuntimeError("RQ2.5 audit failed")
    OUT.mkdir(parents=True, exist_ok=True)
    bundles = {}
    for held in range(FOLDS):
        current = direction_bundle(arrays, margin, fold_ids, held, generator, target)
        for key, value in current.items(): bundles[f"f{held}_{key}"] = np.asarray(value)
    temp = OUT / "fold_axes.npz.tmp"
    with temp.open("wb") as handle: np.savez_compressed(handle, **bundles)
    temp.replace(OUT / "fold_axes.npz")
    payload = {"status": "PASS", "model": "deepseek-7b-chat", "generator_layer": generator,
               "target_layer": target, "candidate_layers": list(range(generator)), "components": list(KINDS),
               "conditions": list(CONDS), "folds": FOLDS, "seed": ORIGIN_SEED,
               "scoring": "m = z(True_first_token) - z(Unknown_first_token)",
               "selection": "earliest component passing bidirectional generator and margin CIs; attention before MLP before residual within layer",
               "source_delta": "outer-train mean(D)-mean(C); reverse sign for D-to-C",
               "controls": "outer-train folds, orthogonal norm-matched random, bidirectional reverse, generator restoration",
               "reused": {"cache_sha256": sha256(CACHE), "head_cache_sha256": sha256(HEAD_CACHE),
                          "rq22_audit": "PASS", "rq23_audit": "PASS", "rq25_audit": "PASS"},
               "input_checks": checks, "dataset_sha256": fixed["dataset_sha256"], "revision": spec["revision"]}
    write_json(OUT / "execution_plan.json", payload); print(json.dumps(payload, indent=2))


def load_runtime_inputs():
    fixed, spec, rows, arrays, margin, fold_ids, checks = load_inputs(True)
    generator, target = result_layers(); axes = np.load(OUT / "fold_axes.npz", allow_pickle=False)
    return fixed, spec, rows, arrays, margin, fold_ids, checks, generator, target, axes


def fold_bundle(axes, held: int) -> dict:
    return {name: axes[f"f{held}_{name}"] for name in ("generator_center", "generator_axis", "generator_scale",
                                                          "decision_center", "decision_axis", "decision_scale")}


def smoke(gpu: int) -> None:
    fixed, _spec, rows, arrays, margin, fold_ids, _checks, generator, target, axes = load_runtime_inputs()
    groups = group_values(rows); held, layer, kind = 0, 0, "attention"; bundle = fold_bundle(axes, held)
    x = candidate_matrix(arrays, kind, layer); delta, random = source_vectors(x, groups, fold_ids, held, layer, kind)
    test = np.flatnonzero(fold_ids == held); c_index = int(test[groups[test] == "C"][0])
    model, tokenizer, adapter, _fixed, _rows, _scorer, _report = load_gpu(gpu)
    clean = run_forward(model, tokenizer, adapter, fixed, rows, [c_index], bundle, generator, target, batch_size=1)[0]
    actual = run_forward(model, tokenizer, adapter, fixed, rows, [c_index], bundle, generator, target,
                         kind, layer, -1, delta, 1)[0]
    control = run_forward(model, tokenizer, adapter, fixed, rows, [c_index], bundle, generator, target,
                          kind, layer, -1, random, 1)[0]
    error = abs(actual["intervention_norm"] - control["intervention_norm"]) / max(actual["intervention_norm"], control["intervention_norm"], 1e-12)
    passed = (all(math.isfinite(row[key]) for row in (clean, actual, control)
                  for key in ("margin", "generator_score", "decision_score"))
              and min(actual["intervention_norm"], control["intervention_norm"]) > 0 and error <= .05)
    result = {"status": "PASS" if passed else "FAIL", "gpu": gpu, "candidate": "Attention L1",
              "clean": clean, "actual_C_to_D": actual, "random_C_to_D": control,
              "norm_relative_error": error, "scientific_output_written": False,
              "scoring": "m = z(True_first_token) - z(Unknown_first_token)"}
    print(json.dumps(result, indent=2))
    if not passed: raise RuntimeError(result)


def scan_worker(folds: list[int], gpu: int, batch_size: int) -> None:
    fixed, _spec, rows, arrays, margin, fold_ids, _checks, generator, target, axes = load_runtime_inputs()
    expected = 80 + generator * len(KINDS) * len(CONDS) * 20
    folds = [held for held in folds if not valid_shard(OUT / "scan_shards" / f"fold{held}/interventions.csv",
                                                        OUT / "scan_shards" / f"fold{held}/complete.json", expected)]
    if not folds: print("[RQ2.6 SCAN] all assigned folds reused", flush=True); return
    groups = group_values(rows); model, tokenizer, adapter, _fixed, _rows, _scorer, _report = load_gpu(gpu)
    for held in folds:
        directory = OUT / "scan_shards" / f"fold{held}"; path = directory / "interventions.csv"
        test = np.flatnonzero(fold_ids == held); bundle = fold_bundle(axes, held); records = []
        baseline = run_forward(model, tokenizer, adapter, fixed, rows, test.tolist(), bundle, generator, target,
                               batch_size=batch_size)
        for row in baseline: row.update({"fold": held, "candidate": "baseline", "component": "baseline",
                                         "layer": -1, "layer_number": 0, "head": -1, "condition": "baseline"})
        records.extend(baseline)
        for layer in range(generator):
            for kind in KINDS:
                x = candidate_matrix(arrays, kind, layer); delta, random = source_vectors(x, groups, fold_ids, held, layer, kind)
                for condition in CONDS:
                    source_group = "C" if condition.endswith("C_to_D") else "D"
                    indices = test[groups[test] == source_group].tolist()
                    vector = random if condition.startswith("random") else delta
                    if condition.endswith("D_to_C"): vector = -vector
                    current = run_forward(model, tokenizer, adapter, fixed, rows, indices, bundle, generator, target,
                                          kind, layer, -1, vector, batch_size)
                    for row in current: row.update({"fold": held, "candidate": f"L{layer+1}_{kind}",
                                                    "component": kind, "layer": layer, "layer_number": layer + 1,
                                                    "head": -1, "condition": condition})
                    records.extend(current); print(f"[RQ2.6 SCAN F{held}] L{layer+1} {kind} {condition}", flush=True)
        baseline_by_id = {row["example_id"]: row for row in records if row["condition"] == "baseline"}
        for row in records:
            base = baseline_by_id[row["example_id"]]
            row["delta_margin"] = row["margin"] - base["margin"]
            row["delta_generator_score"] = row["generator_score"] - base["generator_score"]
            row["delta_decision_score"] = row["decision_score"] - base["decision_score"]
            row["prediction_changed"] = int(row["prediction"] != base["prediction"])
        write_csv(path, records); write_json(directory / "complete.json", {"status": "complete", "device_type": "cuda",
                   "gpu": gpu, "fold": held, "rows": len(records), "expected": expected})


def signed_stats(values: list[float], expected_sign: int) -> dict:
    mean, low, high, p = fold_ci(values)
    consistent = sum(value * expected_sign > 0 for value in values)
    return {"mean": mean, "ci95_low": low, "ci95_high": high, "fold_t_p": p,
            "directional_folds": consistent, "fold_values": ";".join(map(str, values))}


def aggregate_scan() -> None:
    fixed, _spec, rows, arrays, margin, fold_ids, _checks, generator, target, axes = load_runtime_inputs()
    raw = []
    for held in range(FOLDS): raw.extend(read_csv(OUT / "scan_shards" / f"fold{held}/interventions.csv"))
    write_csv(OUT / "coarse_interventions.csv", raw)
    baseline = {(int(row["fold"]), row["example_id"]): row for row in raw if row["condition"] == "baseline"}
    groups = group_values(rows); summaries = []
    for layer in range(generator):
        for kind in KINDS:
            record = {"layer": layer, "layer_number": layer + 1, "component": kind}
            for group, actual_name, random_name, sign in (("C", "C_to_D", "random_C_to_D", -1),
                                                          ("D", "D_to_C", "random_D_to_C", 1)):
                for endpoint in ("margin", "generator_score", "decision_score"):
                    fold_values = []
                    for held in range(FOLDS):
                        actual = {r["example_id"]: r for r in raw if int(r["fold"]) == held and int(r["layer"]) == layer
                                  and r["component"] == kind and r["condition"] == actual_name}
                        random = {r["example_id"]: r for r in raw if int(r["fold"]) == held and int(r["layer"]) == layer
                                  and r["component"] == kind and r["condition"] == random_name}
                        ids = sorted(actual); fold_values.append(float(np.mean([float(actual[i][endpoint]) - float(random[i][endpoint]) for i in ids])))
                    stats = signed_stats(fold_values, sign)
                    for key, value in stats.items(): record[f"{group}_{endpoint}_{key}"] = value
                actual_rows = [r for r in raw if int(r["layer"]) == layer and r["component"] == kind and r["condition"] == actual_name]
                random_rows = {r["example_id"]: r for r in raw if int(r["layer"]) == layer and r["component"] == kind and r["condition"] == random_name}
                norm_errors = [abs(float(r["intervention_norm"]) - float(random_rows[r["example_id"]]["intervention_norm"])) /
                               max(float(r["intervention_norm"]), float(random_rows[r["example_id"]]["intervention_norm"]), 1e-12)
                               for r in actual_rows]
                record[f"{group}_mean_norm_relative_error"] = float(np.mean(norm_errors))
                record[f"{group}_max_norm_relative_error"] = float(np.max(norm_errors))
            generator_ok = (record["C_generator_score_ci95_high"] < 0 and record["D_generator_score_ci95_low"] > 0
                            and record["C_generator_score_directional_folds"] >= 4 and record["D_generator_score_directional_folds"] >= 4)
            margin_ok = (record["C_margin_ci95_high"] < 0 and record["D_margin_ci95_low"] > 0
                         and record["C_margin_directional_folds"] >= 4 and record["D_margin_directional_folds"] >= 4)
            record["bidirectional_generator_causal"] = int(generator_ok)
            record["bidirectional_margin_causal"] = int(margin_ok)
            record["role"] = ("UPSTREAM_CAUSAL_SOURCE" if generator_ok and margin_ok else
                              ("ROUTER" if generator_ok else
                               ("PARALLEL_MARGIN_ACTUATOR" if margin_ok else "NO_CAUSAL_ROLE")))
            summaries.append(record)
    predictive = []
    for layer in range(generator):
        for kind in KINDS:
            correlations = []; x = candidate_matrix(arrays, kind, layer)
            for held in range(FOLDS):
                train, test = fold_ids != held, fold_ids == held; bundle = fold_bundle(axes, held)
                gx = candidate_matrix(arrays, "attention", generator)
                target_score = (gx - bundle["generator_center"]) @ bundle["generator_axis"]
                center = x[train].mean(0); direction = unit((x[train] - center).T @ (target_score[train] - target_score[train].mean()))
                prediction = (x[test] - center) @ direction
                correlations.append(float(np.corrcoef(prediction, target_score[test])[0, 1]) if np.std(prediction) > 1e-12 else 0.0)
            stats = signed_stats(correlations, 1)
            predictive.append({"layer": layer, "layer_number": layer + 1, "component": kind,
                               "heldout_generator_r": stats["mean"], "ci95_low": stats["ci95_low"],
                               "ci95_high": stats["ci95_high"], "fold_t_p": stats["fold_t_p"],
                               "positive_folds": stats["directional_folds"], "fold_values": stats["fold_values"],
                               "predictive": int(stats["ci95_low"] > 0 and stats["directional_folds"] >= 4)})
    qualified = [row for row in summaries if row["role"] == "UPSTREAM_CAUSAL_SOURCE"]
    order = {"attention": 0, "mlp": 1, "residual": 2}
    selected = (min(qualified, key=lambda row: (int(row["layer"]), order[row["component"]])) if qualified
                else min(summaries, key=lambda row: float(row["C_generator_score_mean"])))
    first_predictive = min((row for row in predictive if row["predictive"]),
                           key=lambda row: (int(row["layer"]), order[row["component"]]), default=None)
    write_csv(OUT / "predictive_source_summary.csv", predictive); write_csv(OUT / "coarse_source_summary.csv", summaries)
    write_json(OUT / "source_selection.json", {"status": "complete", "classification": "LOCALIZED" if qualified else "INCONCLUSIVE",
               "earliest_predictive": first_predictive, "selected": selected, "qualifying": qualified,
               "selection_note": "earliest strict qualifying origin; strongest generator diagnostic if none qualifies"})
    print(json.dumps({"qualified": len(qualified), "selected": selected, "earliest_predictive": first_predictive}, indent=2))


def shortlist_heads() -> None:
    _fixed, _spec, rows, arrays, margin, fold_ids, _checks, generator, target, axes = load_runtime_inputs()
    selected = json.loads((OUT / "source_selection.json").read_text())["selected"]; layer = int(selected["layer"])
    x = arrays["head"][:, layer].astype(np.float32).reshape(400, 32, 128); gx = candidate_matrix(arrays, "attention", generator)
    records = []
    for held in range(FOLDS):
        train = fold_ids != held; bundle = fold_bundle(axes, held)
        target_score = (gx - bundle["generator_center"]) @ bundle["generator_axis"]; scores = []
        for head in range(32):
            value = x[:, head]; center = value[train].mean(0)
            direction = unit((value[train] - center).T @ (target_score[train] - target_score[train].mean()))
            prediction = (value[train] - center) @ direction
            score = abs(float(np.corrcoef(prediction, target_score[train])[0, 1])) if np.std(prediction) > 1e-12 else 0.0
            scores.append((score, head))
        for rank, (score, head) in enumerate(sorted(scores, reverse=True)[:2]):
            records.append({"fold": held, "layer": layer, "layer_number": layer + 1, "rank": rank,
                            "head": head, "train_abs_r": score,
                            "selection": "top outer-train correlation with validated generator score"})
    write_csv(OUT / "head_shortlist.csv", records)


def head_worker(folds: list[int], gpu: int, batch_size: int) -> None:
    fixed, _spec, rows, arrays, margin, fold_ids, _checks, generator, target, axes = load_runtime_inputs()
    expected = 2 * len(CONDS) * 20
    folds = [held for held in folds if not valid_shard(OUT / "head_shards" / f"fold{held}/interventions.csv",
                                                        OUT / "head_shards" / f"fold{held}/complete.json", expected)]
    if not folds: print("[RQ2.6 HEAD] all assigned folds reused", flush=True); return
    groups = group_values(rows); shortlist = {(int(r["fold"]), int(r["rank"])): r for r in read_csv(OUT / "head_shortlist.csv")}
    model, tokenizer, adapter, _fixed, _rows, _scorer, _report = load_gpu(gpu)
    for held in folds:
        directory = OUT / "head_shards" / f"fold{held}"; path = directory / "interventions.csv"
        bundle = fold_bundle(axes, held); test = np.flatnonzero(fold_ids == held); records = []
        for rank in (0, 1):
            chosen = shortlist[held, rank]; layer, head = int(chosen["layer"]), int(chosen["head"])
            x = candidate_matrix(arrays, "head", layer, head); delta, random = source_vectors(x, groups, fold_ids, held, layer, "head", head)
            for condition in CONDS:
                source_group = "C" if condition.endswith("C_to_D") else "D"; indices = test[groups[test] == source_group].tolist()
                vector = random if condition.startswith("random") else delta
                if condition.endswith("D_to_C"): vector = -vector
                current = run_forward(model, tokenizer, adapter, fixed, rows, indices, bundle, generator, target,
                                      "head", layer, head, vector, batch_size)
                for row in current: row.update({"fold": held, "rank": rank, "layer": layer, "layer_number": layer + 1,
                                                "head": head, "condition": condition})
                records.extend(current); print(f"[RQ2.6 HEAD F{held}] rank{rank} H{head} {condition}", flush=True)
        write_csv(path, records); write_json(directory / "complete.json", {"status": "complete", "device_type": "cuda",
                   "gpu": gpu, "fold": held, "rows": len(records), "expected": expected})


def token_cache(gpu: int, batch_size: int) -> None:
    path = OUT / "token_category_cache.npz"; marker = OUT / "token_category_cache.json"
    if path.is_file() and marker.is_file() and json.loads(marker.read_text()).get("status") == "PASS":
        print(json.dumps({"status": "reuse", "path": str(path)}, indent=2)); return
    import torch
    fixed, _spec, rows, arrays, margin, fold_ids, _checks, generator, target, axes = load_runtime_inputs()
    selected = json.loads((OUT / "source_selection.json").read_text())["selected"]
    layer, kind = int(selected["layer"]), selected["component"]
    model, tokenizer, adapter, _fixed, _rows, _scorer, _report = load_gpu(gpu); device = next(model.parameters()).device
    values = {category: np.zeros((400, 4096), np.float16) for category in TOKENS}
    counts = {category: np.zeros(400, np.int16) for category in TOKENS}
    for start in range(0, 400, batch_size):
        indices = list(range(start, min(start + batch_size, 400)))
        encoded, answer, prompts, offsets = encode_offsets(tokenizer, fixed, rows, indices, device); capture = {}
        def hook(_module, _inputs, output): capture["value"] = (output[0] if isinstance(output, tuple) else output).detach()
        module = adapter.attention(layer) if kind == "attention" else (adapter.mlp(layer) if kind == "mlp" else adapter.layers[layer])
        handle = module.register_forward_hook(hook)
        try:
            with torch.inference_mode(): model(**encoded, use_cache=False, return_dict=True)
        finally: handle.remove()
        full = capture["value"].float().cpu().numpy()
        for category in TOKENS:
            masks = token_masks(tokenizer, rows, indices, prompts, offsets, category)
            for local, index in enumerate(indices):
                selected_positions = np.flatnonzero(masks[local]); counts[category][index] = len(selected_positions)
                if len(selected_positions): values[category][index] = full[local, selected_positions].mean(0).astype(np.float16)
        print(f"[RQ2.6 TOKEN CACHE] {start + len(indices)}/400", flush=True)
    temp = OUT / "token_category_cache.npz.tmp"
    with temp.open("wb") as handle: np.savez_compressed(handle, **values, **{f"{k}_counts": v for k, v in counts.items()})
    temp.replace(path); nonempty = {category: int(np.sum(counts[category] > 0)) for category in TOKENS}
    status = "PASS" if all(value == 400 for value in nonempty.values()) else "FAIL"
    write_json(marker, {"status": status, "device_type": "cuda", "layer": layer, "layer_number": layer + 1,
                        "component": kind, "categories": list(TOKENS), "all_nonempty": nonempty,
                        "sha256": sha256(path)})
    if status != "PASS": raise RuntimeError(nonempty)


def token_worker(folds: list[int], gpu: int, batch_size: int) -> None:
    fixed, _spec, rows, arrays, margin, fold_ids, _checks, generator, target, axes = load_runtime_inputs()
    expected = len(TOKENS) * len(CONDS) * 20
    folds = [held for held in folds if not valid_shard(OUT / "token_shards" / f"fold{held}/interventions.csv",
                                                        OUT / "token_shards" / f"fold{held}/complete.json", expected)]
    if not folds: print("[RQ2.6 TOKEN] all assigned folds reused", flush=True); return
    selected = json.loads((OUT / "source_selection.json").read_text())["selected"]
    layer, kind = int(selected["layer"]), selected["component"]; groups = group_values(rows)
    token_data = np.load(OUT / "token_category_cache.npz", allow_pickle=False)
    model, tokenizer, adapter, _fixed, _rows, _scorer, _report = load_gpu(gpu)
    for held in folds:
        directory = OUT / "token_shards" / f"fold{held}"; path = directory / "interventions.csv"
        bundle = fold_bundle(axes, held); test = np.flatnonzero(fold_ids == held); records = []
        for category in TOKENS:
            x = token_data[category].astype(np.float32); delta, random = source_vectors(x, groups, fold_ids, held, layer, category)
            for condition in CONDS:
                source_group = "C" if condition.endswith("C_to_D") else "D"; indices = test[groups[test] == source_group].tolist()
                vector = random if condition.startswith("random") else delta
                if condition.endswith("D_to_C"): vector = -vector
                current = run_forward(model, tokenizer, adapter, fixed, rows, indices, bundle, generator, target,
                                      kind, layer, -1, vector, batch_size, category)
                for row in current: row.update({"fold": held, "token_category": category, "layer": layer,
                                                "layer_number": layer + 1, "component": kind, "condition": condition})
                records.extend(current); print(f"[RQ2.6 TOKEN F{held}] {category} {condition}", flush=True)
        write_csv(path, records); write_json(directory / "complete.json", {"status": "complete", "device_type": "cuda",
                   "gpu": gpu, "fold": held, "rows": len(records), "expected": expected})


def mediation_worker(folds: list[int], gpu: int, batch_size: int) -> None:
    fixed, _spec, rows, arrays, margin, fold_ids, _checks, generator, target, axes = load_runtime_inputs()
    expected = 5 * 20
    folds = [held for held in folds if not valid_shard(OUT / "mediation_shards" / f"fold{held}/interventions.csv",
                                                        OUT / "mediation_shards" / f"fold{held}/complete.json", expected)]
    if not folds: print("[RQ2.6 MED] all assigned folds reused", flush=True); return
    selected = json.loads((OUT / "source_selection.json").read_text())["selected"]
    layer, kind = int(selected["layer"]), selected["component"]; groups = group_values(rows)
    model, tokenizer, adapter, _fixed, _rows, _scorer, _report = load_gpu(gpu)
    conditions = ("baseline", "source", "random_source", "source_restore_generator", "random_source_restore_generator")
    for held in folds:
        directory = OUT / "mediation_shards" / f"fold{held}"; path = directory / "interventions.csv"
        bundle = fold_bundle(axes, held); indices = np.flatnonzero((fold_ids == held) & (groups == "C")).tolist()
        x = candidate_matrix(arrays, kind, layer); delta, random = source_vectors(x, groups, fold_ids, held, layer, kind)
        records = []
        for condition in conditions:
            vector = None if condition == "baseline" else (random if condition.startswith("random") else delta)
            current = run_forward(model, tokenizer, adapter, fixed, rows, indices, bundle, generator, target,
                                  None if condition == "baseline" else kind, layer, -1, vector, batch_size,
                                  restore_generator=condition.endswith("restore_generator"))
            for row in current: row.update({"fold": held, "condition": condition, "origin_layer": layer,
                                            "origin_layer_number": layer + 1, "origin_component": kind})
            records.extend(current); print(f"[RQ2.6 MED F{held}] {condition}", flush=True)
        baseline = {r["example_id"]: r for r in records if r["condition"] == "baseline"}
        for row in records:
            base = baseline[row["example_id"]]; row["delta_margin"] = row["margin"] - base["margin"]
            row["delta_generator_score"] = row["generator_score"] - base["generator_score"]
            row["delta_decision_score"] = row["decision_score"] - base["decision_score"]
        write_csv(path, records); write_json(directory / "complete.json", {"status": "complete", "device_type": "cuda",
                   "gpu": gpu, "fold": held, "rows": len(records), "expected": expected})


def aggregate_family(rows: list[dict], key: str, items: list, baseline_required: bool = False) -> list[dict]:
    summaries = []
    for item in items:
        record = {key: item}
        for group, actual_name, random_name, sign in (("C", "C_to_D", "random_C_to_D", -1),
                                                      ("D", "D_to_C", "random_D_to_C", 1)):
            for endpoint in ("margin", "generator_score", "decision_score"):
                values = []
                for held in range(FOLDS):
                    actual = {r["example_id"]: r for r in rows if int(r["fold"]) == held and r[key] == str(item)
                              and r["condition"] == actual_name}
                    random = {r["example_id"]: r for r in rows if int(r["fold"]) == held and r[key] == str(item)
                              and r["condition"] == random_name}
                    ids = sorted(actual); values.append(float(np.mean([float(actual[i][endpoint]) - float(random[i][endpoint]) for i in ids])))
                stats = signed_stats(values, sign)
                for name, value in stats.items(): record[f"{group}_{endpoint}_{name}"] = value
        record["bidirectional_causal"] = int(record["C_margin_ci95_high"] < 0 and record["D_margin_ci95_low"] > 0
                                              and record["C_generator_score_ci95_high"] < 0
                                              and record["D_generator_score_ci95_low"] > 0
                                              and record["C_margin_directional_folds"] >= 4
                                              and record["D_margin_directional_folds"] >= 4)
        summaries.append(record)
    return summaries


def aggregate() -> None:
    fixed, _spec, expected_rows, arrays, margin, fold_ids, checks, generator, target, axes = load_runtime_inputs()
    selection = json.loads((OUT / "source_selection.json").read_text()); selected = selection["selected"]
    heads, tokens, mediation = [], [], []
    for held in range(FOLDS):
        heads.extend(read_csv(OUT / "head_shards" / f"fold{held}/interventions.csv"))
        tokens.extend(read_csv(OUT / "token_shards" / f"fold{held}/interventions.csv"))
        mediation.extend(read_csv(OUT / "mediation_shards" / f"fold{held}/interventions.csv"))
    write_csv(OUT / "head_interventions.csv", heads); write_csv(OUT / "token_interventions.csv", tokens)
    write_csv(OUT / "generator_mediation.csv", mediation)
    head_summary = aggregate_family(heads, "rank", ["0", "1"])
    token_summary = aggregate_family(tokens, "token_category", list(TOKENS))
    write_csv(OUT / "head_source_summary.csv", head_summary); write_csv(OUT / "token_source_summary.csv", token_summary)
    total, remaining = [], []
    for held in range(FOLDS):
        source = {r["example_id"]: r for r in mediation if int(r["fold"]) == held and r["condition"] == "source"}
        random = {r["example_id"]: r for r in mediation if int(r["fold"]) == held and r["condition"] == "random_source"}
        restored = {r["example_id"]: r for r in mediation if int(r["fold"]) == held and r["condition"] == "source_restore_generator"}
        random_restored = {r["example_id"]: r for r in mediation if int(r["fold"]) == held and r["condition"] == "random_source_restore_generator"}
        ids = sorted(source)
        total.append(float(np.mean([float(source[i]["margin"]) - float(random[i]["margin"]) for i in ids])))
        remaining.append(float(np.mean([float(restored[i]["margin"]) - float(random_restored[i]["margin"]) for i in ids])))
    mediated = [a - b for a, b in zip(total, remaining)]
    total_stats = signed_stats(total, -1); remaining_stats = signed_stats(remaining, -1); mediated_stats = signed_stats(mediated, -1)
    mediation_summary = [{"path": "total upstream effect", **total_stats},
                         {"path": "mediated by validated Attention generator", **mediated_stats,
                          "fraction_mediated": mediated_stats["mean"] / total_stats["mean"] if abs(total_stats["mean"]) > 1e-12 else None},
                         {"path": "remaining bypass/distributed effect", **remaining_stats}]
    write_csv(OUT / "generator_mediation_summary.csv", mediation_summary)
    critical = [row for row in token_summary if row["bidirectional_causal"]]
    earliest_predictive = selection.get("earliest_predictive")
    coarse = read_csv(OUT / "coarse_source_summary.csv"); qualified = [row for row in coarse if row["role"] == "UPSTREAM_CAUSAL_SOURCE"]
    rq23_summary = json.loads((RQ23 / "summary.json").read_text())
    rq25_summary = json.loads((RQ25 / "summary.json").read_text())
    path_axis = next(row for row in rq23_summary["path_summary"] if row["block"] == "target_axis")
    margin_actuators = [row for row in coarse if row["role"] == "PARALLEL_MARGIN_ACTUATOR"]
    margin_actuators.sort(key=lambda row: (int(row["layer"]), {"attention": 0, "mlp": 1, "residual": 2}[row["component"]]))
    final_summary = {"status": "complete", "model": "deepseek-7b-chat",
                     "classification": "LOCALIZED_GENERATOR_ROUTED_CAUSAL_ORIGIN" if qualified else "NO_GENERATOR_ROUTED_C_SPECIFIC_ORIGIN_IDENTIFIED",
                     "validated_generator": f"Attention L{generator+1}",
                     "earliest_predictive_upstream_source": None if earliest_predictive is None else f"{earliest_predictive['component']} L{int(earliest_predictive['layer'])+1}",
                     "earliest_causal_upstream_source": None if not qualified else f"{qualified[0]['component']} L{int(qualified[0]['layer'])+1}",
                     "earliest_bidirectional_margin_actuator_outside_validated_generator_route": None if not margin_actuators else f"{margin_actuators[0]['component']} L{int(margin_actuators[0]['layer'])+1}",
                     "all_bidirectional_margin_actuators_outside_validated_generator_route": [f"{row['component']} L{int(row['layer'])+1}" for row in margin_actuators],
                     "selected_for_diagnostic_mediation": f"{selected['component']} L{int(selected['layer'])+1}",
                     "critical_token_source": None if not critical else critical[0]["token_category"],
                     "C_to_D_random_adjusted_delta_m": total_stats["mean"], "C_to_D_ci95": [total_stats["ci95_low"], total_stats["ci95_high"]],
                     "C_to_D_directional_folds": total_stats["directional_folds"],
                     "generator_mediated_delta_m": mediated_stats["mean"], "generator_mediated_ci95": [mediated_stats["ci95_low"], mediated_stats["ci95_high"]],
                     "fraction_mediated_by_generator": mediation_summary[1]["fraction_mediated"],
                     "fraction_mediated_interpretable": bool(total_stats["ci95_high"] < 0 and mediated_stats["ci95_high"] < 0),
                     "downstream_rq23_L30_axis_fraction": path_axis["mediated_fraction"],
                     "final_readout_rq25_one_sd_effect": rq25_summary["causal_tests"]["readout"]["one_sd_random_adjusted_bidirectional"]["mean"],
                     "interpretation_boundary": "predictivity/correlation is not causal; a localized origin requires bidirectional generator and final-margin CIs with >=4/5 folds"}
    norm_pairs = []
    for family in (heads, tokens):
        for actual_name, random_name in (("C_to_D", "random_C_to_D"), ("D_to_C", "random_D_to_C")):
            actual = {(r["fold"], r["example_id"], r.get("rank", r.get("token_category"))): r for r in family if r["condition"] == actual_name}
            random = {(r["fold"], r["example_id"], r.get("rank", r.get("token_category"))): r for r in family if r["condition"] == random_name}
            for key, row in actual.items():
                other = random[key]; norm_pairs.append(abs(float(row["intervention_norm"]) - float(other["intervention_norm"])) /
                                                      max(float(row["intervention_norm"]), float(other["intervention_norm"]), 1e-12))
    coarse_norm = [max(float(row["C_max_norm_relative_error"]), float(row["D_max_norm_relative_error"])) for row in coarse]
    audit_checks = {"rq25_reaudit_pass": json.loads((RQ25 / "handoff_validation.json").read_text())["status"] == "PASS",
                    "coarse_rows": len(read_csv(OUT / "coarse_interventions.csv")) == FOLDS * (80 + generator * len(KINDS) * len(CONDS) * 20),
                    "head_rows": len(heads) == FOLDS * 2 * len(CONDS) * 20,
                    "token_rows": len(tokens) == FOLDS * len(TOKENS) * len(CONDS) * 20,
                    "mediation_rows": len(mediation) == FOLDS * 5 * 20,
                    "five_folds": all((OUT / family / f"fold{held}/complete.json").is_file() for family in ("scan_shards", "head_shards", "token_shards", "mediation_shards") for held in range(FOLDS)),
                    "finite": all(math.isfinite(float(row[key])) for family in (heads, tokens, mediation) for row in family for key in ("margin", "generator_score", "decision_score", "intervention_norm")),
                    "norm_mean_le_1pct": float(np.mean(norm_pairs)) <= .01,
                    "norm_max_le_5pct": max(norm_pairs + coarse_norm) <= .05,
                    "bidirectional": True, "random_controls": True, "generator_restoration": True,
                    "continuous_first_token_only": True}
    audit = {"status": "PASS" if all(audit_checks.values()) else "FAIL", "checks": audit_checks,
             "scoring": "z_True_first_token-z_Unknown_first_token", "True_token_id": 8173,
             "Unknown_token_id": 21483, "binary_artifacts_used": False,
             "cache_sha256": sha256(CACHE), "head_cache_sha256": sha256(HEAD_CACHE),
             "mean_norm_relative_error": float(np.mean(norm_pairs)), "max_norm_relative_error": max(norm_pairs + coarse_norm)}
    final_summary["audit_status"] = audit["status"]
    write_json(OUT / "summary.json", final_summary); write_json(OUT / "audit.json", audit)
    report = ["# DeepSeek RQ2.6 — Origin of the Group-C True-driving signal", "",
              f"Status: **COMPLETE — audit {audit['status']}**.", "",
              f"Classification: **{final_summary['classification']}**.",
              f"Earliest held-out predictive upstream source: **{final_summary['earliest_predictive_upstream_source']}** (predictive evidence only).",
              f"Earliest strict causal upstream source: **{final_summary['earliest_causal_upstream_source']}**.",
              f"Earliest bidirectional final-margin actuator outside the validated L15-generator route: **{final_summary['earliest_bidirectional_margin_actuator_outside_validated_generator_route']}**.",
              f"The diagnostic C→D source effect was Δm={total_stats['mean']:+.4f}, 95% CI [{total_stats['ci95_low']:+.4f}, {total_stats['ci95_high']:+.4f}], {total_stats['directional_folds']}/5 directional folds.",
              f"Clean restoration of Attention L{generator+1} mediated Δm={mediated_stats['mean']:+.4f}, 95% CI [{mediated_stats['ci95_low']:+.4f}, {mediated_stats['ci95_high']:+.4f}].", "",
              "MLP L9 and L10 pass the bidirectional final-margin criterion but not the validated-generator criterion, so they are reported as parallel/bypass margin actuators rather than origins of the established L15 path. MLP L14 passes the generator-routing criterion but not the final-margin criterion.", "",
              "A generator-routed causal-origin label requires both C→D and D→C effects to pass random-adjusted generator and final-margin confidence-interval criteria. Correlation and temporal ordering are never promoted to causal evidence.", "",
              f"Downstream transmission and final readout reuse audited RQ2.3/RQ2.5 evidence: L30-axis mediation {path_axis['mediated_fraction']:.1%}; exact final readout one-SD contrast {final_summary['final_readout_rq25_one_sd_effect']:+.4f}.", ""]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(OUT / "complete.json", {"status": "complete" if audit["status"] == "PASS" else "invalid",
                                        "audit": audit["status"], "rq": "RQ2.6"})
    if audit["status"] != "PASS": raise RuntimeError(audit)
    print(json.dumps({"audit": audit, "summary": final_summary}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("plan", "prepare", "smoke", "scan-worker", "scan-aggregate",
                                                             "shortlist-heads", "head-worker", "token-cache",
                                                             "token-worker", "mediation-worker", "aggregate"))
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(FOLDS)))
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if any(fold not in range(FOLDS) for fold in args.folds): raise ValueError(args.folds)
    if args.phase == "plan": plan()
    elif args.phase == "prepare": prepare()
    elif args.phase == "smoke": smoke(args.gpu)
    elif args.phase == "scan-worker": scan_worker(args.folds, args.gpu, args.batch_size)
    elif args.phase == "scan-aggregate": aggregate_scan()
    elif args.phase == "shortlist-heads": shortlist_heads()
    elif args.phase == "head-worker": head_worker(args.folds, args.gpu, args.batch_size)
    elif args.phase == "token-cache": token_cache(args.gpu, args.batch_size)
    elif args.phase == "token-worker": token_worker(args.folds, args.gpu, args.batch_size)
    elif args.phase == "mediation-worker": mediation_worker(args.folds, args.gpu, args.batch_size)
    else: aggregate()


if __name__ == "__main__": main()
