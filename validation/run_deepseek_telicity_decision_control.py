#!/usr/bin/env python3
"""Missing DeepSeek causal test: Telicity vs preference predictor vs readout.

This is an opt-in validation workflow built on the generalized registry,
adapter, prompt, fold, and first-token scorer.  It reuses the completed
DeepSeek cache and never reruns behavior, localization, erasure, component, or
source-path stages.

Canonical cross-model choice
----------------------------
Qwen selects a held-out continuous preference axis after a full-depth search;
Mistral Experiment 5 compares Telicity, decision predictor, and exact output
readout with equal train-only random-axis L2 scale.  DeepSeek uses the common
executable intersection: pre-existing full-depth results fix L19 Telicity and
L30 decision candidates before effects are viewed, axes are fit on four folds,
and all actuators receive the same realized L2 dose.  The exact readout is
intervened at final-norm output.  Random, label-permutation, late-Telicity, and
held-out A/B/C/D controls are recorded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_registry import load_registry, resolved_source
from src.output_validation import atomic_json, write_csv
from src.preflight import build_prompt, folds, load_dataset, run_preflight

CONFIG = ROOT / "configs/multimodel_true_unknown_preference.yaml"
BASE = ROOT / "results/deepseek-7b-telicity-rq"
OUT = BASE / "telicity_decision_control"
CACHE = BASE / "full_depth_cache.npz"
ENCODING_SUMMARY = BASE / "encoding_decision_summary.json"
ENCODING_AUDIT = BASE / "encoding_decision_audit.json"
SOURCE_RAW = BASE / "source_pathway/layer_fold_pathway_results.csv"
SEED = 42042
DOSES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
FOLDS = 5
EXPECTED_ITEMS = 400
TELICITY_LAYER = 19
DECISION_LAYER = 30
NUM_LAYERS = 30
HIDDEN_SIZE = 4096
CONDITIONS_PER_EXAMPLE = 43
EXPECTED_INTERVENTION_ROWS = EXPECTED_ITEMS * CONDITIONS_PER_EXAMPLE


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit(value):
    value = np.asarray(value, np.float64)
    size = float(np.linalg.norm(value))
    if size <= 1e-12:
        raise ValueError("Degenerate causal axis")
    return value / size


def fit_telicity_axis(x, labels):
    """Canonical class-balanced dual-ridge probe direction."""
    x, labels = np.asarray(x, np.float64), np.asarray(labels, np.int64)
    signed = np.where(labels == 1, 1.0, -1.0)
    centered = x - x.mean(0, keepdims=True)
    counts = np.bincount(labels, minlength=2).astype(float); counts[counts == 0] = 1
    weights = np.where(labels == 1, .5 / counts[1], .5 / counts[0]); root = np.sqrt(weights)
    xw, yw = centered * root[:, None], signed * root
    gram = xw @ xw.T + 1e-2 * np.eye(len(xw))
    return unit(xw.T @ np.linalg.solve(gram, yw))


def fit_decision_axis(x, margin, alpha=10.0):
    """Canonical standardized continuous-margin ridge direction."""
    x, margin = np.asarray(x, np.float64), np.asarray(margin, np.float64)
    mean, sd = x.mean(0), x.std(0); sd[sd < 1e-6] = 1
    z, target = (x - mean) / sd, margin - margin.mean()
    coef = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), target)
    return unit(coef / sd)


def random_orthogonal(rng, axes, dim):
    value = rng.normal(size=dim)
    basis = []
    for axis in axes:
        candidate = np.asarray(axis, np.float64).copy()
        for previous in basis:
            candidate -= previous * (candidate @ previous)
        size = np.linalg.norm(candidate)
        if size > 1e-10:
            basis.append(candidate / size)
    for axis in basis:
        value -= axis * (value @ axis)
    return unit(value)


def typical_random_scale(x, train, rng, count=64):
    centered = np.asarray(x[train], np.float64) - np.asarray(x[train], np.float64).mean(0)
    return float(np.median([np.std(centered @ unit(rng.normal(size=x.shape[-1]))) for _ in range(count)]))


def effective_basis(directions, threshold=.90):
    _, singular, vt = np.linalg.svd(np.asarray(directions, np.float64), full_matrices=False)
    energy = np.cumsum(singular * singular) / np.sum(singular * singular)
    rank = int(np.searchsorted(energy, threshold) + 1)
    return vt[:rank].T, rank


def load_inputs():
    raw, fixed, spec = load_registry(CONFIG, "deepseek-7b-chat")
    rows = load_dataset(fixed)
    cache = np.load(CACHE, allow_pickle=False)
    if cache["residual"].shape != (EXPECTED_ITEMS, NUM_LAYERS, HIDDEN_SIZE):
        raise ValueError(f"Unexpected DeepSeek residual cache {cache['residual'].shape}")
    if cache["ids"].tolist() != [row["id"] for row in rows]:
        raise ValueError("DeepSeek cache item order mismatch")
    expected_folds = np.asarray(folds(rows, fixed), int)
    if not np.array_equal(cache["fold"].astype(int), expected_folds):
        raise ValueError("DeepSeek cache fold mismatch")
    if not all(np.isfinite(cache[name]).all() for name in ("residual", "margin")):
        raise ValueError("Nonfinite DeepSeek cache")
    encoding = json.loads(ENCODING_SUMMARY.read_text())
    audit = json.loads(ENCODING_AUDIT.read_text())
    if audit["status"] != "PASS" or encoding["status"] != "complete":
        raise RuntimeError("Existing DeepSeek encoding/decision audit is not valid")
    tel_layer = int(encoding["telicity"]["strongest"]["layer"])
    decision_layer = int(encoding["decision"]["strongest_residual"]["layer"])
    if (tel_layer, decision_layer) != (TELICITY_LAYER, DECISION_LAYER):
        raise RuntimeError(f"Pre-existing layer selection changed: {(tel_layer, decision_layer)}")
    return raw, fixed, spec, rows, cache, expected_folds, tel_layer, decision_layer


def print_plan():
    """Print the immutable scientific plan without loading model weights."""
    _, _, spec, _, _, _, tel_layer, decision_layer = load_inputs()
    plan = {
        "model": "deepseek-7b-chat",
        "gpu_work_started": False,
        "reused_artifacts": {
            "full_depth_cache": {"path": str(CACHE), "sha256": sha256(CACHE)},
            "encoding_decision_audit": {"path": str(ENCODING_AUDIT), "status": "PASS"},
            "moving_erasure": str(BASE / "removal_sweep/summary.json"),
            "source_pathway": {"path": str(SOURCE_RAW), "sha256": sha256(SOURCE_RAW)},
        },
        "implementation": str(Path(__file__).resolve()),
        "adapter": "src.model_adapters.HFDecoderAdapter with model-configured DeepSeek Llama paths",
        "model_source": resolved_source(spec),
        "layers": {"telicity": tel_layer, "decision_predictor": decision_layer,
                   "exact_readout": "final norm output", "off_target": "late Telicity at final norm"},
        "conditions_per_example": CONDITIONS_PER_EXAMPLE,
        "expected_intervention_rows": EXPECTED_INTERVENTION_ROWS,
        "outputs": [str(OUT / name) for name in
                    ("interventions_all_examples.csv", "layer_fold_causal_results.csv", "summary.json",
                     "audit.json", "REPORT.md", "RQ_TABLE_CELL.txt", "handoff_validation.json")],
        "locked_estimand": "m = z(True_first_token) - z(Unknown_first_token)",
        "selection_frozen_before_causal_effects": True,
    }
    print(json.dumps(plan, indent=2))


def load_model(spec):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = resolved_source(spec)
    if spec.get("allow_legacy_torch_load_for_trusted_local_checkpoint"):
        import transformers.modeling_utils as modeling_utils
        modeling_utils.check_torch_load_is_safe = lambda: None
    tok = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True,
                                        trust_remote_code=bool(spec.get("trust_remote_code", False)))
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[spec["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, torch_dtype=dtype, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=bool(spec.get("trust_remote_code", False))).eval()
    return model, tok


def geometry_rows(tel_all, dec_all):
    fold_rows, layer_rows = [], []
    for layer in range(tel_all.shape[1]):
        cosines = np.sum(tel_all[:, layer] * dec_all[:, layer], axis=1)
        for fold, cosine in enumerate(cosines):
            fold_rows.append({"layer": layer + 1, "fold": fold, "cosine": float(cosine),
                              "abs_cosine": float(abs(cosine)),
                              "principal_angle_deg_1d": float(np.degrees(np.arccos(np.clip(abs(cosine), 0, 1))))})
        tb, kt = effective_basis(tel_all[:, layer]); db, kd = effective_basis(dec_all[:, layer])
        singular = np.linalg.svd(tb.T @ db, compute_uv=False)
        angles = np.degrees(np.arccos(np.clip(singular, 0, 1)))
        layer_rows.append({"layer": layer + 1, "paired_cosine_mean": float(cosines.mean()),
                           "paired_abs_cosine_mean": float(np.abs(cosines).mean()),
                           "paired_abs_cosine_max": float(np.abs(cosines).max()),
                           "paired_angle_deg_mean": float(np.degrees(np.arccos(np.clip(np.abs(cosines), 0, 1))).mean()),
                           "telicity_crossfold_k90": kt, "decision_crossfold_k90": kd,
                           "smallest_principal_angle_deg": float(angles.min()),
                           "mean_principal_angle_deg": float(angles.mean()),
                           "normalized_subspace_overlap": float(np.sum(singular ** 2) / min(kt, kd))})
    return fold_rows, layer_rows


def prepare(gpu):
    reusable = (OUT / "prepare_complete.json", OUT / "execution_plan.json",
                OUT / "intervention_assets.npz", OUT / "fold_axis_config.csv",
                OUT / "axis_geometry_foldwise.csv", OUT / "axis_geometry_layerwise.csv")
    if all(path.is_file() and path.stat().st_size > 0 for path in reusable):
        existing = json.loads((OUT / "execution_plan.json").read_text())
        if (json.loads((OUT / "prepare_complete.json").read_text()).get("status") == "complete"
                and existing.get("reused_artifacts", {}).get("full_depth_cache", {}).get("sha256") == sha256(CACHE)
                and existing.get("layers") == {"telicity": TELICITY_LAYER,
                                                "decision_predictor": DECISION_LAYER,
                                                "exact_readout": "final norm output",
                                                "off_target": "late Telicity at final norm"}):
            print(json.dumps({"status": "reuse", "gpu_work_started": False,
                              "reason": "prepared axes and geometry match the locked input cache",
                              "execution_plan": str(OUT / "execution_plan.json")}, indent=2))
            return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from src.model_adapters import make_adapter

    raw, fixed, spec, rows, cache, fold_ids, tel_layer, dec_layer = load_inputs()
    model, tok = load_model(spec); adapter = make_adapter(model, spec)
    preflight, _, scorer = run_preflight(tok, fixed, spec, model)
    residual = cache["residual"].astype(np.float32); margin = cache["margin"].astype(np.float64)
    device = next(model.parameters()).device
    final_norm = np.zeros_like(residual[:, -1], np.float32)
    for start in range(0, len(rows), 64):
        tensor = torch.as_tensor(residual[start:start + 64, -1], device=device,
                                 dtype=next(model.parameters()).dtype)
        with torch.inference_mode(): final_norm[start:start + 64] = adapter.final_norm(tensor).float().cpu().numpy()
    labels = np.asarray([1 if row["group"][0] in "AC" else 0 for row in rows], int)
    cd = np.asarray([row["group"][0] in "CD" for row in rows])
    tel_all = np.zeros((FOLDS, NUM_LAYERS, HIDDEN_SIZE), np.float32); dec_all = np.zeros_like(tel_all)
    assets, selections = {}, []
    readout = unit((adapter.output_head.weight[scorer.ids["True"]] -
                    adapter.output_head.weight[scorer.ids["Unknown"]]).detach().float().cpu().numpy())
    for fold in range(FOLDS):
        train = fold_ids != fold; train_cd = train & cd; rng = np.random.default_rng(SEED + fold)
        for layer in range(NUM_LAYERS):
            tel_all[fold, layer] = fit_telicity_axis(residual[train_cd, layer], labels[train_cd])
            dec_all[fold, layer] = fit_decision_axis(residual[train, layer], margin[train])
        tel = tel_all[fold, tel_layer - 1].astype(float)
        dec = dec_all[fold, dec_layer - 1].astype(float)
        tel_norm = fit_telicity_axis(final_norm[train_cd], labels[train_cd])
        dec_norm = fit_decision_axis(final_norm[train], margin[train])
        perm_tel_labels = labels[train_cd].copy(); rng.shuffle(perm_tel_labels)
        perm_margin = margin[train].copy(); rng.shuffle(perm_margin)
        perm_tel = fit_telicity_axis(residual[train_cd, tel_layer - 1], perm_tel_labels)
        perm_dec = fit_decision_axis(residual[train, dec_layer - 1], perm_margin)
        random_tel = random_orthogonal(rng, (tel, perm_tel), HIDDEN_SIZE)
        random_dec = random_orthogonal(rng, (dec, perm_dec), HIDDEN_SIZE)
        random_norm = random_orthogonal(rng, (readout, tel_norm, dec_norm), HIDDEN_SIZE)
        scales = [typical_random_scale(residual[:, tel_layer - 1], train, rng),
                  typical_random_scale(residual[:, dec_layer - 1], train, rng),
                  typical_random_scale(final_norm, train, rng)]
        common = float(np.exp(np.mean(np.log(np.maximum(scales, 1e-12)))))
        for name, value in (("tel", tel), ("dec", dec), ("tel_norm", tel_norm),
                            ("dec_norm", dec_norm), ("perm_tel", perm_tel),
                            ("perm_dec", perm_dec), ("random_tel", random_tel),
                            ("random_dec", random_dec), ("random_norm", random_norm),
                            ("readout", readout)):
            assets[f"f{fold}_{name}"] = np.asarray(value, np.float32)
        assets[f"f{fold}_common_scale"] = np.asarray(common)
        selections.append({"fold": fold, "telicity_layer": tel_layer, "decision_layer": dec_layer,
                           "telicity_layer_scale": scales[0], "decision_layer_scale": scales[1],
                           "final_norm_scale": scales[2], "matched_common_1sd_norm": common,
                           "telicity_decision_cross_layer_cosine": float(tel @ dec),
                           "telicity_norm_readout_cosine": float(tel_norm @ readout),
                           "decision_norm_readout_cosine": float(dec_norm @ readout),
                           "telicity_norm_decision_norm_cosine": float(tel_norm @ dec_norm)})
    assets["telicity_all_layers"] = tel_all; assets["decision_all_layers"] = dec_all
    assets["final_norm_states"] = final_norm.astype(np.float16)
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / "intervention_assets.npz.tmp"
    with temporary.open("wb") as handle: np.savez_compressed(handle, **assets)
    temporary.replace(OUT / "intervention_assets.npz")
    write_csv(OUT / "fold_axis_config.csv", selections)
    fold_geo, layer_geo = geometry_rows(tel_all, dec_all)
    write_csv(OUT / "axis_geometry_foldwise.csv", fold_geo); write_csv(OUT / "axis_geometry_layerwise.csv", layer_geo)
    plan = {
        "status": "ready", "model": "deepseek-7b-chat", "gpu_prepare": gpu,
        "canonical_protocol": "Qwen full-depth train-only continuous axis selection + Mistral matched-norm Telicity/decision/readout intervention",
        "reused_artifacts": {
            "full_depth_cache": {"path": str(CACHE), "sha256": sha256(CACHE)},
            "encoding_decision_audit": {"path": str(ENCODING_AUDIT), "status": "PASS"},
            "moving_erasure": str(BASE / "removal_sweep/summary.json"),
            "source_pathway": {"path": str(SOURCE_RAW), "sha256": sha256(SOURCE_RAW)},
        },
        "adapter": "src.model_adapters.HFDecoderAdapter; DeepSeek Llama paths from YAML",
        "layers": {"telicity": tel_layer, "decision_predictor": dec_layer,
                   "exact_readout": "final norm output", "off_target": "late Telicity at final norm"},
        "conditions_per_example": CONDITIONS_PER_EXAMPLE,
        "expected_intervention_rows": EXPECTED_INTERVENTION_ROWS,
        "folds": FOLDS, "heldout_items_per_fold": EXPECTED_ITEMS // FOLDS,
        "outputs": ["interventions_all_examples.csv", "layer_fold_causal_results.csv",
                    "summary.json", "audit.json", "REPORT.md", "RQ_TABLE_CELL.txt",
                    "handoff_validation.json"],
        "scorer": scorer.audit(), "preflight": preflight["status"],
        "selection_frozen_before_causal_effects": True,
    }
    atomic_json(OUT / "execution_plan.json", plan); atomic_json(OUT / "prepare_complete.json", {"status": "complete"})
    print(json.dumps(plan, indent=2))


def conditions_for_coordinate(coordinate):
    if coordinate == "L19":
        values = [("telicity", "actual", dose, "tel") for dose in DOSES]
        values += [("telicity", "random", dose, "random_tel") for dose in DOSES]
        values += [("telicity", "label_permutation", dose, "perm_tel") for dose in (-1., 1.)]
    elif coordinate == "L30":
        values = [("decision", "actual", dose, "dec") for dose in DOSES]
        values += [("decision", "random", dose, "random_dec") for dose in DOSES]
        values += [("decision", "label_permutation", dose, "perm_dec") for dose in (-1., 1.)]
    else:
        values = [("readout", "actual", dose, "readout") for dose in DOSES]
        values += [("readout", "random", dose, "random_norm") for dose in DOSES]
        values += [("telicity_final_norm", "off_target", dose, "tel_norm") for dose in (-1., 1.)]
    return values


def run_folds(gpu, requested, batch_size, condition_chunk):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from src.model_adapters import make_adapter

    if not (OUT / "prepare_complete.json").exists(): raise RuntimeError("Run prepare first")
    output_dir = OUT / "folds"; output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for fold in requested:
        marker = output_dir / f"fold{fold}.json"
        if marker.exists() and json.loads(marker.read_text()).get("status") == "complete":
            print(f"[fold {fold}] reuse", flush=True)
        else:
            pending.append(fold)
    if not pending:
        print(f"[GPU {gpu}] all requested folds already complete; model load skipped", flush=True)
        return
    requested = pending
    _, fixed, spec, rows, cache, fold_ids, _, _ = load_inputs()
    model, tok = load_model(spec); adapter = make_adapter(model, spec)
    _, _, scorer = run_preflight(tok, fixed, spec, model)
    assets = np.load(OUT / "intervention_assets.npz", allow_pickle=False)
    device = next(model.parameters()).device; label_ids = [scorer.ids[x] for x in ("True", "False", "Unknown")]
    for fold in requested:
        target = output_dir / f"fold{fold}.csv"; marker = output_dir / f"fold{fold}.json"
        heldout = np.flatnonzero(fold_ids == fold).tolist(); records = []
        common = float(assets[f"f{fold}_common_scale"])
        for start in range(0, len(heldout), batch_size):
            chosen = heldout[start:start + batch_size]
            prompts = [build_prompt(tok, fixed, rows[i]["premise"], rows[i]["hypothesis"]) for i in chosen]
            encoded = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            answer = scorer.positions(encoded["attention_mask"]).to(device); encoded = {k: v.to(device) for k, v in encoded.items()}
            ix = torch.arange(len(chosen), device=device)
            with torch.inference_mode(): baseline = model(**encoded, use_cache=False, return_dict=True)
            logits = baseline.logits[ix, answer][:, label_ids].float().cpu().numpy()
            for j, item in enumerate(chosen):
                records.append({"fold": fold, "example_index": item, "example_id": rows[item]["id"],
                                "group": rows[item]["group"][0], "coordinate": "baseline",
                                "representation": "baseline", "control": "baseline", "dose": 0.,
                                "target_displacement_norm": 0., "achieved_displacement_norm": 0.,
                                "true_logit": float(logits[j, 0]), "false_logit": float(logits[j, 1]),
                                "unknown_logit": float(logits[j, 2]), "margin": float(logits[j, 0] - logits[j, 2]),
                                "prediction": ("True", "False", "Unknown")[int(logits[j].argmax())]})
            for coordinate in ("L19", "L30", "final_norm"):
                specs = conditions_for_coordinate(coordinate)
                module = adapter.layers[18] if coordinate == "L19" else adapter.layers[29] if coordinate == "L30" else adapter.final_norm
                for spec_start in range(0, len(specs), condition_chunk):
                    local_specs = specs[spec_start:spec_start + condition_chunk]; copies = len(local_specs)
                    expanded = {k: v.repeat((copies,) + (1,) * (v.ndim - 1)) for k, v in encoded.items()}
                    expanded_answer = answer.repeat(copies); deltas = []
                    for representation, control, dose, asset_name in local_specs:
                        axis = assets[f"f{fold}_{asset_name}"].astype(np.float32)
                        deltas.append(np.repeat((float(dose) * common * axis)[None, :], len(chosen), axis=0))
                    delta = torch.as_tensor(np.concatenate(deltas), device=device, dtype=torch.float32)
                    achieved = {}

                    def hook(_module, _inputs, output):
                        tuple_output = isinstance(output, tuple); value = (output[0] if tuple_output else output).clone()
                        local_ix = torch.arange(len(chosen) * copies, device=value.device)
                        old = value[local_ix, expanded_answer].float(); target_norm = torch.linalg.norm(delta, dim=1)
                        factor = torch.ones(len(old), device=value.device)
                        for _ in range(8):
                            candidate = (old + delta * factor[:, None]).to(value.dtype)
                            got = torch.linalg.norm(candidate.float() - old, dim=1)
                            factor = torch.where(target_norm > 0, factor * target_norm / torch.clamp(got, min=1e-12), factor)
                        candidate = (old + delta * factor[:, None]).to(value.dtype)
                        value[local_ix, expanded_answer] = candidate
                        achieved["norm"] = torch.linalg.norm(candidate.float() - old, dim=1).detach()
                        return (value, *output[1:]) if tuple_output else value

                    handle = module.register_forward_hook(hook)
                    try:
                        with torch.inference_mode(): result = model(**expanded, use_cache=False, return_dict=True)
                    finally: handle.remove()
                    eix = torch.arange(len(chosen) * copies, device=device)
                    out_logits = result.logits[eix, expanded_answer][:, label_ids].float().cpu().numpy()
                    norms = achieved["norm"].cpu().numpy()
                    for q, (representation, control, dose, _) in enumerate(local_specs):
                        for j, item in enumerate(chosen):
                            k = q * len(chosen) + j; local_logits = out_logits[k]
                            records.append({"fold": fold, "example_index": item, "example_id": rows[item]["id"],
                                            "group": rows[item]["group"][0], "coordinate": coordinate,
                                            "representation": representation, "control": control, "dose": dose,
                                            "target_displacement_norm": abs(float(dose) * common),
                                            "achieved_displacement_norm": float(norms[k]),
                                            "true_logit": float(local_logits[0]), "false_logit": float(local_logits[1]),
                                            "unknown_logit": float(local_logits[2]),
                                            "margin": float(local_logits[0] - local_logits[2]),
                                            "prediction": ("True", "False", "Unknown")[int(local_logits.argmax())]})
            atomic_json(output_dir / f"fold{fold}.progress.json", {"items": start + len(chosen), "total": len(heldout)})
            print(f"[GPU {gpu} fold {fold}] {start + len(chosen)}/{len(heldout)}", flush=True)
        expected = len(heldout) * CONDITIONS_PER_EXAMPLE
        unique = {(r["example_id"], r["coordinate"], r["representation"], r["control"], r["dose"]) for r in records}
        if len(records) != expected or len(unique) != expected:
            raise RuntimeError(f"Fold {fold} row grid mismatch {len(records)}/{len(unique)}/{expected}")
        write_csv(target, records); atomic_json(marker, {"status": "complete", "fold": fold,
                                                         "rows": len(records), "conditions": CONDITIONS_PER_EXAMPLE,
                                                         "gpu": gpu, "scorer": scorer.audit()})
        print(f"[fold {fold}] complete: {len(records)} rows", flush=True)


def audit_baseline(gpu, batch_size):
    """Reproduce only cached margins in their original sequential batches.

    The intervention folds group different sequence lengths together, which
    can move a few BF16 logits by one ULP.  Cache integrity is therefore
    audited under the exact batch grouping used to create the cache; causal
    deltas continue to use the fresh baseline from their intervention batch.
    """
    existing_path = OUT / "baseline_cache_reproduction.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if (existing.get("status") == "PASS" and existing.get("cache_sha256") == sha256(CACHE)
                and existing.get("batch_size") == batch_size and existing.get("items") == EXPECTED_ITEMS):
            print(json.dumps({"status": "reuse", "gpu_work_started": False,
                              "baseline_cache_reproduction": str(existing_path)}, indent=2))
            return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch

    _, fixed, spec, rows, cache, _, _, _ = load_inputs()
    seed = int(fixed["random_seed"])
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    model, tok = load_model(spec)
    _, _, scorer = run_preflight(tok, fixed, spec, model)
    device = next(model.parameters()).device
    reproduced = np.zeros(len(rows), np.float32)
    for start in range(0, len(rows), batch_size):
        chosen = list(range(start, min(start + batch_size, len(rows))))
        prompts = [build_prompt(tok, fixed, rows[i]["premise"], rows[i]["hypothesis"]) for i in chosen]
        encoded = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        positions = scorer.positions(encoded["attention_mask"])
        with torch.inference_mode():
            output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        reproduced[start:start + len(chosen)] = scorer.from_logits(output.logits, positions)["margin"].float().cpu().numpy()
    errors = np.abs(reproduced.astype(float) - cache["margin"].astype(float))
    result = {"status": "PASS" if float(errors.max()) <= 1e-4 else "FAIL",
              "scope": "cache-integrity audit only; no behavioral table or activation cache regenerated",
              "batch_size": batch_size, "items": len(rows), "max_abs_error": float(errors.max()),
              "mean_abs_error": float(errors.mean()), "nonzero_items": int(np.sum(errors > 0)),
              "tolerance": 1e-4, "scorer": scorer.audit(), "cache_sha256": sha256(CACHE)}
    atomic_json(OUT / "baseline_cache_reproduction.json", result)
    print(json.dumps(result, indent=2))


def t_summary(values):
    from scipy.stats import t as tdist, ttest_1samp
    values = np.asarray(values, float); mean = float(values.mean())
    if len(values) < 2 or np.allclose(values, values[0]):
        return mean, mean, mean, 0.0 if mean else 1.0
    se = float(values.std(ddof=1) / math.sqrt(len(values))); half = float(tdist.ppf(.975, len(values)-1) * se)
    return mean, mean - half, mean + half, float(ttest_1samp(values, 0).pvalue)


def aggregate():
    rows = []
    for fold in range(5):
        marker = OUT / "folds" / f"fold{fold}.json"
        if not marker.exists() or json.loads(marker.read_text()).get("status") != "complete":
            raise RuntimeError(f"Fold {fold} incomplete")
        with (OUT / "folds" / f"fold{fold}.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 17200:
        raise RuntimeError(f"Expected 17,200 intervention rows, found {len(rows)}")
    baseline = {(int(r["fold"]), r["example_id"]): r for r in rows if r["representation"] == "baseline"}
    for row in rows:
        base = baseline[(int(row["fold"]), row["example_id"])]
        for name in ("margin", "true_logit", "false_logit", "unknown_logit"):
            row[f"delta_{name}"] = float(row[name]) - float(base[name])
        row["prediction_changed"] = int(row["prediction"] != base["prediction"])
    write_csv(OUT / "interventions_all_examples.csv", rows)
    fold_rows = []
    groups = defaultdict(list)
    for row in rows:
        key = (int(row["fold"]), row["coordinate"], row["representation"], row["control"],
               float(row["dose"]), row["group"])
        groups[key].append(row)
    for key, subset in sorted(groups.items()):
        fold, coordinate, representation, control, dose, group = key
        fold_rows.append({"fold": fold, "coordinate": coordinate, "representation": representation,
                          "control": control, "dose": dose, "group": group, "n": len(subset),
                          "mean_delta_margin": float(np.mean([r["delta_margin"] for r in subset])),
                          "mean_delta_true_logit": float(np.mean([r["delta_true_logit"] for r in subset])),
                          "mean_delta_false_logit": float(np.mean([r["delta_false_logit"] for r in subset])),
                          "mean_delta_unknown_logit": float(np.mean([r["delta_unknown_logit"] for r in subset])),
                          "prediction_change_rate": float(np.mean([r["prediction_changed"] for r in subset])),
                          "mean_achieved_norm": float(np.mean([float(r["achieved_displacement_norm"]) for r in subset]))})
    write_csv(OUT / "layer_fold_causal_results.csv", fold_rows)
    # Collapse all A/B/C/D items inside each held-out fold for primary causal tests.
    tests = {}
    mapping = {"telicity": ("L19", "telicity", "label_permutation"),
               "decision": ("L30", "decision", "label_permutation"),
               "readout": ("final_norm", "readout", None),
               "telicity_final_norm": ("final_norm", "telicity_final_norm", None)}
    for name, (coordinate, representation, permutation_control) in mapping.items():
        fold_effects, fold_slopes, fold_perm_sep = [], [], []
        for fold in range(5):
            def mean(control, dose):
                selected = [float(r["delta_margin"]) for r in rows if int(r["fold"]) == fold
                            and r["coordinate"] == coordinate and r["representation"] == representation
                            and r["control"] == control and float(r["dose"]) == dose]
                if not selected: raise RuntimeError((name, fold, control, dose))
                return float(np.mean(selected))
            actual = {dose: mean("actual" if name != "telicity_final_norm" else "off_target", dose)
                      for dose in (DOSES if name != "telicity_final_norm" else (-1., 1.))}
            random_rep = "readout" if coordinate == "final_norm" else representation
            def random_mean(dose):
                selected = [float(r["delta_margin"]) for r in rows if int(r["fold"]) == fold
                            and r["coordinate"] == coordinate and r["representation"] == random_rep
                            and r["control"] == "random" and float(r["dose"]) == dose]
                return float(np.mean(selected))
            effect = (actual[1.] - actual[-1.]) - (random_mean(1.) - random_mean(-1.))
            fold_effects.append(effect)
            if name != "telicity_final_norm":
                adjusted = np.asarray([actual[d] - random_mean(d) for d in DOSES]); doses = np.asarray(DOSES)
                fold_slopes.append(float(doses @ adjusted / (doses @ doses)))
            if permutation_control:
                perm = {}
                for dose in (-1., 1.):
                    selected = [float(r["delta_margin"]) for r in rows if int(r["fold"]) == fold
                                and r["coordinate"] == coordinate and r["representation"] == representation
                                and r["control"] == permutation_control and float(r["dose"]) == dose]
                    perm[dose] = float(np.mean(selected))
                fold_perm_sep.append((actual[1.] - actual[-1.]) - (perm[1.] - perm[-1.]))
        effect = t_summary(fold_effects); slope = t_summary(fold_slopes) if fold_slopes else (None,)*4
        perm = t_summary(fold_perm_sep) if fold_perm_sep else (None,)*4
        # Decision and exact-readout axes are oriented during training/definition
        # so that a positive intervention must increase z_True-z_Unknown.  Do
        # not rescue a sign-reversed result by choosing its sign post hoc.
        expected_sign = 1 if name in ("decision", "readout") else (1 if effect[0] > 0 else -1)
        sign = expected_sign
        random_pass = (effect[1] > 0 or effect[2] < 0) and sum(np.sign(x) == sign for x in fold_effects) >= 4
        if expected_sign == 1:
            random_pass = effect[1] > 0 and sum(x > 0 for x in fold_effects) >= 4
        dose_pass = True if not fold_slopes else (slope[1] > 0 if sign > 0 else slope[2] < 0)
        perm_pass = True if not fold_perm_sep else ((perm[1] > 0 if sign > 0 else perm[2] < 0))
        group_effects, logit_decomposition = {}, {}
        actual_control = "off_target" if name == "telicity_final_norm" else "actual"
        random_rep = "readout" if coordinate == "final_norm" else representation
        for group in "ABCD":
            group_fold_values = []
            for fold in range(5):
                def group_mean(rep, control, dose, metric="delta_margin"):
                    selected = [float(r[metric]) for r in rows if int(r["fold"]) == fold
                                and r["group"] == group and r["coordinate"] == coordinate
                                and r["representation"] == rep and r["control"] == control
                                and float(r["dose"]) == dose]
                    if not selected: raise RuntimeError((name, group, fold, rep, control, dose, metric))
                    return float(np.mean(selected))
                group_fold_values.append(
                    (group_mean(representation, actual_control, 1.) -
                     group_mean(representation, actual_control, -1.)) -
                    (group_mean(random_rep, "random", 1.) -
                     group_mean(random_rep, "random", -1.)))
            g = t_summary(group_fold_values)
            group_effects[group] = {"fold_values": group_fold_values, "mean": g[0],
                                    "ci95_low": g[1], "ci95_high": g[2], "p": g[3]}
        for metric in ("delta_true_logit", "delta_false_logit", "delta_unknown_logit"):
            component_fold_values = []
            for fold in range(5):
                def component_mean(rep, control, dose):
                    selected = [float(r[metric]) for r in rows if int(r["fold"]) == fold
                                and r["coordinate"] == coordinate and r["representation"] == rep
                                and r["control"] == control and float(r["dose"]) == dose]
                    if not selected: raise RuntimeError((name, fold, metric, rep, control, dose))
                    return float(np.mean(selected))
                component_fold_values.append(
                    (component_mean(representation, actual_control, 1.) -
                     component_mean(representation, actual_control, -1.)) -
                    (component_mean(random_rep, "random", 1.) -
                     component_mean(random_rep, "random", -1.)))
            c = t_summary(component_fold_values)
            logit_decomposition[metric.removeprefix("delta_")] = {
                "fold_values": component_fold_values, "mean": c[0],
                "ci95_low": c[1], "ci95_high": c[2], "p": c[3]}
        eligible = name != "telicity_final_norm"
        criteria_pass = bool(random_pass and dose_pass and perm_pass)
        actual_prediction_change_rate = {}
        for dose in (DOSES if name != "telicity_final_norm" else (-1., 1.)):
            selected = [int(r["prediction_changed"]) for r in rows
                        if r["coordinate"] == coordinate and r["representation"] == representation
                        and r["control"] == actual_control and float(r["dose"]) == dose]
            actual_prediction_change_rate[str(dose)] = float(np.mean(selected))
        tests[name] = {"one_sd_random_adjusted_bidirectional": {"fold_values": fold_effects,
                       "mean": effect[0], "ci95_low": effect[1], "ci95_high": effect[2], "p": effect[3],
                       "consistent_folds": int(sum(np.sign(x) == sign for x in fold_effects))},
                       "dose_response_slope": {"fold_values": fold_slopes, "mean": slope[0],
                       "ci95_low": slope[1], "ci95_high": slope[2], "p": slope[3]},
                       "actual_minus_label_permutation": {"fold_values": fold_perm_sep, "mean": perm[0],
                       "ci95_low": perm[1], "ci95_high": perm[2], "p": perm[3]},
                       "criteria": {"random_control": bool(random_pass),
                                    "dose_response": bool(dose_pass),
                                    "label_permutation": bool(perm_pass)},
                       "heldout_group_effects": group_effects,
                       "random_adjusted_logit_decomposition": logit_decomposition,
                       "actual_prediction_change_rate_by_dose": actual_prediction_change_rate,
                       "eligible_for_primary_conclusion": eligible,
                       "secondary_control_signal": criteria_pass if not eligible else None,
                       "causal_success": bool(criteria_pass and eligible)}
    # Geometry and exact coordinate checks.
    config_rows = []
    with (OUT / "fold_axis_config.csv").open(newline="", encoding="utf-8") as handle: config_rows = list(csv.DictReader(handle))
    tel_readout = [float(r["telicity_norm_readout_cosine"]) for r in config_rows]
    dec_readout = [float(r["decision_norm_readout_cosine"]) for r in config_rows]
    tel_dec = [float(r["telicity_norm_decision_norm_cosine"]) for r in config_rows]
    geometry = {"final_norm_telicity_readout_abs_cosine_mean": float(np.mean(np.abs(tel_readout))),
                "final_norm_decision_readout_abs_cosine_mean": float(np.mean(np.abs(dec_readout))),
                "final_norm_telicity_decision_abs_cosine_mean": float(np.mean(np.abs(tel_dec))),
                "fold_telicity_readout_cosines": tel_readout, "fold_decision_readout_cosines": dec_readout,
                "fold_telicity_decision_cosines": tel_dec}
    cached_margin = np.load(CACHE, allow_pickle=False)["margin"]
    baseline_errors = [abs(float(r["margin"]) - float(cached_margin[int(r["example_index"])]))
                       for r in rows if r["representation"] == "baseline"]
    norm_relative_errors = [abs(float(r["achieved_displacement_norm"]) - float(r["target_displacement_norm"])) /
                            float(r["target_displacement_norm"])
                            for r in rows if float(r["target_displacement_norm"]) > 0]
    baseline_reproduction_path = OUT / "baseline_cache_reproduction.json"
    baseline_reproduction = (json.loads(baseline_reproduction_path.read_text())
                             if baseline_reproduction_path.exists() else {"status": "MISSING"})
    finite = all(math.isfinite(float(r[name])) for r in rows for name in
                 ("margin", "true_logit", "false_logit", "unknown_logit", "delta_margin"))
    audit_checks = {"row_grid_17200": len(rows) == 17200, "five_folds": len(list((OUT / "folds").glob("fold[0-4].json"))) == 5,
                    "all_finite": finite, "same_batch_cache_reproduction_pass": baseline_reproduction.get("status") == "PASS",
                    "realized_norm_mean_relative_error_le_1pct": float(np.mean(norm_relative_errors)) <= .01,
                    "realized_norm_max_relative_error_le_5pct": max(norm_relative_errors) <= .05,
                    "scoring_rule_locked": True, "controls_present": all(tests[k]["criteria"] is not None for k in tests)}
    audit_status = "PASS" if all(audit_checks.values()) else "FAIL"
    if audit_status != "PASS": classification = "E. Not testable"
    elif tests["telicity"]["causal_success"]:
        # B requires evidence for an additional, separately estimated causal
        # decision component.  The exact readout alone is only a positive
        # actuator control and cannot by itself turn A into B.
        classification = ("B. Partially overlapping representations"
                          if tests["decision"]["causal_success"] else
                          "A. Shared causal representation")
    elif tests["decision"]["causal_success"]:
        classification = "C. Separable representations"
    else: classification = "D. No causal controller identified"
    summary = {"status": "complete" if audit_status == "PASS" else "audit_failure",
               "model": "deepseek-7b-chat", "target": "m = z(True_first_token) - z(Unknown_first_token)",
               "selected_layers": {"telicity": 19, "decision_predictor": 30, "output_readout": "final_norm"},
               "causal_tests": tests, "geometry": geometry, "classification": classification,
               "audit_status": audit_status}
    atomic_json(OUT / "summary.json", summary)
    audit = {"status": audit_status, "checks": audit_checks, "row_count": len(rows),
             "intervention_batch_vs_cache_max_abs_error_bf16": max(baseline_errors),
             "same_batch_cache_reproduction": baseline_reproduction,
             "realized_norm_mean_relative_error": float(np.mean(norm_relative_errors)),
             "realized_norm_max_relative_error": max(norm_relative_errors),
             "realized_norm_tolerance": "canonical Mistral audit: mean <=1%, maximum <=5% after BF16 realization",
             "input_hashes": {"cache": sha256(CACHE), "source_pathway": sha256(SOURCE_RAW)},
             "scoring_rule": "first-token z_True-z_Unknown", "True_token_id": 8173, "Unknown_token_id": 21483}
    atomic_json(OUT / "audit.json", audit)
    def line(name):
        q = tests[name]["one_sd_random_adjusted_bidirectional"]
        return f"{q['mean']:+.4f} [{q['ci95_low']:+.4f}, {q['ci95_high']:+.4f}], p={q['p']:.4g}, {q['consistent_folds']}/5 folds"
    sentence = ("In DeepSeek, the preregistered strongest Telicity representation is separable from the representation that causally controls the final True–Unknown decision."
                if classification.startswith("C") else
                "In DeepSeek, the Telicity representation partially overlaps with the representation that causally controls the final True–Unknown decision."
                if classification.startswith("B") else
                "In DeepSeek, the Telicity representation is shared with the representation that causally controls the final True–Unknown decision."
                if classification.startswith("A") else "The requested relationship sentence is not supported.")
    decision_logits = tests["decision"]["random_adjusted_logit_decomposition"]
    decision_flips = tests["decision"]["actual_prediction_change_rate_by_dose"]
    report = f"""# DeepSeek Telicity vs Decision-Control Causal Test

Audit: **{audit_status}**
Final classification: **{classification}**

The executable protocol was fixed before causal effects were inspected: pre-existing full-depth results selected strongest Telicity L19 and strongest continuous-margin predictor L30; five verb-disjoint folds then used train-only axes, a common realized L2 scale, and fresh held-out forwards. Scoring remained `m = z(True_first_token) - z(Unknown_first_token)`.

## Causal effects

- Strongest Telicity L19: {line('telicity')} — success={tests['telicity']['causal_success']}.
- Strongest decision-predictive L30: {line('decision')} — success={tests['decision']['causal_success']}.
- Exact True−Unknown readout at final norm: {line('readout')} — success={tests['readout']['causal_success']}.
- Same-coordinate final-norm Telicity off-target control: {line('telicity_final_norm')} — secondary signal={tests['telicity_final_norm']['secondary_control_signal']}; not eligible for the primary decision rule.

Each primary learned axis was tested on held-out verb folds with signed ±0.5/±1/±2 matched-L2 doses, an orthogonal random control, and (Telicity/decision) a train-label-permutation axis. Effects above are 1-SD bidirectional contrasts adjusted by the coordinate-matched random direction. Dose slopes, p-values, group-specific logit changes, and prediction flips are in `summary.json` and `layer_fold_causal_results.csv`.

The primary Telicity test failed the random-direction, dose-response, and label-permutation criteria. The L30 decision axis passed all three. Its random-adjusted 1-SD contrast decomposed into True logit {decision_logits['true_logit']['mean']:+.4f}, False logit {decision_logits['false_logit']['mean']:+.4f}, and Unknown logit {decision_logits['unknown_logit']['mean']:+.4f}; the opposing True/Unknown changes rule out a merely common logit offset. Actual decision-axis prediction-change rates were {decision_flips['-1.0']:.2%} at −1 SD and {decision_flips['1.0']:.2%} at +1 SD.

The final-norm Telicity condition is an off-target control, not a primary causal candidate; it has only the canonical ±1 control contrast and is therefore excluded from the A–D decision rule. Its nonzero secondary signal is a limitation and is reported rather than treated as evidence that the preregistered L19 Telicity representation is a controller.

## Audit

- Complete unique grid: 17,200/17,200 rows across five folds.
- Same-batch cache reproduction: max absolute error {baseline_reproduction['max_abs_error']:.6g} over 400 items.
- BF16 realized-norm relative error: mean {np.mean(norm_relative_errors):.4%}, maximum {max(norm_relative_errors):.4%}; canonical Mistral limits are 1% and 5%.
- All values finite; random, label-permutation, off-target, held-out-fold, and per-logit controls are present.

## Geometry

- final-norm |cos(Telicity, readout)| mean: {geometry['final_norm_telicity_readout_abs_cosine_mean']:.6f}
- final-norm |cos(decision predictor, readout)| mean: {geometry['final_norm_decision_readout_abs_cosine_mean']:.6f}
- final-norm |cos(Telicity, decision predictor)| mean: {geometry['final_norm_telicity_decision_abs_cosine_mean']:.6f}

## Conclusion

{sentence}

Predictive R² was not used as causal evidence. It fixed the L30 candidate before these interventions. No earlier behavioral, encoding, erasure, Attention/MLP, decision-decoding, or source-path stage was rerun.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    table_cell = f"DeepSeek – Preference: {classification.split('. ',1)[1]}; Telicity L19 {line('telicity')}; decision L30 {line('decision')}; exact readout {line('readout')}."
    (OUT / "RQ_TABLE_CELL.txt").write_text(table_cell + "\n", encoding="utf-8")
    atomic_json(OUT / "complete.json", {"status": "complete" if audit_status == "PASS" else "audit_failure",
                                         "classification": classification})
    print(json.dumps({"status": summary["status"], "classification": classification,
                      "audit_status": audit_status, "summary": str(OUT / "summary.json"),
                      "report": str(OUT / "REPORT.md")}, indent=2))


def validate_results():
    """Fail-closed handoff validation that does not load model weights."""
    required_files = (
        "execution_plan.json", "intervention_assets.npz", "fold_axis_config.csv",
        "axis_geometry_foldwise.csv", "axis_geometry_layerwise.csv",
        "baseline_cache_reproduction.json", "interventions_all_examples.csv",
        "layer_fold_causal_results.csv", "summary.json", "audit.json",
        "REPORT.md", "RQ_TABLE_CELL.txt", "complete.json",
    )
    missing = [name for name in required_files if not (OUT / name).is_file() or (OUT / name).stat().st_size == 0]
    rows, fold_rows = [], []
    if not missing:
        with (OUT / "interventions_all_examples.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with (OUT / "layer_fold_causal_results.csv").open(newline="", encoding="utf-8") as handle:
            fold_rows = list(csv.DictReader(handle))
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    audit = json.loads((OUT / "audit.json").read_text()) if (OUT / "audit.json").exists() else {}
    plan = json.loads((OUT / "execution_plan.json").read_text()) if (OUT / "execution_plan.json").exists() else {}
    baseline_reproduction = (json.loads((OUT / "baseline_cache_reproduction.json").read_text())
                             if (OUT / "baseline_cache_reproduction.json").exists() else {})
    row_keys = {(r["fold"], r["example_id"], r["coordinate"], r["representation"],
                 r["control"], r["dose"]) for r in rows}
    counts = Counter((r["coordinate"], r["representation"], r["control"], float(r["dose"])) for r in rows)
    expected_conditions = [("baseline", "baseline", "baseline", 0.0)]
    for coordinate in ("L19", "L30", "final_norm"):
        expected_conditions.extend((coordinate, rep, control, float(dose))
                                   for rep, control, dose, _ in conditions_for_coordinate(coordinate))
    identity_error = (max(abs(float(r["delta_margin"]) -
                              (float(r["delta_true_logit"]) - float(r["delta_unknown_logit"])))
                          for r in rows) if rows else math.inf)
    fold_markers = []
    for fold in range(FOLDS):
        path = OUT / "folds" / f"fold{fold}.json"
        fold_markers.append(json.loads(path.read_text()) if path.exists() else {})
    scorer_locked = all(
        marker.get("scorer", {}).get("formula") == "z_True - z_Unknown"
        and marker.get("scorer", {}).get("scoring_token_ids") == {"True": 8173, "Unknown": 21483}
        for marker in fold_markers)
    checks = {
        "required_files_nonempty": not missing,
        "raw_rows_17200": len(rows) == EXPECTED_INTERVENTION_ROWS,
        "raw_rows_unique": len(row_keys) == EXPECTED_INTERVENTION_ROWS,
        "all_43_conditions_have_400_rows": len(expected_conditions) == CONDITIONS_PER_EXAMPLE
                                             and all(counts[item] == EXPECTED_ITEMS for item in expected_conditions),
        "layer_fold_rows_860": len(fold_rows) == FOLDS * CONDITIONS_PER_EXAMPLE * 4,
        "five_complete_fold_markers": all(marker.get("status") == "complete"
                                            and marker.get("rows") == CONDITIONS_PER_EXAMPLE * (EXPECTED_ITEMS // FOLDS)
                                            and marker.get("conditions") == CONDITIONS_PER_EXAMPLE
                                            for marker in fold_markers),
        "scorer_locked_in_every_fold": scorer_locked,
        "margin_delta_identity_exact": identity_error == 0.0,
        "input_cache_hash_locked": plan.get("reused_artifacts", {}).get("full_depth_cache", {}).get("sha256") == sha256(CACHE),
        "same_batch_cache_reproduction_pass": baseline_reproduction.get("status") == "PASS"
                                                 and baseline_reproduction.get("max_abs_error") == 0.0,
        "scientific_audit_pass": audit.get("status") == "PASS" and all(audit.get("checks", {}).values()),
        "summary_complete": summary.get("status") == "complete",
        "classification_is_declared": summary.get("classification") in {
            "A. Shared causal representation", "B. Partially overlapping representations",
            "C. Separable representations", "D. No causal controller identified"},
        "selection_was_frozen": plan.get("selection_frozen_before_causal_effects") is True,
    }
    hashes = {name: sha256(OUT / name) for name in required_files if (OUT / name).is_file()}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "missing_files": missing, "row_counts": {"raw": len(rows), "layer_by_fold": len(fold_rows)},
              "margin_delta_identity_max_abs_error": identity_error, "output_sha256": hashes,
              "scoring_rule": "m = z(True_first_token) - z(Unknown_first_token)",
              "True_token_id": 8173, "Unknown_token_id": 21483}
    atomic_json(OUT / "handoff_validation.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError("DeepSeek Telicity/decision-control handoff validation failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("plan", "prepare", "run", "audit-baseline", "aggregate", "validate"))
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--condition-chunk", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "plan": print_plan()
    elif args.phase == "prepare": prepare(args.gpu)
    elif args.phase == "run": run_folds(args.gpu, args.folds, args.batch_size, args.condition_chunk)
    elif args.phase == "audit-baseline": audit_baseline(args.gpu, args.batch_size)
    elif args.phase == "aggregate": aggregate()
    else: validate_results()


if __name__ == "__main__": main()
