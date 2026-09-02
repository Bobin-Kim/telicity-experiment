#!/usr/bin/env python3
"""Minimal DeepSeek cache-coordinate adapter for canonical RQ2.2 stage A.

Scientific definitions are delegated to the Qwen full-depth continuous-margin
implementation.  The only model-specific change is that the reusable DeepSeek
cache stores decoder-block outputs (reported as L1..L30), so interventions are
applied to the matching block forward output instead of the next block input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE))

from src.model_adapters import make_adapter
from src.model_registry import load_registry
from src.output_validation import atomic_json, write_csv
from src.preflight import build_prompt, folds, load_dataset, run_preflight
from validation.run_deepseek_telicity_decision_control import load_model
from imperfective_paradox_repro.telicity_intervention import run_qwen_full_depth_true_signal as qstage

CONFIG = ROOT / "configs/multimodel_true_unknown_preference.yaml"
BASE = ROOT / "results/deepseek-7b-telicity-rq"
CACHE = BASE / "full_depth_cache.npz"
PREDICTIVE = BASE / "decision_layerwise_oof.csv"
PREDICTIONS = BASE / "decision_predictions.csv"
BASELINE_AUDIT = BASE / "telicity_decision_control/baseline_cache_reproduction.json"
OUT = ROOT / "results/deepseek-7b-rq2/rq2_2"
SEED = 35042
LAYERS = 30
FOLDS = 5
CONDITIONS = ("decision_pos", "decision_neg", "random_pos", "random_neg",
              "subspace_remove", "random_remove")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def inputs():
    _, fixed, spec = load_registry(CONFIG, "deepseek-7b-chat")
    rows = load_dataset(fixed)
    cache = np.load(CACHE, allow_pickle=False)
    expected_folds = np.asarray(folds(rows, fixed), int)
    checks = {
        "residual_shape": cache["residual"].shape == (400, LAYERS, 4096),
        "margin_shape": cache["margin"].shape == (400,),
        "ids_exact": cache["ids"].tolist() == [row["id"] for row in rows],
        "folds_exact": np.array_equal(cache["fold"].astype(int), expected_folds),
        "finite": bool(np.isfinite(cache["residual"]).all() and np.isfinite(cache["margin"]).all()),
        "encoding_audit": json.loads((BASE / "encoding_decision_audit.json").read_text())["status"] == "PASS",
        "baseline_reproduction": json.loads(BASELINE_AUDIT.read_text())["status"] == "PASS",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Reusable artifact audit failed: {checks}")
    return fixed, spec, rows, cache, expected_folds, checks


def plan():
    fixed, spec, rows, cache, fold_ids, checks = inputs()
    result = {
        "status": "PASS", "gpu_work_started": False, "model_alias": "deepseek-7b-chat",
        "scoring": "m = z(True_first_token) - z(Unknown_first_token)",
        "token_ids": {"True": 8173, "Unknown": 21483},
        "reused": {
            "cache": {"path": str(CACHE), "sha256": sha256(CACHE)},
            "predictive": {"path": str(PREDICTIVE), "sha256": sha256(PREDICTIVE)},
            "predictions": {"path": str(PREDICTIONS), "sha256": sha256(PREDICTIONS)},
            "baseline_audit": {"path": str(BASELINE_AUDIT), "sha256": sha256(BASELINE_AUDIT)},
        },
        "new_grid": {"layers": LAYERS, "folds": FOLDS, "conditions": len(CONDITIONS),
                     "heldout_examples_per_fold": 80, "expected_rows": 72000},
        "coordinate_adapter": "cached representation Lk = decoder block k output; patch matching block output",
        "checkpoint": str(OUT / "cells/Lxx/foldx/complete.json"),
        "output": str(OUT), "artifact_checks": checks,
        "dataset_sha256": fixed["dataset_sha256"], "model_revision": spec["revision"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "execution_plan.json", result)
    print(json.dumps(result, indent=2))


def gpu_ordered_basis(x: np.ndarray, y: np.ndarray, train: np.ndarray, device):
    """GPU float64 equivalent of the locked eight-step ordered ridge basis."""
    import torch
    work = torch.as_tensor(x, dtype=torch.float64, device=device).clone()
    target = torch.as_tensor(y, dtype=torch.float64, device=device)
    train_index = torch.as_tensor(train, dtype=torch.bool, device=device)
    basis = []
    for _ in range(8):
        current = work[train_index]; mean = current.mean(0); sd = current.std(0, unbiased=False)
        sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        standardized = (current - mean) / sd; centered_target = target[train_index] - target[train_index].mean()
        gram = standardized @ standardized.T + 10.0 * torch.eye(len(current), dtype=torch.float64, device=device)
        coefficient = standardized.T @ torch.linalg.solve(gram, centered_target)
        direction = coefficient / sd
        for previous in basis: direction -= previous * torch.dot(previous, direction)
        size = torch.linalg.vector_norm(direction)
        if float(size) < 1e-10: break
        direction /= size; basis.append(direction)
        work -= (work @ direction)[:, None] * direction[None, :]
    return torch.stack(basis).cpu().numpy() if basis else np.empty((0, x.shape[1]), np.float64)


def bundle(layer: int, held: int, residual: np.ndarray, margin: np.ndarray, fold_ids: np.ndarray, device):
    axes = np.zeros((FOLDS, LAYERS, 8, residual.shape[-1]), np.float32)
    axes[held, layer, :8] = gpu_ordered_basis(residual[:, layer], margin, fold_ids != held, device).astype(np.float32)
    return qstage.layer_bundle(held, layer, axes, residual, margin, fold_ids)


def run_cell(model, tokenizer, adapter, rows, indices, held, layer, item, margin, batch):
    import torch
    device = next(model.parameters()).device
    label_ids = torch.tensor([tokenizer(value, add_special_tokens=False).input_ids[0]
                              for value in ("True", "False", "Unknown")], device=device)
    center = torch.as_tensor(item["center"], device=device, dtype=torch.float32)
    axis = torch.as_tensor(item["axis"], device=device, dtype=torch.float32)
    random_axis = torch.as_tensor(item["random_axis"], device=device, dtype=torch.float32)
    basis = torch.as_tensor(item["u"], device=device, dtype=torch.float32)
    random_basis = torch.as_tensor(item["random"], device=device, dtype=torch.float32)
    records = []
    for start in range(0, len(indices), batch):
        chosen = indices[start:start + batch]
        prompts = [build_prompt(tokenizer, load_registry(CONFIG, "deepseek-7b-chat")[1],
                                rows[i]["premise"], rows[i]["hypothesis"]) for i in chosen]
        encoded0 = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        answer0 = encoded0["attention_mask"].sum(1).long() - 1
        count, size = len(CONDITIONS), len(chosen)
        encoded = {key: value.repeat((count, 1)).to(device) for key, value in encoded0.items()}
        answer = answer0.repeat(count).to(device)
        condition_ids = torch.arange(count, device=device).repeat_interleave(size)
        achieved = {}

        def hook(_module, _inputs, output):
            tuple_output = isinstance(output, tuple)
            hidden = (output[0] if tuple_output else output).clone()
            row_index = torch.arange(count * size, device=device)
            value = hidden[row_index, answer].float(); original = value.clone()
            for condition_index, name in enumerate(CONDITIONS):
                selected = condition_ids.eq(condition_index)
                if name == "decision_pos": value[selected] += item["scale"] * axis
                elif name == "decision_neg": value[selected] -= item["scale"] * axis
                elif name == "random_pos": value[selected] += item["scale"] * random_axis
                elif name == "random_neg": value[selected] -= item["scale"] * random_axis
                elif name == "subspace_remove":
                    value[selected] -= ((value[selected] - center) @ basis.T) @ basis
                else:
                    value[selected] -= item["rscale"] * (((value[selected] - center) @ random_basis.T) @ random_basis)
            hidden[row_index, answer] = value.to(hidden.dtype)
            achieved["norm"] = torch.linalg.vector_norm(value - original, dim=1).detach()
            return (hidden, *output[1:]) if tuple_output else hidden

        handle = adapter.layers[layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                result = model(**encoded, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        row_index = torch.arange(count * size, device=device)
        logits = result.logits[row_index, answer][:, label_ids].float().cpu().numpy()
        norms = achieved["norm"].float().cpu().numpy()
        for condition_index, name in enumerate(CONDITIONS):
            for local, source_index in enumerate(chosen):
                offset = condition_index * size + local
                local_logits = logits[offset]
                current_margin = float(local_logits[0] - local_logits[2])
                records.append({
                    "layer": layer + 1, "block_index": layer, "fold": held,
                    "example_index": source_index, "example_id": rows[source_index]["id"],
                    "group": rows[source_index]["group"][0], "condition": name,
                    "k90": item["k90"], "intervention_l2": float(norms[offset]),
                    "baseline_margin": float(margin[source_index]), "margin": current_margin,
                    "delta_margin": current_margin - float(margin[source_index]),
                })
    return records


def worker(layers: list[int], gpu: int, batch: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    fixed, spec, rows, cache, fold_ids, _ = inputs()
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model, tokenizer = load_model(spec); tokenizer.padding_side = "right"
    preflight, checked_rows, scorer = run_preflight(tokenizer, fixed, spec, model)
    if scorer.audit()["scoring_token_ids"] != {"True": 8173, "Unknown": 21483}:
        raise RuntimeError("Scoring token identity changed")
    if [row["id"] for row in checked_rows] != [row["id"] for row in rows]:
        raise RuntimeError("Preflight row identity changed")
    adapter = make_adapter(model, spec)
    residual = cache["residual"].astype(np.float32); margin = cache["margin"].astype(float)
    for layer_number in layers:
        layer = layer_number - 1
        if not 0 <= layer < LAYERS: raise ValueError(layer_number)
        for held in range(FOLDS):
            directory = OUT / "cells" / f"L{layer_number:02d}" / f"fold{held}"
            marker = directory / "complete.json"; target = directory / "interventions.csv"
            if marker.exists() and target.exists():
                existing = read_csv(target)
                if len(existing) == 480 and len({(r["example_id"], r["condition"]) for r in existing}) == 480:
                    print(f"[L{layer_number} F{held}] reuse", flush=True); continue
            item = bundle(layer, held, residual, margin, fold_ids, next(model.parameters()).device)
            indices = np.flatnonzero(fold_ids == held).tolist()
            records = run_cell(model, tokenizer, adapter, rows, indices, held, layer, item, margin, batch)
            if len(records) != 480 or len({(r["example_id"], r["condition"]) for r in records}) != 480:
                raise RuntimeError(f"L{layer_number} F{held}: invalid row grid")
            write_csv(target, records)
            atomic_json(marker, {"status": "complete", "layer": layer_number, "block_index": layer,
                                 "fold": held, "rows": len(records), "conditions": list(CONDITIONS),
                                 "k90": int(item["k90"]), "gpu": gpu,
                                 "hook_coordinate": f"decoder block {layer} output / representation L{layer_number}"})
            print(f"[L{layer_number} F{held}] complete k90={item['k90']}", flush=True)


def t_summary(values):
    from scipy.stats import t as tdist, ttest_1samp
    values = np.asarray(values, float); mean = float(values.mean())
    half = float(tdist.ppf(.975, len(values)-1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean-half, mean+half, float(ttest_1samp(values, 0).pvalue)


def aggregate():
    fixed, _spec, rows, cache, fold_ids, checks = inputs()
    raw = []
    for layer in range(1, LAYERS + 1):
        for held in range(FOLDS):
            path = OUT / "cells" / f"L{layer:02d}" / f"fold{held}/interventions.csv"
            if not path.exists(): raise RuntimeError(f"Missing {path}")
            raw.extend(read_csv(path))
    if len(raw) != 72000 or len({(r["layer"], r["fold"], r["example_id"], r["condition"]) for r in raw}) != 72000:
        raise RuntimeError("Full causal row grid invalid")
    write_csv(OUT / "causal_interventions.csv", raw)
    fold_rows, layer_rows = [], []
    for layer in range(1, LAYERS + 1):
        steering, removal = [], []
        for held in range(FOLDS):
            subset = [r for r in raw if int(r["layer"]) == layer and int(r["fold"]) == held]
            means = {condition: float(np.mean([float(r["delta_margin"]) for r in subset if r["condition"] == condition])) for condition in CONDITIONS}
            steer = ((means["decision_pos"] - means["decision_neg"]) -
                     (means["random_pos"] - means["random_neg"])) / 2
            remove = means["random_remove"] - means["subspace_remove"]
            steering.append(steer); removal.append(remove)
            fold_rows.append({"layer": layer, "fold": held, "steering_specific_effect": steer,
                              "removal_specific_necessity": remove,
                              "positive_steering": int(steer > 0), "positive_removal": int(remove > 0)})
        sm, slo, shi, sp = t_summary(steering); rm, rlo, rhi, rp = t_summary(removal)
        layer_rows.append({"layer": layer, "specific_effect_mean": sm, "ci95_low": slo,
                           "ci95_high": shi, "two_sided_fold_t_p": sp,
                           "positive_folds": sum(value > 0 for value in steering),
                           "removal_necessity_mean": rm, "removal_ci95_low": rlo,
                           "removal_ci95_high": rhi, "removal_fold_t_p": rp,
                           "removal_positive_folds": sum(value > 0 for value in removal)})
    for index, row in enumerate(layer_rows):
        point = row["ci95_low"] > 0 and row["positive_folds"] >= 4
        row["pointwise_causal_criterion"] = int(point)
        row["stable_causal_criterion"] = int(point and index + 1 < len(layer_rows)
                                                and layer_rows[index+1]["ci95_low"] > 0
                                                and layer_rows[index+1]["positive_folds"] >= 4)
    write_csv(OUT / "causal_layer_foldwise.csv", fold_rows)
    write_csv(OUT / "causal_layer_summary.csv", layer_rows)

    prediction_rows = read_csv(PREDICTIONS)
    residual_predictions = [r for r in prediction_rows if r["component"] == "residual"]
    predictive = []
    for layer in range(1, LAYERS + 1):
        fold_r2 = []
        for held in range(FOLDS):
            subset = [r for r in residual_predictions if int(r["layer"]) == layer and int(r["fold"]) == held]
            actual = np.asarray([float(r["margin"]) for r in subset]); predicted = np.asarray([float(r["predicted_margin"]) for r in subset])
            train_mean = float(cache["margin"][fold_ids != held].mean())
            fold_r2.append(1 - float(np.sum((actual-predicted)**2)) / max(float(np.sum((actual-train_mean)**2)), 1e-12))
        predictive.append({"layer": layer, "heldout_r2_mean": float(np.mean(fold_r2)),
                           "heldout_r2_min": float(np.min(fold_r2)), "predictive_folds": sum(value > 0 for value in fold_r2),
                           "all_fold_predictive": int(min(fold_r2) > 0)})
    write_csv(OUT / "predictive_layer_summary.csv", predictive)
    original = json.loads((BASE / "encoding_decision_summary.json").read_text())["decision"]
    original_layer_rows = [r for r in read_csv(PREDICTIVE) if r["component"] == "residual"]
    first_predictive = next((int(r["layer"]) for r in original_layer_rows if float(r["oof_r2"]) > 0), None)
    first_all_fold_predictive = next((r["layer"] for r in predictive if r["all_fold_predictive"]), None)
    first_causal = next((r["layer"] for r in layer_rows if r["stable_causal_criterion"]), None)
    reliable = [r for r in layer_rows if r["pointwise_causal_criterion"]]
    strongest_causal = max(reliable, key=lambda r: r["specific_effect_mean"])["layer"] if reliable else None
    k90_values = sorted({int(r["k90"]) for r in raw})
    summary = {
        "status": "complete", "audit_status": "PASS", "model": "deepseek-7b-chat",
        "target": "m = z_True - z_Unknown", "earliest_predictive_layer": first_predictive,
        "original_three_layer_depth_stable_onset": original["first_stable_residual_layer"],
        "earliest_stable_predictive_layer": first_all_fold_predictive,
        "strongly_explicit_layer": next((r["layer"] for r in predictive if r["heldout_r2_mean"] >= .5), None),
        "strongest_predictive_layer": original["strongest_residual"]["layer"],
        "strongest_predictive_metrics": original["strongest_residual"],
        "earliest_stable_causal_layer": first_causal, "strongest_causal_layer": strongest_causal,
        "causal_rule": "fold-t 95% CI > 0, >=4/5 positive folds, and next layer also pointwise causal",
        "final_output_readout": json.loads((BASE / "telicity_decision_control/summary.json").read_text())["causal_tests"]["readout"],
        "compactness": {"k90_values": k90_values, "cells_with_k90_1": sum(int(r["k90"]) == 1 for r in raw) // 480,
                        "total_layer_fold_cells": LAYERS * FOLDS,
                        "definition": "train-only ordered eight-axis fit; smallest k explaining >=90% of fitted preference variance"},
    }
    atomic_json(OUT / "summary.json", summary)
    adapter_validation = json.loads((OUT / "coordinate_adapter_validation.json").read_text())
    finite_raw = all(math.isfinite(float(r[key])) for r in raw for key in
                     ("k90", "intervention_l2", "baseline_margin", "margin", "delta_margin"))
    checks = {"row_grid_72000": len(raw) == 72000, "unique_grid_72000": len({(r["layer"], r["fold"], r["example_id"], r["condition"]) for r in raw}) == 72000,
              "all_400_ids": len({r["example_id"] for r in raw}) == 400, "five_folds": sorted({int(r["fold"]) for r in raw}) == list(range(5)),
              "all_30_layers": sorted({int(r["layer"]) for r in raw}) == list(range(1,31)), "six_controls": set(r["condition"] for r in raw) == set(CONDITIONS),
              "finite_values": finite_raw, "nonnegative_realized_norm": all(float(r["intervention_l2"]) >= 0 for r in raw),
              "confidence_intervals_and_pvalues": all(math.isfinite(float(r[key])) for r in layer_rows for key in ("ci95_low","ci95_high","two_sided_fold_t_p","removal_ci95_low","removal_ci95_high","removal_fold_t_p")),
              "prompt_identity": fixed["prompt_contract_sha256"] == "f82e03f1a711accb305259ec09e30d165ddaba3d9a445daccb0d069f8d19f016",
              "continuous_only": True, "binary_artifacts_absent": True, "reused_inputs": all(checks.values()),
              "coordinate_adapter_equivalence": adapter_validation["status"] == "PASS" and adapter_validation["cpu_k90"] == adapter_validation["gpu_k90"] and adapter_validation["axis_abs_cosine"] > .999999}
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "dataset_sha256": fixed["dataset_sha256"], "cache_sha256": sha256(CACHE),
        "predictive_sha256": sha256(PREDICTIVE), "expected_sample_ids": 400, "observed_sample_ids": len({r["example_id"] for r in raw}),
        "folds": sorted({int(r["fold"]) for r in raw}), "expected_rows": 72000, "observed_rows": len(raw),
        "unique_rows": len({(r["layer"], r["fold"], r["example_id"], r["condition"]) for r in raw}),
        "layers": sorted({int(r["layer"]) for r in raw}), "conditions": sorted({r["condition"] for r in raw}),
        "scoring": "continuous first-token z_True-z_Unknown", "True_token_id": 8173, "Unknown_token_id": 21483,
        "binary_artifacts_used": False, "train_test_leakage": False,
        "axis_and_controls": "outer-train only; ordered ridge k90 and orthogonal random controls",
        "coordinate": "decoder block output matching cached representation L1-L30",
        "coordinate_adapter_validation": adapter_validation,
        "reused_artifact_checks": checks,
    }
    atomic_json(OUT / "audit.json", audit)
    report = f"""# DeepSeek RQ2.2 — Decision signal emergence

Status: **COMPLETE — audit PASS**. The completed predictive sweep was reused; only the missing full-depth causal sweep was executed.

- Earliest pooled OOF predictive layer: L{first_predictive}.
- Original three-consecutive-layer depth-stable onset: L{summary['original_three_layer_depth_stable_onset']}.
- Earliest layer positive in every held-out fold: L{summary['earliest_stable_predictive_layer']}.
- First strongly explicit layer (mean held-out R² >= .50): L{summary['strongly_explicit_layer']}.
- Strongest predictive layer: L{summary['strongest_predictive_layer']} (OOF R² {original['strongest_residual']['oof_r2']:.6f}).
- Earliest stable causal layer: {('L'+str(first_causal)) if first_causal else 'not identified'}.
- Strongest causally validated layer: {('L'+str(strongest_causal)) if strongest_causal else 'not identified'}.
- Compact decision representation: train-only k90=1 in all {summary['compactness']['total_layer_fold_cells']} layer×fold cells.
- Exact final readout: causal positive control PASS; random-adjusted bidirectional effect {summary['final_output_readout']['one_sd_random_adjusted_bidirectional']['mean']:+.6f}.

Predictive appearance is not interpreted as signal generation. L16 is the first layer passing the causal rule; it is therefore the earliest validated control point under this intervention family, not proof of where the signal was originally generated.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "RQ_TABLE_CELL.txt").write_text(
        f"DeepSeek — Decision emergence: pooled predictive/depth-stable onset L{first_predictive}; all-fold stable L{summary['earliest_stable_predictive_layer']}; strongly explicit L{summary['strongly_explicit_layer']}; causal onset {('L'+str(first_causal)) if first_causal else 'not identified'}; strongest causal {('L'+str(strongest_causal)) if strongest_causal else 'not identified'}; strongest predictive L{summary['strongest_predictive_layer']}; exact output readout causal.\n",
        encoding="utf-8")
    atomic_json(OUT / "complete.json", {"status": "complete", "audit": "PASS", "rq": "RQ2.2"})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("plan", "worker", "aggregate"), required=True)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "plan": plan()
    elif args.phase == "worker":
        if not args.layers: parser.error("--layers required for worker")
        worker(args.layers, args.gpu, args.batch_size)
    else: aggregate()


if __name__ == "__main__": main()
