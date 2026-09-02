#!/usr/bin/env python3
"""DeepSeek full-depth Telicity/re-emergence/decision analysis controller."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE))

from src.model_adapters import make_adapter
from src.model_registry import load_registry, resolved_source
from src.output_validation import atomic_json, write_csv
from src.preflight import build_prompt, folds, run_preflight
from imperfective_paradox_repro.telicity_intervention.run_telicity_subspace_removal import (
    fit_ridge_probe, probe_balanced_accuracy,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = 2.7764451052 * float(values.std(ddof=1)) / np.sqrt(5)
    return mean, mean - half, mean + half


def continuous_ridge(train_x, train_y, test_x, alpha=10.0):
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    mean, sd = train_x.mean(0), train_x.std(0)
    sd[sd < 1e-6] = 1
    a, b = (train_x - mean) / sd, (test_x - mean) / sd
    centered = train_y - train_y.mean()
    coef = a.T @ np.linalg.solve(a @ a.T + alpha * np.eye(len(a)), centered)
    return b @ coef + train_y.mean()


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else 0.0


def capture(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    started = time.perf_counter()
    raw, fixed, spec = load_registry(args.config.resolve(), args.model)
    seed = int(fixed["random_seed"])
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = resolved_source(spec)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True,
                                               trust_remote_code=bool(spec.get("trust_remote_code", False)))
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = spec["tokenizer"]["padding_side"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[spec["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, torch_dtype=dtype,
                                                 device_map=spec.get("device_map", "auto"), low_cpu_mem_usage=True,
                                                 trust_remote_code=bool(spec.get("trust_remote_code", False))).eval()
    preflight, rows, scorer = run_preflight(tokenizer, fixed, spec, model)
    adapter = make_adapter(model, spec); device = next(model.parameters()).device
    n, layers, hidden = len(rows), adapter.num_layers, adapter.hidden_size
    residual = np.zeros((n, layers, hidden), np.float16)
    attention = np.zeros_like(residual); mlp = np.zeros_like(residual)
    margin = np.zeros(n, np.float32); fold_ids = np.asarray(folds(rows, fixed), np.int16)
    for start in range(0, n, args.batch_size):
        indices = list(range(start, min(start + args.batch_size, n)))
        prompts = [build_prompt(tokenizer, fixed, rows[i]["premise"], rows[i]["hypothesis"]) for i in indices]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        positions = scorer.positions(encoded["attention_mask"]); ix = torch.arange(len(indices), device=device)
        cap_a, cap_m, handles = [None] * layers, [None] * layers, []
        for layer in range(layers):
            def ah(_module, _inputs, output, layer=layer):
                cap_a[layer] = (output[0] if isinstance(output, tuple) else output).detach()
            def mh(_module, _inputs, output, layer=layer): cap_m[layer] = output.detach()
            handles += [adapter.attention(layer).register_forward_hook(ah), adapter.mlp(layer).register_forward_hook(mh)]
        try:
            with torch.inference_mode():
                output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
        finally:
            for handle in handles: handle.remove()
        stop = start + len(indices)
        residual[start:stop] = torch.stack([output.hidden_states[layer + 1][ix, positions] for layer in range(layers)], 1).half().cpu().numpy()
        attention[start:stop] = torch.stack([cap_a[layer][ix, positions] for layer in range(layers)], 1).half().cpu().numpy()
        mlp[start:stop] = torch.stack([cap_m[layer][ix, positions] for layer in range(layers)], 1).half().cpu().numpy()
        margin[start:stop] = scorer.from_logits(output.logits, positions)["margin"].float().cpu().numpy()
        print(f"[CACHE] {stop}/{n}", flush=True)
    if not all(np.isfinite(x).all() for x in (residual, attention, mlp, margin)):
        raise RuntimeError("Nonfinite cache")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "full_depth_cache.npz"
    temp = cache_path.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez(handle, residual=residual, attention=attention, mlp=mlp, margin=margin,
                 fold=fold_ids, ids=np.asarray([r["id"] for r in rows]))
    temp.replace(cache_path)
    cd = np.asarray([r["group"][0] in "CD" for r in rows])
    cd_path = args.output_dir / "baseline_cd_reps.npz"; temp = cd_path.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez(handle, answer=residual[cd], ids=np.asarray([r["id"] for r in rows])[cd])
    temp.replace(cd_path)
    audit = {"status": "PASS", "model": args.model, "source": source, "revision": spec["revision"],
             "rows": n, "layers": layers, "hidden_size": hidden, "fold_counts": np.bincount(fold_ids).tolist(),
             "cache_shapes": {"residual": list(residual.shape), "attention": list(attention.shape),
                              "mlp": list(mlp.shape), "margin": list(margin.shape)},
             "all_finite": True, "hooks_removed": True, "scorer": scorer.audit(),
             "dataset_sha256": fixed["dataset_sha256"], "preflight": preflight["status"],
             "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
             "runtime_seconds": time.perf_counter() - started, "package_version": raw["package_version"]}
    atomic_json(args.output_dir / "cache_audit.json", audit)
    print(json.dumps(audit, indent=2))


def analyze(args):
    _, fixed, spec = load_registry(args.config.resolve(), args.model)
    rows = json.loads(Path(fixed["dataset_path"]).read_text())
    cache = np.load(args.output_dir / "full_depth_cache.npz", allow_pickle=False)
    arrays = {name: cache[name].astype(np.float32) for name in ("residual", "attention", "mlp")}
    margins, fold_ids = cache["margin"].astype(float), cache["fold"].astype(int)
    groups = np.asarray([r["group"][0] for r in rows]); labels = np.isin(groups, list("AC")).astype(int)
    populations = {"CD": np.isin(groups, list("CD")), "AB": np.isin(groups, list("AB")), "ALL": np.ones(len(rows), bool)}
    probe_rows = []
    for population, mask in populations.items():
        for component, values in arrays.items():
            for layer in range(values.shape[1]):
                for held in range(5):
                    train, test = mask & (fold_ids != held), mask & (fold_ids == held)
                    probe = fit_ridge_probe(values[train, layer], labels[train])
                    ba = probe_balanced_accuracy(probe, values[test, layer], labels[test])
                    probe_rows.append({"population": population, "component": component, "layer": layer + 1,
                                       "fold": held, "balanced_accuracy": ba, "train_n": int(train.sum()), "test_n": int(test.sum())})
    write_csv(args.output_dir / "telicity_probe_foldwise.csv", probe_rows)
    probe_summary = []
    for population in populations:
        for component in arrays:
            for layer in range(1, arrays[component].shape[1] + 1):
                values = [r["balanced_accuracy"] for r in probe_rows if r["population"] == population and r["component"] == component and r["layer"] == layer]
                mean, lo, hi = fold_ci(values)
                probe_summary.append({"population": population, "component": component, "layer": layer,
                                      "ba_mean": mean, "ba_std": float(np.std(values)), "ba_min_fold": min(values),
                                      "ba_max_fold": max(values), "ci95_low": lo, "ci95_high": hi,
                                      "above_chance_folds": sum(v > .5 for v in values)})
    write_csv(args.output_dir / "telicity_probe_summary.csv", probe_summary)
    cd_res = [r for r in probe_summary if r["population"] == "CD" and r["component"] == "residual"]
    stable = []
    for i in range(len(cd_res) - 2):
        if all(r["ba_mean"] >= .80 and r["ba_min_fold"] >= .70 for r in cd_res[i:i + 3]): stable.append(cd_res[i]["layer"])
    first_stable = stable[0] if stable else None; strongest = max(cd_res, key=lambda r: r["ba_mean"])

    decision_rows, decision_predictions = [], []
    for component, values in arrays.items():
        for layer in range(values.shape[1]):
            oof = np.zeros(len(rows), float); fold_r2 = []; fold_r = []
            for held in range(5):
                train, test = fold_ids != held, fold_ids == held
                pred = continuous_ridge(values[train, layer], margins[train], values[test, layer], 10.0)
                oof[test] = pred
                denom = float(np.sum((margins[test] - margins[train].mean()) ** 2))
                fold_r2.append(1 - float(np.sum((margins[test] - pred) ** 2)) / max(denom, 1e-12)); fold_r.append(correlation(margins[test], pred))
                for i, p in zip(np.flatnonzero(test), pred):
                    decision_predictions.append({"component": component, "layer": layer + 1, "fold": held,
                                                 "example_id": rows[i]["id"], "margin": margins[i], "predicted_margin": float(p)})
            denom = float(np.sum((margins - margins.mean()) ** 2))
            decision_rows.append({"component": component, "layer": layer + 1,
                                  "oof_r2": 1 - float(np.sum((margins - oof) ** 2)) / max(denom, 1e-12),
                                  "oof_pearson_r": correlation(margins, oof),
                                  "fold_r2_mean": float(np.mean(fold_r2)), "fold_r_mean": float(np.mean(fold_r))})
    write_csv(args.output_dir / "decision_layerwise_oof.csv", decision_rows)
    write_csv(args.output_dir / "decision_predictions.csv", decision_predictions)
    residual_dec = [r for r in decision_rows if r["component"] == "residual"]
    decision_stable = []
    for i in range(len(residual_dec) - 2):
        if all(r["oof_r2"] > 0 and r["oof_pearson_r"] >= .50 for r in residual_dec[i:i + 3]): decision_stable.append(residual_dec[i]["layer"])
    decision_first = decision_stable[0] if decision_stable else None
    decision_best = max(residual_dec, key=lambda r: r["oof_r2"])
    component_best = {component: max((r for r in decision_rows if r["component"] == component), key=lambda r: r["oof_r2"])
                      for component in arrays}
    summary = {"status": "complete", "model": args.model,
               "telicity": {"population": "C(telic) vs D(atelic), ambiguous condition controlled",
                              "stable_rule": "mean BA >= .80 and every fold BA >= .70 for 3 consecutive layers",
                              "first_stable_layer": first_stable, "strongest": strongest,
                              "component_peaks_CD": {c: max((r for r in probe_summary if r["population"] == "CD" and r["component"] == c), key=lambda r: r["ba_mean"]) for c in arrays}},
               "decision": {"target": "m=z_True-z_Unknown", "stable_rule": "OOF R2 > 0 and Pearson r >= .50 for 3 consecutive layers",
                            "first_stable_residual_layer": decision_first, "strongest_residual": decision_best,
                            "component_peaks": component_best},
               "scientific_unit": "five held-out verb folds", "leakage_control": "all probes fit on other four folds"}
    atomic_json(args.output_dir / "encoding_decision_summary.json", summary)
    audit = {"status": "PASS", "probe_rows": len(probe_rows), "expected_probe_rows": 3 * 3 * arrays["residual"].shape[1] * 5,
             "decision_rows": len(decision_rows), "expected_decision_rows": 3 * arrays["residual"].shape[1],
             "decision_prediction_rows": len(decision_predictions), "expected_decision_prediction_rows": 3 * arrays["residual"].shape[1] * len(rows),
             "all_finite": all(math.isfinite(float(r[k])) for r in probe_summary for k in ("ba_mean", "ci95_low", "ci95_high")) and all(math.isfinite(float(r[k])) for r in decision_rows for k in ("oof_r2", "oof_pearson_r")),
             "cache_sha256": sha256(args.output_dir / "full_depth_cache.npz"), "dataset_sha256": fixed["dataset_sha256"]}
    if not (audit["probe_rows"] == audit["expected_probe_rows"] and audit["decision_rows"] == audit["expected_decision_rows"] and audit["decision_prediction_rows"] == audit["expected_decision_prediction_rows"] and audit["all_finite"]): audit["status"] = "FAIL"
    atomic_json(args.output_dir / "encoding_decision_audit.json", audit)
    print(json.dumps({"summary": summary, "audit": audit}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cache", "analyze"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/multimodel_true_unknown_preference.yaml")
    parser.add_argument("--model", default="deepseek-7b-chat")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/deepseek-7b-telicity-rq")
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(); args.output_dir = args.output_dir.resolve()
    capture(args) if args.phase == "cache" else analyze(args)


if __name__ == "__main__": main()
