#!/usr/bin/env python3
"""Independent completeness audit and layerwise summary for DeepSeek Telicity RQs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/deepseek-7b-telicity-rq"
SWEEP = OUT / "removal_sweep"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    removal = read_csv(SWEEP / "removal_layer_sweep_fold_results.csv")
    recovery = read_csv(SWEEP / "recovery_curve_fold_results.csv")
    components = read_csv(SWEEP / "next_block_components_fold_results.csv")
    rm = {(int(r["removal_layer"]), int(r["heldout_fold"])): r for r in removal}
    rec = defaultdict(dict)
    for row in recovery:
        rec[(int(row["removal_layer"]), int(row["heldout_fold"]))][int(row["relative_layer"])] = row
    comp = {(int(r["removal_layer"]), int(r["heldout_fold"])): r for r in components}

    cache_failures = []
    cache_count = 0
    for layer in range(1, 28):
        if not (SWEEP / f"L{layer:02d}" / "complete.json").exists():
            cache_failures.append(f"L{layer:02d}: missing complete marker")
        for fold in range(5):
            path = SWEEP / f"L{layer:02d}" / f"fold{fold}_intervened_reps.npz"
            if not path.exists():
                cache_failures.append(f"L{layer:02d}/fold{fold}: missing cache")
                continue
            cache_count += 1
            with np.load(path, allow_pickle=False) as data:
                if data["recovery"].shape != (200, 4, 4096) or data["block"].shape != (200, 3, 4096):
                    cache_failures.append(f"{path}: bad shapes")
                if not np.isfinite(data["recovery"]).all() or not np.isfinite(data["block"]).all():
                    cache_failures.append(f"{path}: nonfinite values")

    layerwise = []
    valid_pairs = []
    for layer in range(1, 28):
        keys = [(layer, fold) for fold in range(5)]
        valid = [key for key in keys if int(float(rm[key]["removed_subspace_dimension"])) > 0]
        valid_pairs.extend(valid)
        baseline = np.mean([float(rm[k]["baseline_ba"]) for k in keys])
        post = np.mean([float(rm[k]["post_removal_ba"]) for k in keys])
        offsets = {}
        for offset in (1, 2, 3):
            values = [float(rec[k][offset]["delta_from_relative_0"]) for k in valid]
            offsets[offset] = {
                "mean": float(np.mean(values)) if values else None,
                "positive": sum(value > 0 for value in values),
            }
        first = next((offset for offset in (1, 2, 3)
                      if len(valid) >= 4 and offsets[offset]["positive"] >= 4
                      and offsets[offset]["mean"] >= .10), None)
        att = [float(comp[k]["delta_attention"]) for k in valid]
        mlp = [float(comp[k]["delta_mlp"]) for k in valid]
        layerwise.append({
            "removal_layer": layer,
            "valid_erasure_folds": len(valid),
            "baseline_ba_mean": baseline,
            "post_erasure_ba_mean": post,
            "erasure_drop_mean": baseline - post,
            "removed_dimension_mean": float(np.mean([float(rm[k]["removed_subspace_dimension"]) for k in keys])),
            "delta_Lplus1_mean_valid": offsets[1]["mean"],
            "positive_Lplus1_folds_valid": offsets[1]["positive"],
            "delta_Lplus2_mean_valid": offsets[2]["mean"],
            "positive_Lplus2_folds_valid": offsets[2]["positive"],
            "delta_Lplus3_mean_valid": offsets[3]["mean"],
            "positive_Lplus3_folds_valid": offsets[3]["positive"],
            "first_reemergence_offset": first,
            "attention_delta_mean_valid": float(np.mean(att)) if att else None,
            "positive_attention_folds_valid": sum(value > 0 for value in att),
            "mlp_delta_mean_valid": float(np.mean(mlp)) if mlp else None,
            "positive_mlp_folds_valid": sum(value > 0 for value in mlp),
            "immediate_dominant_component": ("attention" if att and np.mean(att) > np.mean(mlp) else
                                             "mlp" if mlp else "not_testable"),
        })
    path = SWEEP / "layerwise_mechanism.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(layerwise[0]))
        writer.writeheader(); writer.writerows(layerwise)

    relevant_numeric = []
    for row in removal:
        relevant_numeric.extend(float(row[key]) for key in (
            "baseline_ba", "post_removal_ba", "removed_subspace_dimension",
            "projection_residual_l2_mean", "projection_residual_maxabs"))
    for row in recovery:
        relevant_numeric.extend(float(row[key]) for key in ("balanced_accuracy", "delta_from_relative_0"))
    for row in components:
        relevant_numeric.extend(float(row[key]) for key in (
            "ba_block_input", "ba_after_attention", "ba_after_mlp", "delta_attention", "delta_mlp"))

    att = np.asarray([float(comp[k]["delta_attention"]) for k in valid_pairs])
    mlp = np.asarray([float(comp[k]["delta_mlp"]) for k in valid_pairs])
    valid_rec1 = np.asarray([float(rec[k][1]["delta_from_relative_0"]) for k in valid_pairs])
    audit = {
        "status": "PASS",
        "row_counts": {"removal": len(removal), "recovery": len(recovery), "components": len(components)},
        "expected_row_counts": {"removal": 135, "recovery": 540, "components": 135},
        "complete_layer_markers": sum((SWEEP / f"L{layer:02d}" / "complete.json").exists() for layer in range(1, 28)),
        "intervention_cache_count": cache_count,
        "cache_failures": cache_failures,
        "all_relevant_values_finite": all(math.isfinite(value) for value in relevant_numeric),
        "max_projection_residual_abs": max(float(r["projection_residual_maxabs"]) for r in removal),
        "valid_erasure_fold_pairs": len(valid_pairs),
        "valid_erasure_layers": [r["removal_layer"] for r in layerwise if r["valid_erasure_folds"] >= 4],
        "immediate_reemergence_positive_pairs": int((valid_rec1 > 0).sum()),
        "attention_positive_pairs": int((att > 0).sum()),
        "attention_greater_than_mlp_pairs": int((att > mlp).sum()),
        "baseline_cache_sha256": sha256(OUT / "baseline_cd_reps.npz"),
        "full_depth_cache_sha256": sha256(OUT / "full_depth_cache.npz"),
    }
    if (audit["row_counts"] != audit["expected_row_counts"] or audit["complete_layer_markers"] != 27
            or cache_count != 135 or cache_failures or not audit["all_relevant_values_finite"]):
        audit["status"] = "FAIL"
    atomic_json(SWEEP / "final_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
