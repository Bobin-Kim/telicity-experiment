#!/usr/bin/env python3
"""Small, non-scientific end-to-end engineering pilot for a registered model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_adapters import make_adapter
from src.model_registry import load_registry, resolved_source
from src.output_validation import atomic_json, write_csv
from src.pipeline_controller import norm, ridge
from src.preflight import build_prompt, folds, run_preflight


COMPONENTS = ("residual", "attention", "mlp")
CONDITIONS = ("decision_pos", "decision_neg", "random_pos", "random_neg")


def encode(tokenizer, fixed, rows, indices, device):
    prompts = [build_prompt(tokenizer, fixed, rows[i]["premise"], rows[i]["hypothesis"]) for i in indices]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) for k, v in encoded.items()}


def capture_selected(model, adapter, scorer, encoded, layers):
    positions = scorer.positions(encoded["attention_mask"])
    captured = {(component, layer): None for component in COMPONENTS for layer in layers}
    handles = []

    for layer in layers:
        handles.append(adapter.layers[layer].register_forward_pre_hook(
            lambda _module, inputs, layer=layer: captured.__setitem__(("residual", layer), inputs[0].detach())
        ))

        def post(component, layer):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                captured[component, layer] = value.detach()
            return hook

        handles.append(adapter.attention(layer).register_forward_hook(post("attention", layer)))
        handles.append(adapter.mlp(layer).register_forward_hook(post("mlp", layer)))

    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    ix = torch.arange(encoded["input_ids"].shape[0], device=positions.device)
    activations = {
        key: value[ix, positions].float().cpu().numpy()
        for key, value in captured.items()
    }
    shapes = {f"{component}_L{layer}": list(captured[component, layer].shape) for component in COMPONENTS for layer in layers}
    return output, positions, activations, shapes


def install_intervention(adapter, component, layer, positions, direction, scale, sign):
    ix = torch.arange(len(positions), device=positions.device)
    delta = torch.as_tensor(sign * scale * direction, device=positions.device, dtype=torch.float32)

    def patch(value):
        result = value.clone()
        result[ix, positions] = (result[ix, positions].float() + delta).to(result.dtype)
        return result

    if component == "residual":
        return adapter.layers[layer].register_forward_pre_hook(
            lambda _module, inputs: (patch(inputs[0]), *inputs[1:])
        )

    module = adapter.attention(layer) if component == "attention" else adapter.mlp(layer)

    def output_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (patch(output[0]), *output[1:])
        return patch(output)

    return module.register_forward_hook(output_hook)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--train-per-group", type=int, default=3)
    parser.add_argument("--test-per-group", type=int, default=2)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    start = time.perf_counter()
    raw, fixed, spec = load_registry(args.config.resolve(), args.model)
    seed = int(fixed["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = resolved_source(spec)
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, use_fast=bool(spec["tokenizer"]["use_fast"]),
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = spec["tokenizer"]["padding_side"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[spec["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, torch_dtype=dtype, device_map=spec.get("device_map", "auto"),
        low_cpu_mem_usage=True, trust_remote_code=bool(spec.get("trust_remote_code", False)),
    ).eval()
    preflight, rows, scorer = run_preflight(tokenizer, fixed, spec, model)
    adapter = make_adapter(model, spec)
    device = next(model.parameters()).device
    fold_ids = np.asarray(folds(rows, fixed))
    groups = np.asarray([row["group"][0] for row in rows])

    train_indices, test_indices = [], []
    for group in "ABCD":
        train_indices.extend(np.flatnonzero((groups == group) & (fold_ids != args.heldout_fold))[:args.train_per_group].tolist())
        test_indices.extend(np.flatnonzero((groups == group) & (fold_ids == args.heldout_fold))[:args.test_per_group].tolist())
    selected = train_indices + test_indices
    if len(train_indices) != 4 * args.train_per_group or len(test_indices) != 4 * args.test_per_group:
        raise RuntimeError("Insufficient balanced pilot rows")

    encoded = encode(tokenizer, fixed, rows, selected, device)
    output, positions, activations, activation_shapes = capture_selected(model, adapter, scorer, encoded, args.layers)
    scores = scorer.from_logits(output.logits, positions)
    behavior = []
    for j, index in enumerate(selected):
        behavior.append({
            "pilot_role": "train" if index in train_indices else "heldout",
            "example_index": index, "example_id": rows[index]["id"], "group": groups[index],
            "fold": int(fold_ids[index]), "true_logit": float(scores["true_logit"][j]),
            "unknown_logit": float(scores["unknown_logit"][j]), "margin": float(scores["margin"][j]),
        })

    train_local = np.arange(len(train_indices))
    target = np.asarray([behavior[i]["margin"] for i in train_local], dtype=float)
    axes = {}
    for component in COMPONENTS:
        for layer in args.layers:
            x = activations[component, layer]
            mu, sd, coef, _ = ridge(x[train_local], target, float(fixed["preference_basis"]["ridge_alpha"]))
            axis = norm(coef / sd)
            scale = float(np.std((x[train_local] - mu) @ axis))
            if not np.isfinite(axis).all() or not math.isfinite(scale) or scale <= 0:
                raise RuntimeError(f"Invalid pilot axis for {component} L{layer}")
            rng = np.random.default_rng(seed + layer * 10 + COMPONENTS.index(component))
            random_axis = rng.normal(size=len(axis))
            random_axis -= axis * np.dot(axis, random_axis)
            random_axis = norm(random_axis)
            axes[component, layer] = (axis, random_axis, scale)

    test_encoded = encode(tokenizer, fixed, rows, test_indices, device)
    test_positions = scorer.positions(test_encoded["attention_mask"])
    with torch.inference_mode():
        clean_output = model(**test_encoded, use_cache=False, return_dict=True)
    clean_margin = scorer.from_logits(clean_output.logits, test_positions)["margin"].float().cpu().numpy()
    intervention_rows = []
    active_handle_count = 0
    for component in COMPONENTS:
        for layer in args.layers:
            axis, random_axis, scale = axes[component, layer]
            for condition in CONDITIONS:
                direction = random_axis if condition.startswith("random") else axis
                sign = 1 if condition.endswith("pos") else -1
                handle = install_intervention(adapter, component, layer, test_positions, direction, scale, sign)
                try:
                    with torch.inference_mode():
                        changed = model(**test_encoded, use_cache=False, return_dict=True)
                finally:
                    handle.remove()
                changed_margin = scorer.from_logits(changed.logits, test_positions)["margin"].float().cpu().numpy()
                for j, index in enumerate(test_indices):
                    intervention_rows.append({
                        "example_index": index, "example_id": rows[index]["id"], "group": groups[index],
                        "heldout_fold": args.heldout_fold, "layer": layer, "component": component,
                        "condition": condition, "baseline_margin": float(clean_margin[j]),
                        "margin": float(changed_margin[j]), "delta_margin": float(changed_margin[j] - clean_margin[j]),
                        "steering_scale": scale,
                    })
                module = adapter.layers[layer] if component == "residual" else (adapter.attention(layer) if component == "attention" else adapter.mlp(layer))
                active_handle_count += len(module._forward_hooks) + len(module._forward_pre_hooks)

    aggregate_rows = []
    for component in COMPONENTS:
        for layer in args.layers:
            means = {}
            for condition in CONDITIONS:
                values = [row["delta_margin"] for row in intervention_rows if row["component"] == component and row["layer"] == layer and row["condition"] == condition]
                means[condition] = float(np.mean(values))
                aggregate_rows.append({"component": component, "layer": layer, "condition": condition, "n": len(values), "mean_delta_margin": means[condition]})
            aggregate_rows.append({
                "component": component, "layer": layer, "condition": "control_adjusted_contrast", "n": len(test_indices),
                "mean_delta_margin": (means["decision_pos"] - means["decision_neg"]) - (means["random_pos"] - means["random_neg"]),
            })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pilot_behavior.csv", behavior)
    write_csv(output_dir / "pilot_interventions.csv", intervention_rows)
    write_csv(output_dir / "pilot_aggregation.csv", aggregate_rows)
    expected = len(COMPONENTS) * len(args.layers) * len(CONDITIONS) * len(test_indices)
    numeric_finite = all(math.isfinite(float(row[key])) for row in intervention_rows for key in ("baseline_margin", "margin", "delta_margin", "steering_scale"))
    unique = len({(row["example_id"], row["layer"], row["component"], row["condition"]) for row in intervention_rows})
    audit = {
        "status": "PASS" if len(intervention_rows) == expected and unique == expected and numeric_finite and active_handle_count == 0 else "FAIL",
        "engineering_only": True, "scientific_interpretation_allowed": False, "model": args.model,
        "model_source": source, "revision": spec["revision"], "shared_scorer": scorer.audit(),
        "heldout_fold": args.heldout_fold, "layers": args.layers, "components": list(COMPONENTS),
        "conditions": list(CONDITIONS), "train_rows": len(train_indices), "heldout_rows": len(test_indices),
        "train_group_counts": dict(Counter(groups[train_indices])), "heldout_group_counts": dict(Counter(groups[test_indices])),
        "train_test_disjoint": not bool(set(train_indices) & set(test_indices)),
        "activation_shapes": activation_shapes, "intervention_rows": len(intervention_rows),
        "expected_intervention_rows": expected, "unique_intervention_rows": unique,
        "all_numeric_finite": numeric_finite, "active_hooks_after_removal": active_handle_count,
        "actual_forward": True, "random_controls": True, "aggregation_complete": len(aggregate_rows) == len(COMPONENTS) * len(args.layers) * 5,
        "full_rq_started": False, "binary_artifact_invoked": False,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_seconds": float(time.perf_counter() - start),
        "package_version": raw["package_version"], "preflight_status": preflight["status"],
    }
    atomic_json(output_dir / "pilot_audit.json", audit)
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
