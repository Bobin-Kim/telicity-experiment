"""Self-contained, model-adapted telicity re-emergence workflow.

The locked source-path estimand is copied from the completed Qwen/Mistral
Experiment-B protocol: Residual, Context-V, and Answer-Q.  Context-K and
Answer-Gate are an explicitly named extension and are never folded into the
locked three-path normalization or historical equivalence claim.

Layer numbers in this module are scientific, one-based hidden-state indices:
removing at L means editing the output of decoder block L and examining block
L+1.  Adapter indices remain zero based.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .output_validation import atomic_json, write_csv
from .preflight import build_prompt


EPS = 1e-8
LOCKED_PATHS = ("residual", "context_value", "answer_query")
EXTENDED_PATHS = ("residual", "context_value", "context_key", "answer_query", "answer_gate")
PROJECTION_PATHS = {
    "context_value": ("value", "context"),
    "context_key": ("key", "context"),
    "answer_query": ("query", "answer"),
    "answer_gate": ("gate", "answer"),
}


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def balanced_accuracy(y, pred):
    y, pred = np.asarray(y, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        raise ValueError("Balanced accuracy requires both telicity classes")
    return float(((pred[pos] == 1).mean() + (pred[neg] == 0).mean()) / 2)


def fit_probe(x, y, reg=1e-2):
    """Canonical class-balanced dual-ridge telicity probe."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.int64)
    signed = np.where(y == 1, 1.0, -1.0)
    mean = x.mean(0, keepdims=True)
    centered = x - mean
    counts = np.bincount(y, minlength=2).astype(np.float64)
    counts[counts == 0] = 1
    weights = np.where(y == 1, .5 / counts[1], .5 / counts[0])
    root = np.sqrt(weights)
    xw, yw = centered * root[:, None], signed * root
    gram = xw @ xw.T + reg * np.eye(len(x), dtype=np.float64)
    return {"mean": mean, "weight": xw.T @ np.linalg.solve(gram, yw)}


def probe_scores(probe, x):
    return (np.asarray(x, np.float64) - probe["mean"]) @ probe["weight"]


def probe_ba(probe, x, y):
    return balanced_accuracy(y, probe_scores(probe, x) >= 0)


def project(x, basis):
    return x if not basis.size else x - (x @ basis) @ basis.T


def fit_basis_sequence(x, y, max_directions):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.int64)
    basis = np.zeros((x.shape[1], 0), np.float64)
    for _ in range(max_directions):
        direction = np.asarray(fit_probe(project(x, basis), y)["weight"]).reshape(-1)
        if basis.shape[1]:
            direction -= basis @ (basis.T @ direction)
        size = float(np.linalg.norm(direction))
        if size <= EPS:
            break
        direction /= size
        if basis.shape[1]:
            direction -= basis @ (basis.T @ direction)
            size = float(np.linalg.norm(direction))
            if size <= EPS:
                break
            direction /= size
        basis = np.concatenate((basis, direction[:, None]), axis=1)
    return basis


def segment_profiles(features, min_size=3, max_k=6):
    """Locked BIC contiguous segmentation used for depth-wise regimes."""
    features = np.asarray(features, np.float64)
    n, dim = features.shape
    if n < min_size:
        return {"k": 1, "bic": float("nan"), "sse": 0.0,
                "index_bounds": [(0, n)], "candidates": []}
    prefix = np.vstack((np.zeros((1, dim)), np.cumsum(features, axis=0)))
    square = np.r_[0., np.cumsum(np.sum(features * features, axis=1))]

    def error(start, stop):
        total = prefix[stop] - prefix[start]
        return float(square[stop] - square[start] - total @ total / (stop - start))

    candidates = []
    for count in range(1, min(max_k, n // min_size) + 1):
        dp = np.full((count + 1, n + 1), np.inf)
        previous = np.full((count + 1, n + 1), -1, dtype=int)
        dp[0, 0] = 0
        for segments in range(1, count + 1):
            for stop in range(segments * min_size, n + 1):
                for start in range((segments - 1) * min_size, stop - min_size + 1):
                    value = dp[segments - 1, start] + error(start, stop)
                    if value < dp[segments, stop]:
                        dp[segments, stop], previous[segments, stop] = value, start
        bounds, stop = [], n
        for segments in range(count, 0, -1):
            start = int(previous[segments, stop]); bounds.append((start, stop)); stop = start
        bounds.reverse(); sse = float(dp[count, n]); parameters = count * dim + count - 1
        bic = n * dim * math.log(max(sse / (n * dim), 1e-12)) + parameters * math.log(n * dim)
        candidates.append({"k": count, "bic": float(bic), "sse": sse, "index_bounds": bounds})
    return {**min(candidates, key=lambda row: row["bic"]), "candidates": candidates}


def calibrate_dimension(x, y, rows, target_ba, max_directions, seed, folds=4):
    """Inner verb-disjoint rank choice, identical to the locked moving sweep."""
    import random

    rng = random.Random(seed)
    by_class = defaultdict(list)
    for row, label in zip(rows, y, strict=True):
        if row["verb"] not in by_class[int(label)]:
            by_class[int(label)].append(row["verb"])
    assignment = {}
    for label in (0, 1):
        verbs = by_class[label]
        rng.shuffle(verbs)
        assignment.update({verb: i % folds for i, verb in enumerate(verbs)})
    by_k = defaultdict(list)
    for held in range(folds):
        train = np.asarray([assignment[r["verb"]] != held for r in rows])
        valid = ~train
        basis = fit_basis_sequence(x[train], y[train], max_directions)
        for k in range(basis.shape[1] + 1):
            p = fit_probe(project(x[train], basis[:, :k]), y[train])
            by_k[k].append(probe_ba(p, project(x[valid], basis[:, :k]), y[valid]))
    curve = []
    for k, values in sorted(by_k.items()):
        if len(values) == folds:
            curve.append({"removed_subspace_dimension": k,
                          "inner_cv_ba_mean": float(np.mean(values)),
                          "inner_cv_ba_std": float(np.std(values)),
                          "target_gap": float(abs(np.mean(values) - target_ba))})
    if not curve:
        raise RuntimeError("No complete inner-fold erasure calibration")
    selected = min(curve, key=lambda row: (row["target_gap"], row["removed_subspace_dimension"]))
    full = fit_basis_sequence(x, y, max_directions)
    rank = min(int(selected["removed_subspace_dimension"]), full.shape[1])
    return full[:, :rank], curve, selected


def _span_tokens(offsets, start, stop):
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < stop and b > start]


def source_positions(tokenizer, fixed, row):
    """Exact mutually-exclusive linguistic context and answer coordinates."""
    prompt = build_prompt(tokenizer, fixed, row["premise"], row["hypothesis"])
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    if "offset_mapping" not in encoded:
        raise ValueError("Re-emergence analysis requires a fast tokenizer with offset mappings")
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    premise, hypothesis = row["premise"], row["hypothesis"]
    p0 = prompt.find(premise)
    h0 = prompt.find(hypothesis, p0 + len(premise))
    if p0 < 0 or h0 < 0:
        raise ValueError(f"Prompt source span not found for {row['id']}")
    premise_tokens = _span_tokens(offsets, p0, p0 + len(premise))
    hypothesis_tokens = _span_tokens(offsets, h0, h0 + len(hypothesis))
    match = re.search(r"\b(?P<aux>was|were|is|are|am|been|being)\s+(?P<verb>[A-Za-z][A-Za-z-]*)",
                      premise, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Progressive clause not found for {row['id']}")
    aux_a, aux_b = match.span("aux")
    verb_a, verb_b = match.span("verb")
    ends = [x for x in (premise.find(",", verb_b), premise.lower().find(" but ", verb_b)) if x >= 0]
    clause_end = min(ends) if ends else len(premise)
    spans = {"subject": (0, aux_a), "aspect_marker": (aux_a, aux_b),
             "event_verb": (verb_a, verb_b), "object_endpoint": (verb_b, clause_end)}
    raw = {name: _span_tokens(offsets, p0 + a, p0 + b) for name, (a, b) in spans.items()}
    used, assigned = set(), {}
    for name in ("aspect_marker", "event_verb", "object_endpoint", "subject"):
        assigned[name] = [i for i in raw[name] if i not in used]
        used.update(assigned[name])
    context = sorted(set([i for i in premise_tokens if i not in used] + hypothesis_tokens))
    result = {**assigned, "context": context, "answer": [len(encoded["input_ids"]) - 1]}
    if not context:
        raise ValueError(f"Empty context positions for {row['id']}")
    return result


class ReemergenceWorkflow:
    def __init__(self, controller):
        self.c = controller
        self.model, self.tok, self.a = controller.model, controller.tok, controller.a
        self.fixed, self.root = controller.f, controller.out / "reemergence"
        self.cfg = self.fixed["reemergence_analysis"]
        self.indices = [i for i, row in enumerate(controller.rows) if row["group"][0] in "CD"]
        self.rows = [controller.rows[i] for i in self.indices]
        self.labels = np.asarray([1 if row["group"][0] == "C" else 0 for row in self.rows], np.int64)
        self.fold = np.asarray([controller.fold[i] for i in self.indices], np.int64)
        if len(self.rows) != 200 or np.bincount(self.labels, minlength=2).tolist() != [100, 100]:
            raise ValueError("Re-emergence workflow requires the complete canonical 100-C/100-D set; --max-items is not valid")
        self.device = controller.device
        self.batch = controller.stage_batch("reemergence")
        self.layers = self._layers()

    def _layers(self):
        valid = list(range(1, self.a.num_layers - 2))
        requested = self.c.requested_layers
        if requested == list(range(self.a.num_layers)):
            return valid
        invalid = sorted(set(requested) - set(valid))
        if invalid:
            raise ValueError(f"Re-emergence removal layers must be 1..{self.a.num_layers - 3}; got {invalid}")
        return requested

    def _prompts(self, local_indices):
        return [build_prompt(self.tok, self.fixed, self.rows[i]["premise"], self.rows[i]["hypothesis"])
                for i in local_indices]

    def _encode(self, local_indices, max_length=None):
        kwargs = {"return_tensors": "pt", "padding": "max_length" if max_length else True,
                  "add_special_tokens": False, "truncation": False}
        if max_length:
            kwargs["max_length"] = max_length
        encoded = self.tok(self._prompts(local_indices), **kwargs)
        return {k: v.to(self.device) for k, v in encoded.items()}

    @property
    def cache_path(self):
        return self.root / "clean_sequence_fp16.npy"

    def cache(self):
        self.root.mkdir(parents=True, exist_ok=True)
        prompts = self._prompts(range(len(self.rows)))
        lengths = np.asarray([len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts], np.int64)
        max_len, hidden = int(lengths.max()), self.a.hidden_size
        positions = [source_positions(self.tok, self.fixed, row) for row in self.rows]
        temporary = self.cache_path.with_suffix(".npy.tmp")
        states = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                           shape=(self.a.num_layers, len(self.rows), max_len, hidden))
        try:
            for start in range(0, len(self.rows), self.batch):
                ids = list(range(start, min(start + self.batch, len(self.rows))))
                encoded = self._encode(ids, max_len)
                with torch.inference_mode():
                    output = self.model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
                for layer in range(self.a.num_layers):
                    states[layer, start:start + len(ids)] = output.hidden_states[layer + 1].detach().half().cpu().numpy()
                states.flush()
                print(f"[reemergence-cache] {start + len(ids)}/{len(self.rows)}", flush=True)
            del states
            temporary.replace(self.cache_path)
        finally:
            try:
                del states
            except UnboundLocalError:
                pass
            temporary.unlink(missing_ok=True)
        np.savez_compressed(self.root / "clean_sequence_meta.npz", lengths=lengths,
                            folds=self.fold, labels=self.labels,
                            ids=np.asarray([r["id"] for r in self.rows]))
        atomic_json(self.root / "source_positions.json",
                    {row["id"]: pos for row, pos in zip(self.rows, positions, strict=True)})
        self.c.marker("reemergence_cache", {"rows": len(self.rows), "layers": self.a.num_layers,
                                             "dtype": "float16", "scope": "C/D telicity concept"})

    def _assets(self):
        meta = np.load(self.root / "clean_sequence_meta.npz", allow_pickle=False)
        if meta["ids"].tolist() != [r["id"] for r in self.rows]:
            raise ValueError("Re-emergence cache row identity mismatch")
        max_len = int(meta["lengths"].max())
        states = np.load(self.cache_path, mmap_mode="r")
        expected = (self.a.num_layers, len(self.rows), max_len, self.a.hidden_size)
        if states.shape != expected or not np.isfinite(np.asarray(states[:, :, :1, :1])).all():
            raise ValueError(f"Invalid re-emergence cache shape {states.shape}; expected {expected}")
        lookup = json.loads((self.root / "source_positions.json").read_text())
        return states, meta["lengths"].astype(int), [lookup[r["id"]] for r in self.rows]

    @staticmethod
    def _answer_values(states, lengths, layer):
        return np.asarray([states[layer - 1, i, lengths[i] - 1] for i in range(len(lengths))], np.float64)

    def _erased_forward(self, layer, basis):
        """Canonical real forward: erase block-L answer output, inspect L..L+3."""
        recovery = np.zeros((len(self.rows), 4, self.a.hidden_size), np.float16)
        components = np.zeros((len(self.rows), 3, self.a.hidden_size), np.float16)
        basis_t = torch.as_tensor(basis, device=self.device, dtype=next(self.model.parameters()).dtype)
        removal, next_block = self.a.layers[layer - 1], self.a.layers[layer]
        for start in range(0, len(self.rows), self.batch):
            ids = list(range(start, min(start + self.batch, len(self.rows))))
            encoded = self._encode(ids)
            answer = self.c.s.positions(encoded["attention_mask"])
            captures = {}

            def erase(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                patched = value.clone(); ix = torch.arange(len(ids), device=value.device)
                vectors = patched[ix, answer]
                if basis_t.numel():
                    local = basis_t.to(device=value.device, dtype=value.dtype); vectors = vectors - (vectors @ local) @ local.T
                patched[ix, answer] = vectors
                return (patched, *output[1:]) if isinstance(output, tuple) else patched

            def block_pre(_module, inputs):
                ix = torch.arange(len(ids), device=inputs[0].device)
                captures["input"] = inputs[0][ix, answer].detach()

            def component(name):
                def hook(_module, _inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    ix = torch.arange(len(ids), device=value.device)
                    captures[name] = value[ix, answer].detach()
                return hook

            handles = [removal.register_forward_hook(erase), next_block.register_forward_pre_hook(block_pre),
                       self.a.attention(layer).register_forward_hook(component("attention")),
                       self.a.mlp(layer).register_forward_hook(component("mlp"))]
            try:
                with torch.inference_mode():
                    output = self.model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            ix = torch.arange(len(ids), device=self.device)
            for offset in range(4):
                recovery[start:start + len(ids), offset] = output.hidden_states[layer + offset][ix, answer].half().cpu().numpy()
            after_attention = captures["input"] + captures["attention"]
            after_mlp = after_attention + captures["mlp"]
            for point, values in enumerate((captures["input"], after_attention, after_mlp)):
                components[start:start + len(ids), point] = values.half().cpu().numpy()
        return recovery, components

    def sweep(self):
        states, lengths, _ = self._assets()
        target = float(self.cfg["removal_target_ba"])
        max_directions = int(self.cfg["max_inlp_directions"])
        for layer in self.layers:
            layer_dir = self.root / "removal_sweep" / f"L{layer:02d}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            baseline = self._answer_values(states, lengths, layer)
            removal_rows, recovery_rows, component_rows, calibration_rows = [], [], [], []
            for held in self.c.requested_folds:
                train, test = self.fold != held, self.fold == held
                basis, curve, selected = calibrate_dimension(
                    baseline[train], self.labels[train], [r for r, use in zip(self.rows, train) if use],
                    target, max_directions, int(self.fixed["random_seed"]) + 10000 * layer + held)
                for row in curve:
                    calibration_rows.append({"removal_layer": layer, "heldout_fold": held, **row})
                artifact = layer_dir / f"fold{held}_intervened_reps.npz"
                if artifact.exists():
                    saved = np.load(artifact, allow_pickle=False)
                    recovery, components = saved["recovery"], saved["block"]
                else:
                    recovery, components = self._erased_forward(layer, basis)
                    np.savez_compressed(artifact, recovery=recovery, block=components)
                base_ba = probe_ba(fit_probe(baseline[train], self.labels[train]), baseline[test], self.labels[test])
                post_probe = fit_probe(recovery[train, 0], self.labels[train])
                post_ba = probe_ba(post_probe, recovery[test, 0], self.labels[test])
                removal_rows.append({"removal_layer": layer, "heldout_fold": held,
                                     "baseline_ba": base_ba, "calibration_target_ba": target,
                                     "inner_cv_selected_ba": selected["inner_cv_ba_mean"],
                                     "post_removal_ba": post_ba,
                                     "removed_subspace_dimension": basis.shape[1],
                                     "train_n": int(train.sum()), "test_n": int(test.sum())})
                for offset in range(4):
                    ba = probe_ba(fit_probe(recovery[train, offset], self.labels[train]),
                                  recovery[test, offset], self.labels[test])
                    recovery_rows.append({"removal_layer": layer, "heldout_fold": held,
                                          "absolute_layer": layer + offset, "relative_layer": offset,
                                          "balanced_accuracy": ba, "delta_from_relative_0": ba - post_ba})
                point_ba = [probe_ba(fit_probe(components[train, p], self.labels[train]),
                                     components[test, p], self.labels[test]) for p in range(3)]
                component_rows.append({"removal_layer": layer, "analyzed_block": layer + 1,
                                       "heldout_fold": held, "ba_block_input": point_ba[0],
                                       "ba_after_attention": point_ba[1], "ba_after_mlp": point_ba[2],
                                       "delta_attention": point_ba[1] - point_ba[0],
                                       "delta_mlp": point_ba[2] - point_ba[1]})
                write_csv(layer_dir / "removal_metrics.csv", removal_rows)
                write_csv(layer_dir / "recovery_curve.csv", recovery_rows)
                write_csv(layer_dir / "next_block_components.csv", component_rows)
                write_csv(layer_dir / "calibration_curve.csv", calibration_rows)
            atomic_json(layer_dir / "complete.json", {"status": "complete", "layer": layer,
                                                        "folds": self.c.requested_folds})
        self.c.marker("reemergence_sweep", {"layers": self.layers, "folds": self.c.requested_folds,
                                             "target_ba": target, "protocol": "moving INLP erasure"})

    @staticmethod
    def _matched_donors(receivers, labels, positions, lengths, opposite):
        donors = []
        for receiver in receivers:
            target = 1 - labels[receiver] if opposite else labels[receiver]
            candidates = [i for i in receivers if labels[i] == target and (opposite or i != receiver)]
            if not candidates:
                raise RuntimeError(f"No matched donor for receiver {receiver}")
            donors.append(min(candidates, key=lambda i: (abs(lengths[i] - lengths[receiver]),
                                                          abs(len(positions[i]["context"]) - len(positions[receiver]["context"])), i)))
        return donors

    def _projection_capture(self, layer, basis, local_indices, paths):
        """Capture erased donor Q/K/V/G projections and block-L+1 answer output."""
        encoded = self._encode(local_indices)
        answer = self.c.s.positions(encoded["attention_mask"])
        basis_t = torch.as_tensor(basis, device=self.device, dtype=next(self.model.parameters()).dtype)
        captured = {}

        def erase(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            patched = value.clone(); ix = torch.arange(len(local_indices), device=value.device)
            vectors = patched[ix, answer]
            if basis_t.numel():
                    local = basis_t.to(device=value.device, dtype=value.dtype); vectors = vectors - (vectors @ local) @ local.T
            patched[ix, answer] = vectors
            return (patched, *output[1:]) if isinstance(output, tuple) else patched

        def capture(name):
            def hook(_module, _inputs, output): captured[name] = output.detach()
            return hook

        def block_output(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            ix = torch.arange(len(local_indices), device=value.device)
            captured["block"] = value[ix, answer].detach()

        handles = [self.a.layers[layer - 1].register_forward_hook(erase),
                   self.a.layers[layer].register_forward_hook(block_output)]
        for path in paths:
            accessor, _ = PROJECTION_PATHS[path]
            handles.append(getattr(self.a, accessor)(layer).register_forward_hook(capture(path)))
        try:
            with torch.inference_mode(): self.model(**encoded, use_cache=False, return_dict=True)
        finally:
            for handle in handles: handle.remove()
        return captured, answer.detach().cpu().numpy()

    def _patched_block(self, layer, basis, receivers, donors, path, positions, restore=False):
        donor, donor_answer = self._projection_capture(layer, basis, donors, (path,))
        encoded = self._encode(receivers)
        answer = self.c.s.positions(encoded["attention_mask"])
        basis_t = torch.as_tensor(basis, device=self.device, dtype=next(self.model.parameters()).dtype)
        result, change = {}, np.zeros(len(receivers), np.float32)

        def erase(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            patched = value.clone(); ix = torch.arange(len(receivers), device=value.device)
            vectors = patched[ix, answer]
            if basis_t.numel():
                local = basis_t.to(device=value.device, dtype=value.dtype); vectors = vectors - (vectors @ local) @ local.T
            patched[ix, answer] = vectors
            return (patched, *output[1:]) if isinstance(output, tuple) else patched

        def projection_patch(_module, _inputs, output):
            patched = output.clone(); source = donor[path].to(output.device)
            is_fused_gate = path == "answer_gate" and self.a.spec["modules"].get("gate_output_slice") == "first_half"
            width = output.shape[-1] // 2 if is_fused_gate else output.shape[-1]
            for j, (receiver, donor_id) in enumerate(zip(receivers, donors, strict=True)):
                if PROJECTION_PATHS[path][1] == "answer":
                    before = patched[j, answer[j], :width].clone()
                    patched[j, answer[j], :width] = source[j, donor_answer[j], :width]
                    change[j] = float(torch.linalg.vector_norm(patched[j, answer[j], :width].float() - before.float()))
                else:
                    rp, dp = positions[receiver]["context"], positions[donor_id]["context"]
                    r = torch.as_tensor(rp, device=output.device); d = torch.as_tensor(dp, device=output.device)
                    before = patched[j].index_select(0, r).clone()
                    replacement = source[j].index_select(0, d)
                    if not restore:
                        replacement = replacement.mean(0, keepdim=True).expand(len(rp), -1)
                    patched[j, r] = replacement
                    change[j] = float(torch.linalg.vector_norm(patched[j].index_select(0, r).float() - before.float(), dim=-1).mean())
            return patched

        def block_output(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            ix = torch.arange(len(receivers), device=value.device)
            result["block"] = value[ix, answer].detach()

        module = getattr(self.a, PROJECTION_PATHS[path][0])(layer)
        handles = [self.a.layers[layer - 1].register_forward_hook(erase),
                   module.register_forward_hook(projection_patch),
                   self.a.layers[layer].register_forward_hook(block_output)]
        try:
            with torch.inference_mode(): self.model(**encoded, use_cache=False, return_dict=True)
        finally:
            for handle in handles: handle.remove()
        return result["block"].float().cpu().numpy(), change

    def pathways(self, extended=False):
        states, lengths, positions = self._assets()
        paths = EXTENDED_PATHS if extended else LOCKED_PATHS
        projection_paths = [p for p in paths if p != "residual"]
        max_directions = int(self.cfg["max_inlp_directions"])
        all_rows = []
        for layer in self.layers:
            baseline = self._answer_values(states, lengths, layer)
            layer_dir = self.root / "removal_sweep" / f"L{layer:02d}"
            for held in self.c.requested_folds:
                train, test_mask = self.fold != held, self.fold == held
                receivers = np.flatnonzero(test_mask).tolist()
                metric = next(r for r in _read_csv(layer_dir / "removal_metrics.csv") if int(r["heldout_fold"]) == held)
                rank = int(metric["removed_subspace_dimension"])
                basis = fit_basis_sequence(baseline[train], self.labels[train], max_directions)[:, :rank]
                saved = np.load(layer_dir / f"fold{held}_intervened_reps.npz", allow_pickle=False)["recovery"]
                post, recovered = saved[:, 0].astype(np.float64), saved[:, 1].astype(np.float64)
                probe = fit_probe(recovered[train], self.labels[train])
                normal = probe_scores(probe, recovered)
                score_sd = float(np.std(normal[train])) or 1.0
                causal = self._matched_donors(receivers, self.labels, positions, lengths, True)
                null = self._matched_donors(receivers, self.labels, positions, lengths, False)
                unpatched, _ = self._projection_capture(layer, basis, receivers, ())
                unpatched_scores = probe_scores(probe, unpatched["block"].float().cpu().numpy())
                patched_scores, restore_errors, changes = {}, {}, {}
                for path in projection_paths:
                    for control, donors in (("causal", causal), ("null", null)):
                        values, delta = self._patched_block(layer, basis, receivers, donors, path, positions)
                        patched_scores[(path, control)] = normal[receivers] + probe_scores(probe, values) - unpatched_scores
                        changes[(path, control)] = delta
                    restored, _ = self._patched_block(layer, basis, receivers, receivers, path, positions, restore=True)
                    restore_errors[path] = float(np.max(np.abs(probe_scores(probe, restored) - unpatched_scores)))
                for control, donors in (("causal", causal), ("null", null)):
                    residual = recovered[receivers] + post[donors] - post[receivers]
                    patched_scores[("residual", control)] = probe_scores(probe, residual)
                y = self.labels[receivers]; signed = np.where(y == 1, 1., -1.)
                post_ba = probe_ba(fit_probe(post[train], self.labels[train]), post[receivers], y)
                recovered_ba = balanced_accuracy(y, normal[receivers] >= 0)
                for path in paths:
                    cs, ns = patched_scores[(path, "causal")], patched_scores[(path, "null")]
                    causal_ba, null_ba = balanced_accuracy(y, cs >= 0), balanced_accuracy(y, ns >= 0)
                    cz = float(np.mean(signed * (normal[receivers] - cs)) / score_sd)
                    nz = float(np.mean(signed * (normal[receivers] - ns)) / score_sd)
                    all_rows.append({"protocol": "extended_qkv_gate_v1" if extended else "locked_experiment_b",
                                     "erasure_layer": layer, "recovery_layer": layer + 1, "fold": held,
                                     "pathway": path, "baseline_telicity_ba": float(metric["baseline_ba"]),
                                     "post_erasure_ba": post_ba, "recovered_ba": recovered_ba,
                                     "total_recovery_ba": recovered_ba - post_ba,
                                     "causal_patched_ba": causal_ba, "same_label_null_patched_ba": null_ba,
                                     "causal_recovery_contribution_ba": null_ba - causal_ba,
                                     "causal_score_reduction_z": cz, "same_label_null_score_reduction_z": nz,
                                     "specific_causal_contribution_z": cz - nz,
                                     "restore_max_abs_probe_score_error": restore_errors.get(path, 0.0),
                                     "selected_erasure_rank": rank, "train_n": int(train.sum()),
                                     "test_n": len(receivers), "causal_donor_scope": "opposite-label held-out length-matched",
                                     "null_donor_scope": "same-label held-out length-matched"})
            target = self.root / ("extended_layer_fold_pathway_results.csv" if extended else "locked_layer_fold_pathway_results.csv")
            write_csv(target, all_rows)
        stage = "source_pathway_extended" if extended else "source_pathway"
        self.c.marker(stage, {"layers": self.layers, "folds": self.c.requested_folds,
                              "pathways": list(paths), "protocol": "extension" if extended else "locked",
                              "locked_result_renormalized": False})

    def aggregate(self):
        outputs = {}
        extension_marker = self.c.out / "source_pathway_extended.complete.json"
        extension_valid = False
        if extension_marker.exists():
            try:
                marker = json.loads(extension_marker.read_text())
                extension_valid = (marker.get("status") == "complete"
                                   and marker.get("run_fingerprint") == self.c.fp
                                   and (self.root / "extended_layer_fold_pathway_results.csv").exists())
            except (json.JSONDecodeError, OSError):
                extension_valid = False
        protocols = [("locked", LOCKED_PATHS, "locked_layer_fold_pathway_results.csv")]
        if extension_valid:
            protocols.append(("extended", EXTENDED_PATHS, "extended_layer_fold_pathway_results.csv"))
        for name, paths, filename in protocols:
            raw = _read_csv(self.root / filename)
            expected = len(self.layers) * len(self.c.requested_folds) * len(paths)
            if len(raw) != expected:
                raise RuntimeError(f"Incomplete {name} pathway grid: {len(raw)}/{expected}")
            rows, analysis_layers = [], []
            minimum_folds = int(self.cfg["minimum_verified_folds"])
            minimum_drop = float(self.cfg["verified_erasure_min_drop_ba"])
            for layer in self.layers:
                residual_rows = [r for r in raw if int(r["erasure_layer"]) == layer and r["pathway"] == "residual"]
                valid_folds = [int(r["fold"]) for r in residual_rows
                               if int(float(r["selected_erasure_rank"])) > 0
                               and float(r["baseline_telicity_ba"]) - float(r["post_erasure_ba"]) >= minimum_drop - 1e-9]
                if len(valid_folds) < minimum_folds:
                    rows.append({"protocol": name, "layer": layer, "valid_folds": len(valid_folds),
                                 "dominant_pathway": "not_identifiable"})
                    continue
                effects = {}
                for path in paths:
                    values = [float(r["specific_causal_contribution_z"]) for r in raw
                              if int(r["erasure_layer"]) == layer and r["pathway"] == path
                              and int(r["fold"]) in valid_folds]
                    effects[path] = float(np.mean(values))
                positive = {p: max(v, 0.) for p, v in effects.items()}
                denom = sum(positive.values())
                if denom <= EPS:
                    positive, denom = {p: abs(v) for p, v in effects.items()}, sum(abs(v) for v in effects.values())
                shares = {p: positive[p] / max(denom, EPS) for p in paths}
                rows.append({"protocol": name, "layer": layer, "valid_folds": len(valid_folds),
                             **{f"{p}_effect_z": effects[p] for p in paths},
                             **{f"{p}_share": shares[p] for p in paths},
                             "dominant_pathway": max(paths, key=lambda p: shares[p])})
                analysis_layers.append(layer)
            write_csv(self.root / f"{name}_layerwise_pathway_summary.csv", rows)
            identifiable = [row for row in rows if row["dominant_pathway"] != "not_identifiable"]
            if not identifiable:
                raise RuntimeError(f"No layer has {minimum_folds} verified erasure folds for {name}")
            features = np.asarray([[row[f"{path}_share"] for path in paths] for row in identifiable], np.float64)
            segmentation = segment_profiles(features, int(self.cfg["minimum_regime_layers"]))
            regimes = []
            for start, stop in segmentation["index_bounds"]:
                shares = {path: float(features[start:stop, i].mean()) for i, path in enumerate(paths)}
                if name == "locked" and shares["context_value"] >= .25 and shares["residual"] >= .25:
                    mechanism = "residual+context_value"
                else:
                    mechanism = max(paths, key=lambda path: shares[path])
                regimes.append({"start_layer": analysis_layers[start], "end_layer": analysis_layers[stop - 1],
                                "mechanism": mechanism, "normalized_pathway_shares": shares})
            rng = np.random.default_rng(int(self.fixed["random_seed"]) + 991)
            folds = list(self.c.requested_folds); boot = int(self.cfg["bootstrap_samples"])
            k_counts, boundary_counts = defaultdict(int), defaultdict(int)
            indexed = {(int(r["erasure_layer"]), int(r["fold"]), r["pathway"]):
                       float(r["specific_causal_contribution_z"]) for r in raw}
            for _ in range(boot):
                sampled = rng.choice(folds, size=len(folds), replace=True)
                sample_features = []
                for layer in analysis_layers:
                    valid_here = [int(r["fold"]) for r in raw
                                  if int(r["erasure_layer"]) == layer and r["pathway"] == "residual"
                                  and int(float(r["selected_erasure_rank"])) > 0
                                  and float(r["baseline_telicity_ba"]) - float(r["post_erasure_ba"]) >= minimum_drop - 1e-9]
                    selected_valid = [int(fold) for fold in sampled if int(fold) in valid_here] or valid_here
                    effects = {path: float(np.mean([indexed[(layer, fold, path)] for fold in selected_valid]))
                               for path in paths}
                    positive = {path: max(value, 0.) for path, value in effects.items()}
                    denominator = sum(positive.values())
                    if denominator <= EPS:
                        positive = {path: abs(value) for path, value in effects.items()}
                        denominator = sum(positive.values())
                    sample_features.append([positive[path] / max(denominator, EPS) for path in paths])
                selected = segment_profiles(np.asarray(sample_features), int(self.cfg["minimum_regime_layers"]))
                k_counts[selected["k"]] += 1
                for _, stop in selected["index_bounds"][:-1]:
                    boundary_counts[analysis_layers[stop]] += 1
            regime_report = {"protocol": name, "pathways": list(paths),
                             "selected_num_regimes": segmentation["k"],
                             "transition_layers": [analysis_layers[stop] for _, stop in segmentation["index_bounds"][:-1]],
                             "regimes": regimes, "segmentation_candidates": segmentation["candidates"],
                             "bootstrap_samples": boot,
                             "bootstrap_k_support": {str(k): value / boot for k, value in sorted(k_counts.items())},
                             "bootstrap_boundary_support": {str(k): value / boot for k, value in sorted(boundary_counts.items())}}
            atomic_json(self.root / f"{name}_regime_summary.json", regime_report)
            outputs[name] = {"rows": len(raw), "layers": len(rows), "pathways": list(paths),
                             "selected_num_regimes": segmentation["k"]}
        atomic_json(self.root / "PROTOCOL_BOUNDARY.json", {
            "locked": "Residual/Context-V/Answer-Q; historical Experiment-B reproduction target",
            "extended": "Residual/Context-V/Context-K/Answer-Q/Answer-Gate; new analysis, no historical equivalence claim",
            "extended_execution_status": "complete" if extension_valid else "optional_unexecuted",
            "locked_result_renormalized_by_extension": False,
            "scoring_rule_changed": False,
            "preference_scorer": self.c.s.audit(),
        })
        self.c.marker("reemergence_aggregate", {"outputs": outputs, "protocols_kept_separate": True})


def pathway_contract():
    """Small public contract used by preflight/tests and handoff audits."""
    return {"locked": list(LOCKED_PATHS), "extended": list(EXTENDED_PATHS),
            "locked_is_subset": set(LOCKED_PATHS) < set(EXTENDED_PATHS),
            "renormalize_locked_with_extensions": False}
