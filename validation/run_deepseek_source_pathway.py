#!/usr/bin/env python3
"""DeepSeek adapter for the locked Qwen/Mistral Experiment-B source analysis.

Only architecture/cache plumbing differs.  Donors, folds, interventions,
controls, effects, thresholds, segmentation, and aggregation are delegated to
the original shared implementation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE))

from imperfective_paradox_repro.telicity_intervention import run_experiment_b_depthwise_telicity_recovery as exp
from imperfective_paradox_repro.telicity_intervention.run_telicity_reemergence_source_analysis import source_positions


MODEL = Path("/data/sl_datasets/models/deepseek-ai/deepseek-llm-7b-chat")
BASE = ROOT / "results/deepseek-7b-telicity-rq"
OUTPUT = BASE / "source_pathway"
LAYERS = tuple(range(1, 28))
CACHE_NAME = "clean_sequence_L01_L27_fp16_bits.npy"


# The checkpoint is a fixed, trusted local artifact.  Scope the compatibility
# override to this process; do not alter the shared loader or checkpoint.
import transformers.modeling_utils as _modeling_utils
_modeling_utils.check_torch_load_is_safe = lambda: None


def load_model_and_tokenizer(model_path: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    return model, tokenizer


def _gpu_probe_parts(x, y):
    x = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    y = torch.as_tensor(y, dtype=torch.int64, device="cuda")
    signed = torch.where(y == 1, 1.0, -1.0)
    mean = x.mean(0, keepdim=True)
    centered = x - mean
    counts = torch.bincount(y, minlength=2).to(torch.float64).clamp_min(1)
    weights = torch.where(y == 1, .5 / counts[1], .5 / counts[0])
    root = weights.sqrt(); xw = centered * root[:, None]; yw = signed * root
    gram = xw @ xw.T + 1e-2 * torch.eye(len(x), dtype=torch.float64, device="cuda")
    return mean, xw.T @ torch.linalg.solve(gram, yw)


def fit_basis_sequence(x, y, max_directions):
    """Float64 GPU implementation of the locked ordered INLP basis."""
    x = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    y = torch.as_tensor(y, dtype=torch.int64, device="cuda")
    basis = torch.empty((x.shape[1], 0), dtype=torch.float64, device="cuda")
    for _ in range(max_directions):
        projected = x if basis.shape[1] == 0 else x - (x @ basis) @ basis.T
        _mean, direction = _gpu_probe_parts(projected, y)
        if basis.shape[1]:
            direction -= basis @ (basis.T @ direction)
        size = torch.linalg.vector_norm(direction)
        if float(size) <= exp.EPS:
            break
        direction /= size
        if basis.shape[1]:
            direction -= basis @ (basis.T @ direction)
            size = torch.linalg.vector_norm(direction)
            if float(size) <= exp.EPS:
                break
            direction /= size
        basis = torch.cat((basis, direction[:, None]), 1)
    return basis.cpu().numpy()


def prepare_sequence_cache(model, tokenizer, rows, output_dir: Path, batch_size: int):
    cache_path = output_dir / CACHE_NAME
    meta_path = output_dir / "clean_sequence_meta.npz"
    ids_path = output_dir / "clean_sequence_rows.json"
    positions_path = output_dir / "source_positions.json"
    if all(path.exists() for path in (cache_path, meta_path, ids_path, positions_path)):
        return
    tokenizer.padding_side = "right"
    prompts = [exp.prompt_for_row(tokenizer, row) for row in rows]
    encoded = tokenizer(prompts, return_tensors="np", padding=True, add_special_tokens=False)
    masks = encoded["attention_mask"].astype(np.int64)
    lengths = masks.sum(1).astype(np.int64)
    seq_len, hidden = int(masks.shape[1]), int(model.config.hidden_size)
    tmp = exp.atomic_path(cache_path)
    mmap = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.int16,
        shape=(len(LAYERS), len(rows), seq_len, hidden),
    )
    device = next(model.parameters()).device
    try:
        for start in range(0, len(rows), batch_size):
            prompts_batch = prompts[start:start + batch_size]
            inputs = tokenizer(
                prompts_batch, return_tensors="pt", padding="max_length",
                max_length=seq_len, add_special_tokens=False,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                result = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            for offset, layer in enumerate(LAYERS):
                bits = result.hidden_states[layer].detach().to(torch.float16).cpu().view(torch.int16).numpy()
                mmap[offset, start:start + len(prompts_batch)] = bits
            mmap.flush(); print(f"[prepare] {start + len(prompts_batch)}/{len(rows)}", flush=True)
        del mmap; tmp.replace(cache_path)
    finally:
        try: del mmap
        except UnboundLocalError: pass
        tmp.unlink(missing_ok=True)
    meta_tmp = exp.atomic_path(meta_path)
    with meta_tmp.open("wb") as handle:
        np.savez_compressed(handle, layers=np.asarray(LAYERS), attention_mask=masks,
                            lengths=lengths, cache_dtype=np.asarray("float16"))
    meta_tmp.replace(meta_path)
    exp.write_json(ids_path, [row["id"] for row in rows])
    exp.write_json(positions_path, {row["id"]: source_positions(tokenizer, row) for row in rows})


class SequenceCache:
    def __init__(self, output_dir: Path, rows):
        self.states = np.load(output_dir / CACHE_NAME, mmap_mode="r")
        meta = np.load(output_dir / "clean_sequence_meta.npz", allow_pickle=False)
        self.layers = [int(value) for value in meta["layers"]]
        self.mask = meta["attention_mask"].astype(np.int64)
        self.lengths = meta["lengths"].astype(np.int64)
        stored = json.loads((output_dir / "clean_sequence_rows.json").read_text())
        if stored != [row["id"] for row in rows]:
            raise ValueError("Sequence-cache row order mismatch")
        expected = (len(LAYERS), len(rows), self.mask.shape[1], 4096)
        if self.states.shape != expected:
            raise ValueError(f"Unexpected sequence cache {self.states.shape}; expected {expected}")

    def bits(self, indices, layer):
        return np.asarray(self.states[self.layers.index(layer), indices])


def torch_fp16_from_bits(bits, device):
    return torch.from_numpy(np.array(bits, copy=True)).to(device=device).view(torch.float16)


def load_positions(rows, cache):
    lookup = json.loads((OUTPUT / "source_positions.json").read_text())
    positions = [lookup[row["id"]] for row in rows]
    for index, (row, position) in enumerate(zip(rows, positions, strict=True)):
        if int(position["answer"][0]) != int(cache.lengths[index] - 1):
            raise ValueError(f"Answer position mismatch: {row['id']}")
        if not position["context"]:
            raise ValueError(f"Empty context positions: {row['id']}")
    return positions


# Patch only model/cache adapter surfaces and experiment locations.
exp.MODEL = MODEL
exp.A4 = BASE
exp.A28 = BASE / "removal_sweep"
exp.OUT = OUTPUT
exp.LAYERS = LAYERS
exp.CACHE_LAYERS = LAYERS
exp.MODEL_NUM_LAYERS = 30
exp.load_model_and_tokenizer = load_model_and_tokenizer
exp.fit_basis_sequence = fit_basis_sequence
exp.prepare_sequence_cache = prepare_sequence_cache
exp.SequenceCache = SequenceCache
exp.torch_bf16_from_bits = torch_fp16_from_bits
exp.load_positions = load_positions


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    exp.main()
    config = OUTPUT / "experiment_config.json"
    if config.exists():
        value = json.loads(config.read_text())
        value["sequence_cache_dtype"] = "lossless raw float16 bits"
        value["adapter_note"] = "DeepSeek Llama architecture; scientific protocol unchanged"
        exp.write_json(config, value)
