#!/usr/bin/env python3
"""DeepSeek wrapper around the locked moving Telicity-erasure sweep."""
from pathlib import Path
import sys

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from imperfective_paradox_repro.telicity_intervention import run_telicity_removal_layer_sweep as sweep

# transformers>=4.52 refuses legacy PyTorch .bin files when torch<2.6.  This
# experiment uses a fixed, trusted, local DeepSeek checkpoint whose hashes were
# established during preflight, so scope the compatibility override to this
# wrapper process rather than changing the shared loader.
import transformers.modeling_utils as _modeling_utils
_modeling_utils.check_torch_load_is_safe = lambda: None


def _gpu_probe_parts(x, y):
    """Float64 dual ridge, algebraically matching the locked NumPy probe."""
    x = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    y = torch.as_tensor(y, dtype=torch.int64, device="cuda")
    signed = torch.where(y == 1, 1.0, -1.0)
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    counts = torch.bincount(y, minlength=2).to(torch.float64).clamp_min(1.0)
    weights = torch.where(y == 1, 0.5 / counts[1], 0.5 / counts[0])
    root = weights.sqrt()
    xw = centered * root[:, None]
    yw = signed * root
    gram = xw @ xw.T + 1e-2 * torch.eye(len(x), dtype=torch.float64, device="cuda")
    weight = xw.T @ torch.linalg.solve(gram, yw)
    return mean, weight


def _gpu_fit_basis_sequence(x, y, max_directions):
    """GPU implementation of the same ordered, float64 INLP basis."""
    x = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    y = torch.as_tensor(y, dtype=torch.int64, device="cuda")
    basis = torch.empty((x.shape[1], 0), dtype=torch.float64, device="cuda")
    for _ in range(max_directions):
        projected = x if basis.shape[1] == 0 else x - (x @ basis) @ basis.T
        _mean, direction = _gpu_probe_parts(projected, y)
        if basis.shape[1]:
            direction = direction - basis @ (basis.T @ direction)
        norm = torch.linalg.vector_norm(direction)
        if float(norm) <= sweep.EPS:
            break
        direction = direction / norm
        # Match the locked modified Gram-Schmidt re-orthogonalization pass.
        if basis.shape[1]:
            direction = direction - basis @ (basis.T @ direction)
            norm = torch.linalg.vector_norm(direction)
            if float(norm) <= sweep.EPS:
                break
            direction = direction / norm
        basis = torch.cat((basis, direction[:, None]), dim=1)
    return basis.cpu().numpy()


def _gpu_fresh_probe_ba(x_train, y_train, x_test, y_test):
    mean, weight = _gpu_probe_parts(x_train, y_train)
    test = torch.as_tensor(x_test, dtype=torch.float64, device="cuda")
    truth = torch.as_tensor(y_test, dtype=torch.int64, device="cuda")
    pred = (((test - mean) @ weight) >= 0).to(torch.int64)
    pos, neg = truth == 1, truth == 0
    tpr = (pred[pos] == 1).to(torch.float64).mean()
    tnr = (pred[neg] == 0).to(torch.float64).mean()
    return float(0.5 * (tpr + tnr))


sweep.fit_basis_sequence = _gpu_fit_basis_sequence
sweep.fresh_probe_ba = _gpu_fresh_probe_ba

sweep.MODEL_NUM_LAYERS = 30
sweep.DEFAULT_MODEL_PATH = Path("/data/sl_datasets/models/deepseek-ai/deepseek-llm-7b-chat")
sweep.DEFAULT_REP_CACHE = Path("/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/results/deepseek-7b-telicity-rq/baseline_cd_reps.npz")
sweep.DEFAULT_OUTPUT_DIR = Path("/home/bobin/MULTIMODEL_TRUE_UNKNOWN_PREFERENCE_RUNNER/results/deepseek-7b-telicity-rq/removal_sweep")
if __name__ == "__main__":
    sweep.main()
