"""Shared local utilities; no external experiment imports."""
import json
from pathlib import Path
import numpy as np
import torch
from .data import Context, EpisodeGenerator
from .local_model import local_prediction


def slice_episode(e, start, end):
    return {k: Context(v.inputs[start:end], v.outputs[start:end]) if isinstance(v, Context)
            else v[start:end] if isinstance(v, torch.Tensor) or (isinstance(v, np.ndarray) and k != 'time')
            else v for k, v in e.items()}


def numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def pair_error(state, truth):
    pred = np.concatenate([numpy(v) for v in state], -1)
    target = np.concatenate([numpy(v) for v in truth], -1)
    return np.linalg.norm((pred-target).reshape(len(pred), -1), axis=1)/np.maximum(1e-8, np.linalg.norm(target.reshape(len(pred), -1), axis=1))


def residual(state, previous):
    # RMS of each temporal trajectory, summed across subsystems; absolute units.
    return sum(np.sqrt(np.mean((numpy(a)-numpy(b))**2, axis=(1, 2))) for a, b in zip(state, previous))


def stats(values):
    v = np.asarray(values, dtype=float)
    return {'n': len(v), 'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v)>1 else 0.,
            'median': float(np.median(v)), 'p90': float(np.quantile(v, .9)),
            'sem': float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.}



def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
