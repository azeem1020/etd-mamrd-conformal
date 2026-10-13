"""Persistence + reuse: save/load trained MSDM and run inference on new datasets.

Saves a torch checkpoint (state_dict + the exact init config, so the architecture
rebuilds deterministically) and an .npz of test predictions. `.pt` (torch-native)
is used rather than raw pickle — it is the robust, version-portable format for
model weights; a convenience `.pkl` of the whole model is also written on request.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .model import MSDM


def save_model(model: MSDM, config: dict, path, also_pickle: bool = True):
    """Save weights + init config. `config` must hold the MSDM(...) kwargs."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
    if also_pickle:
        import pickle
        with open(path.with_suffix(".pkl"), "wb") as fh:
            pickle.dump(model.cpu().eval(), fh)
    return str(path)


def load_model(path, device="cpu") -> MSDM:
    """Rebuild MSDM from its saved config and load weights. Ready for inference
    on ANY dataset preprocessed to (N, C_in, T) the same way (min–max + mask)."""
    ckpt = torch.load(path, map_location=device)
    model = MSDM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: MSDM, X: np.ndarray, device="cpu", batch=512) -> np.ndarray:
    """Return sigmoid theft scores for preprocessed X (N, C_in, T)."""
    dev = torch.device(device); model = model.to(dev).eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float().to(dev)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out)


def save_predictions(path, *, test_idx, labels, scores, extra: dict | None = None):
    """Save predictions as .npz for downstream reuse / analysis."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(test_idx=np.asarray(test_idx), labels=np.asarray(labels),
                   scores=np.asarray(scores))
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)
    return str(path)


__all__ = ["save_model", "load_model", "predict", "save_predictions"]
