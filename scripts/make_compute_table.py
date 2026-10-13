"""Computational-effort profile for the seven detectors (mirrors the Derzsi
template's t_proc / M_peak category). Reports trainable parameters, inference
latency per consumer, and peak inference memory. Inference only -- light on GPU.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from phased.models import build_arm, build_a34_generic

W, D, B = 147, 7, 256
DET = [("grud", "GRU-D"), ("hybridattn", "Self-attention"), ("dkgt", "DKGT-Net"),
       ("wdcnn", "Wide-and-deep CNN"), ("msdm_net", "MSDM"), ("horln", "HORLN"),
       ("a34_generic", "MAMRD")]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(name):
    if name == "a34_generic":
        return build_a34_generic(weeks=W, days=D)
    return build_arm(name, weeks=W, days=D)


def profile(model):
    model = model.to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(B, W, D, device=dev)
    x[x > 1.6] = float("nan")                       # realistic missingness
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(5):                          # warmup
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 50
        for _ in range(iters):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    per_cons_us = dt / (iters * B) * 1e6            # microseconds per consumer
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else float("nan")
    return n_par, per_cons_us, peak_mb


def main():
    print(f"[device] {dev}")
    print(f"{'detector':<18}{'params':>12}{'us/consumer':>14}{'peak MB':>10}")
    for key, lab in DET:
        try:
            n, us, mb = profile(build(key))
            kp = n / 1e3
            print(f"{lab:<18}{kp:>10.1f}K{us:>14.2f}{mb:>10.1f}")
        except Exception as e:
            print(f"{lab:<18}  ERROR: {e}")
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
