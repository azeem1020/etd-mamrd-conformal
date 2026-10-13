"""Reproduction sanity-check (post-hoc, NO GPU).

For each baseline we already store, per CV cell, the per-consumer test labels and
scores (val_labels_json / val_scores_json in run_n_clean.py). Those scores come
from one trained model under the SHARED honest protocol. Here we hold the model
and its scores fixed and flip ONLY the *evaluation protocol*:

  (A) NATIVE   : the full test partition at the corpus prevalence (~8.5%).
  (B) BALANCED : a 50:50 test subset (all positives + an equal random sample of
                 negatives), the 'balanced evaluation set' several SGCC headlines
                 report on.

Because the scores are identical, any change in the headline metric is caused by
the evaluation protocol alone -- not by the implementation. This isolates the
'balanced evaluation set' inflation lever with zero degrees of freedom.

ROC-AUC is rank-based and prevalence-invariant, so it is (up to sampling noise)
unchanged by balancing -- which is exactly why a balanced-set ROC headline near
0.98 cannot be explained by balancing and instead implicates the ranking-level
levers (preprocessing / adversarial augmentation) handled by the GPU runs.

Metrics per protocol: AP (full-prevalence average precision), ROC-AUC,
F1 at the best threshold tuned ON THAT protocol's set, and accuracy at 0.5.
Balanced figures are averaged over R seeded negative draws for stability.
"""
from __future__ import annotations
import csv, json, glob
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

csv.field_size_limit(10 ** 7)
B = (RESULTS_DIR / "baselines")
DET = [("wdcnn", "WDCNN"), ("horln", "HORLN"), ("hybridattn", "Self-attn."),
       ("grud", "GRU-D"), ("msdm_net", "MSDM"), ("dkgt", "DKGT-Net")]
R = 20                                       # balanced negative draws per cell


def best_f1(y, s):
    """Max achievable F1 on this set (optimistic, the way an inflated headline is
    produced). Vectorised: scanning the top-k cut, F1 = 2*TP/(k + P) since
    FP+FN = (k-TP)+(P-TP), so the F1 denominator collapses to k + P."""
    P = int(y.sum())
    if P == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[order])                       # TP after predicting top-k
    k = np.arange(1, len(y) + 1)
    f1 = 2.0 * tp / (k + P)
    return float(f1.max())


def cell_metrics(y, s, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    # (A) native
    nat = dict(ap=average_precision_score(y, s), roc=roc_auc_score(y, s),
               f1=best_f1(y, s), acc=float(((s >= 0.5).astype(int) == y).mean()),
               prev=float(y.mean()))
    # (B) balanced: all positives + equal random negatives, averaged over R draws
    bap = bro = bf1 = bacc = 0.0
    for _ in range(R):
        sel = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
        yb, sb = y[sel], s[sel]
        bap += average_precision_score(yb, sb)
        bro += roc_auc_score(yb, sb)
        bf1 += best_f1(yb, sb)
        bacc += float(((sb >= 0.5).astype(int) == yb).mean())
    bal = dict(ap=bap / R, roc=bro / R, f1=bf1 / R, acc=bacc / R, prev=0.5)
    return nat, bal


def load_cell(path):
    row = next(csv.DictReader(open(path)))
    y = np.array(json.loads(row["val_labels_json"]), dtype=int)
    s = np.array(json.loads(row["val_scores_json"]), dtype=float)
    return y, s


def main():
    rng = np.random.default_rng(0)
    print(f"{'detector':<12} | {'NATIVE (prev~0.085)':^34} | {'BALANCED (prev=0.5)':^34}")
    print(f"{'':<12} | {'AP':>7}{'ROC':>9}{'F1':>9}{'Acc':>9} | {'AP':>7}{'ROC':>9}{'F1':>9}{'Acc':>9}")
    print("-" * 100)
    table = {}
    for key, lab in DET:
        cells = sorted(glob.glob(str(B / f"{key}_[0-9]_[0-9].csv")))
        if not cells:
            print(f"{lab:<12} | (no stored cells)")
            continue
        N = {k: [] for k in ("ap", "roc", "f1", "acc")}
        Bm = {k: [] for k in ("ap", "roc", "f1", "acc")}
        for c in cells:
            y, s = load_cell(c)
            if not (0 < y.sum() < len(y)):
                continue
            nat, bal = cell_metrics(y, s, rng)
            for k in N:
                N[k].append(nat[k]); Bm[k].append(bal[k])
        agg = lambda d, k: (np.mean(d[k]), np.std(d[k]))
        m = {f"nat_{k}": agg(N, k) for k in N}
        m.update({f"bal_{k}": agg(Bm, k) for k in Bm})
        table[key] = m
        print(f"{lab:<12} | {m['nat_ap'][0]:>7.3f}{m['nat_roc'][0]:>9.3f}"
              f"{m['nat_f1'][0]:>9.3f}{m['nat_acc'][0]:>9.3f} | "
              f"{m['bal_ap'][0]:>7.3f}{m['bal_roc'][0]:>9.3f}"
              f"{m['bal_f1'][0]:>9.3f}{m['bal_acc'][0]:>9.3f}")
    print("-" * 100)
    print("note: ROC is ~invariant native->balanced (rank-based); F1/Acc/AP inflate "
          "purely from the eval-set prevalence flip, same scores.")
    out = B.parent / "reproduction_balanced.json"
    json.dump({k: {kk: list(vv) for kk, vv in v.items()} for k, v in table.items()},
              open(out, "w"), indent=2)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
