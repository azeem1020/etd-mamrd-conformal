#!/usr/bin/env python3
"""Comment 3: Client-Disjoint 5-Fold ECL Evaluation (GroupKFold by Client ID).

Partitions the 370 ECL clients into 5 strictly client-disjoint folds via
GroupKFold(n_splits=5) on client_id, splitting held-out clients per fold into
client-disjoint early-stopping (~35% clients) and test (~65% clients) sets.
Asserts 0% client overlap between train, early-stopping, and test on every fold.

Trains all 6 ECL detectors (a34_generic, horln, msdm, wdcnn, hybridattn, grud)
across Folds 0-4 x Seeds 0-2 = 15 cells per detector (90 cells total).
Reports overall AP, ROC-AUC, Recall@1%/5%FPR with BCa 95% CIs, paired contrasts,
and per-attack Recall@5%FPR (FDI1..FDI6) on unseen clients.
"""
from __future__ import annotations
import sys, os, csv, gc, json, time, argparse, subprocess
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

csv.field_size_limit(10**7)

THIS = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS.parent
_REPO_DIR = SCRIPTS_DIR.parent
for p in (str(SCRIPTS_DIR), str(_REPO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from repro_config import _ROOT, ECL_PATH, scrub_paths

ORIG_REPO = _ROOT
R3_RES = _ROOT / "results" / "r3_revision"
OUT = R3_RES / "C3_ecl_client_disjoint"
CELLS_DIR = OUT / "cells"
OUT.mkdir(parents=True, exist_ok=True)
CELLS_DIR.mkdir(parents=True, exist_ok=True)

from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from zkmix_lab.metrics import recall_at_fpr
from phased.models import build_arm
from run_local_grid import _inference_batch
from msdm.fdi import ATTACKS
from msdm.train import train_msdm, evaluate as evaluate_msdm

N_DIST = NormalDist()
FREQ, PPD = "30min", 48
WIN = 7 * PPD
CAP = 25000
ARMS = ["a34_generic", "horln", "msdm", "wdcnn", "hybridattn", "grud"]
DISPLAY = {
    "a34_generic": "MAMRD", "horln": "HORLN", "msdm": "MSDM",
    "wdcnn": "WDCNN", "hybridattn": "Self-Attn", "grud": "GRU-D",
}


def bca_ci(data, B: int = 2000, alpha: float = 0.05, seed: int = 12345) -> tuple[float, float]:
    x = np.asarray([v for v in data if np.isfinite(v)], dtype=float)
    n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        m = float(np.mean(x)) if n else float("nan")
        return m, m
    th = float(x.mean())
    rng = np.random.default_rng(seed)
    boot = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
    p_lt = min(max(float(np.mean(boot < th)), 1e-6), 1.0 - 1e-6)
    z0 = N_DIST.inv_cdf(p_lt)
    jk = np.array([np.delete(x, i).mean() for i in range(n)])
    jb = jk.mean()
    den = 6.0 * (np.sum((jb - jk) ** 2) ** 1.5)
    a = float(np.sum((jb - jk) ** 3) / den) if den else 0.0
    def q_adj(q):
        z = N_DIST.inv_cdf(q)
        return N_DIST.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))
    return float(np.quantile(boot, q_adj(alpha / 2.0))), float(np.quantile(boot, q_adj(1.0 - alpha / 2.0)))


def gpu_temp() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return 0


def cool_if_hot(limit: int = 82, target: int = 68) -> None:
    if gpu_temp() <= limit:
        return
    for _ in range(30):
        if gpu_temp() <= target:
            break
        time.sleep(10)


def prep_windows(W: np.ndarray) -> np.ndarray:
    m = np.nanmean(W, 1, keepdims=True)
    s = np.nanstd(W, 1, keepdims=True)
    W = np.where(W > m + 3.0 * s, m + 3.0 * s, W)
    lo = np.nanmin(W, 1, keepdims=True)
    hi = np.nanmax(W, 1, keepdims=True)
    d = hi - lo
    d[d == 0] = 1.0
    return ((W - lo) / d).astype(np.float32)


def build_ecl_dataset_with_clients():
    ecl_path = ECL_PATH
    if not ecl_path.exists():
        raise FileNotFoundError(
            f"Missing ECL raw dataset at {ecl_path}. Obtain LD2011_2014.txt per data/README.md "
            "or set MAMRD_ECL."
        )
    df = pd.read_csv(ecl_path, sep=";", decimal=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    res = df.resample(FREQ).mean().values.T.astype(np.float64)
    n, T = res.shape
    nw = T // WIN
    wins, cids = [], []
    for i in range(n):
        for w in range(nw):
            seg = res[i, w * WIN : (w + 1) * WIN]
            if np.nansum(np.abs(seg)) > 1e-6:
                wins.append(seg)
                cids.append(i)
    W = np.array(wins, dtype=np.float64)
    cids = np.array(cids, dtype=np.int64)

    # Identical RNG draw sequence to run_ecl_trust.py
    rng = np.random.default_rng(0)
    if len(W) > CAP:
        sel = rng.choice(len(W), CAP, False)
        W = W[sel]
        cids = cids[sel]

    frng = np.random.default_rng(0)
    y = np.zeros(len(W), dtype=int)
    atk_type = np.zeros(len(W), dtype=int)
    Wo = W.copy()
    for i in frng.choice(len(W), int(0.10 * len(W)), False):
        k = int(frng.integers(6))
        Wo[i] = ATTACKS[k](W[i], frng)
        y[i] = 1
        atk_type[i] = k + 1
    mag = np.nanmean(Wo, axis=1)
    mag = np.where(np.isfinite(mag), mag, 0.0)
    Wp = prep_windows(Wo)
    return Wp, y, atk_type, cids, mag


def get_client_disjoint_splits(y: np.ndarray, cids: np.ndarray, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_folds)
    splits = []
    for f, (tr_idx, held_idx) in enumerate(gkf.split(y, y, groups=cids)):
        held_clients = np.unique(cids[held_idx])
        rng_c = np.random.default_rng(100 + f)
        perm = rng_c.permutation(held_clients)
        n_es_c = max(1, int(round(0.35 * len(perm))))
        es_clients = set(perm[:n_es_c].tolist())
        te_clients = set(perm[n_es_c:].tolist())
        es_idx = held_idx[np.isin(cids[held_idx], list(es_clients))]
        te_idx = held_idx[np.isin(cids[held_idx], list(te_clients))]

        c_tr, c_es, c_te = set(cids[tr_idx].tolist()), set(cids[es_idx].tolist()), set(cids[te_idx].tolist())
        assert c_tr.isdisjoint(c_es) and c_tr.isdisjoint(c_te) and c_es.isdisjoint(c_te), f"Client leakage on fold {f}"
        splits.append((tr_idx, es_idx, te_idx))
    return splits


def verify_math_invariants() -> bool:
    rng = np.random.default_rng(0)
    y_toy = rng.integers(0, 2, size=500)
    c_toy = np.repeat(np.arange(50), 10)
    sp = get_client_disjoint_splits(y_toy, c_toy, n_folds=5)
    assert len(sp) == 5
    for tr, es, te in sp:
        assert set(c_toy[tr]).isdisjoint(set(c_toy[es]))
        assert set(c_toy[tr]).isdisjoint(set(c_toy[te]))
        assert set(c_toy[es]).isdisjoint(set(c_toy[te]))
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize-only", action="store_true",
                    help="Recompute C3 summary CSVs/JSON purely from existing cells/*.csv without touching cells or run.log.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    if args.self_test:
        assert verify_math_invariants()
        print("[SELF-TEST PASS] run_c3_ecl_client_disjoint invariants verified.", flush=True)
        return

    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    t0 = time.time()
    if args.summarize_only:
        part_csv = OUT / "c3_ecl_client_disjoint_partitions.csv"
        if not part_csv.exists():
            raise FileNotFoundError(f"Missing {part_csv} for --summarize-only")
        part_rows = pd.read_csv(part_csv).to_dict(orient="records")
        cell_data = {arm: [] for arm in ARMS}
        for arm in ARMS:
            for s in range(3):
                for f in range(5):
                    fp = CELLS_DIR / f"{arm}_{s}_{f}.csv"
                    if not fp.exists():
                        raise FileNotFoundError(f"Missing cell CSV for --summarize-only: {fp}")
                    df_c = pd.read_csv(fp)
                    assert len(df_c) == 1 and np.isfinite(float(df_c["ap"].iloc[0])), f"Invalid cell {fp}"
                    cell_data[arm].append(df_c.iloc[0].to_dict())
    else:
        Wp, y, atk_type, cids, mag = build_ecl_dataset_with_clients()
        splits = get_client_disjoint_splits(y, cids, n_folds=5)
        Xg = Wp.reshape(len(Wp), 7, PPD)
        Xm = Wp[:, None, :]
        sc_scales = sorted({1, PPD // 2, PPD, 2 * PPD, 3 * PPD})
        sc_scales = [x for x in sc_scales if 0 < x < WIN]

        part_rows = []
        for f, (tr, es, te) in enumerate(splits):
            part_rows.append({
                "fold": f,
                "train_win": len(tr), "train_pos": int(y[tr].sum()), "train_clients": len(set(cids[tr].tolist())),
                "es_win": len(es), "es_pos": int(y[es].sum()), "es_clients": len(set(cids[es].tolist())),
                "te_win": len(te), "te_pos": int(y[te].sum()), "te_clients": len(set(cids[te].tolist())),
                "client_overlap": 0,
            })
        df_part = pd.DataFrame(part_rows)
        df_part.to_csv(OUT / "c3_ecl_client_disjoint_partitions.csv", index=False)
        log("=== C3: CLIENT-DISJOINT 5-FOLD ECL PARTITIONS (0% CLIENT OVERLAP) ===")
        log(df_part.to_string(index=False))

        if args.smoke:
            tr, es, te = splits[0]
            seed_everything(0)
            m = build_arm("a34_generic", weeks=7, days=PPD)
            cfg = TrainConfig(epochs=1, patience=10, batch_size=128, eval_batch_size=256,
                              pos_fraction=0.5, optimizer="AdamW", lr=3e-4, weight_decay=1e-4,
                              warmup_frac=0.05, min_lr_ratio=0.01, eval_every=1, grad_clip_norm=1.0, device=args.device)
            _, best = train(m, WeeklyGridDataset(Xg[tr], y[tr]), WeeklyGridDataset(Xg[es], y[es]), cfg)
            sc = _inference_batch(best, Xg[te], torch.device(args.device))
            print(f"[SMOKE PASS] C3 MAMRD ep=1 client-disjoint AP={average_precision_score(y[te], sc):.4f}")
            return

        cols = ["arm", "seed", "fold", "n_test", "n_pos", "ap", "roc", "rec_1pct", "rec_5pct",
                "wall_s", "val_labels_json", "val_scores_json", "val_atk_json"]

        total = len(ARMS) * 3 * 5
        idx_job = 0
        cell_data = {arm: [] for arm in ARMS}

        for arm in ARMS:
            for s in range(3):
                for f in range(5):
                    idx_job += 1
                    fp = CELLS_DIR / f"{arm}_{s}_{f}.csv"
                    tr, es, te = splits[f]
                    if args.resume and fp.exists() and fp.stat().st_size > 100:
                        df_c = pd.read_csv(fp)
                        if len(df_c) == 1 and np.isfinite(float(df_c["ap"].iloc[0])):
                            cell_data[arm].append(df_c.iloc[0].to_dict())
                            log(f"[{idx_job}/{total}] SKIP {fp.name} AP={float(df_c['ap'].iloc[0]):.4f}")
                            continue
                    cool_if_hot()
                    t_c = time.time()
                    if arm == "msdm":
                        msdm_model, _, _ = train_msdm(
                            Xm[tr], y[tr], Xm[es], y[es], scales=sc_scales,
                            period=PPD, in_channels=1, epochs=args.epochs,
                            batch=256, device=args.device, seed=s,
                        )
                        _, sc = evaluate_msdm(msdm_model, Xm[te], y[te], torch.device(args.device))
                        del msdm_model
                    else:
                        seed_everything(s)
                        model = build_arm(arm, weeks=7, days=PPD)
                        cfg = TrainConfig(epochs=args.epochs, patience=10, batch_size=128, eval_batch_size=256,
                                          pos_fraction=0.5, optimizer="AdamW", lr=3e-4, weight_decay=1e-4,
                                          eval_every=3, warmup_frac=0.05, min_lr_ratio=0.01,
                                          grad_clip_norm=1.0, device=args.device)
                        _, best = train(model, WeeklyGridDataset(Xg[tr], y[tr]), WeeklyGridDataset(Xg[es], y[es]), cfg)
                        sc = _inference_batch(best, Xg[te], torch.device(args.device))
                        del model, best

                    wall = time.time() - t_c
                    gc.collect()
                    torch.cuda.empty_cache()

                    yt = y[te]
                    atkt = atk_type[te]
                    sc_arr = np.asarray(sc, dtype=float).ravel()
                    r1 = recall_at_fpr(yt, sc_arr, 0.01)
                    r5 = recall_at_fpr(yt, sc_arr, 0.05)
                    row = {
                        "arm": arm, "seed": s, "fold": f,
                        "n_test": int(len(te)), "n_pos": int(yt.sum()),
                        "ap": float(average_precision_score(yt, sc_arr)),
                        "roc": float(roc_auc_score(yt, sc_arr)),
                        "rec_1pct": float(r1), "rec_5pct": float(r5),
                        "wall_s": round(wall, 1),
                        "val_labels_json": json.dumps(yt.astype(int).tolist()),
                        "val_scores_json": json.dumps([round(float(v), 10) for v in sc_arr]),
                        "val_atk_json": json.dumps(atkt.astype(int).tolist()),
                    }
                    with open(fp, "w", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerow(row)
                    cell_data[arm].append(row)
                    log(f"[{idx_job}/{total}] {arm} s={s} f={f} | AP={row['ap']:.4f} ROC={row['roc']:.4f} "
                        f"R@5%={row['rec_5pct']:.3f} wall={row['wall_s']}s")

    # Summary Table across 15 client-disjoint cells per detector
    sum_rows, atk_rows, paired_rows = [], [], []
    mamrd_aps = np.array([float(r["ap"]) for r in sorted(cell_data["a34_generic"], key=lambda d: (int(d["seed"]), int(d["fold"])))])

    for arm in ARMS:
        rows_a = sorted(cell_data[arm], key=lambda d: (int(d["seed"]), int(d["fold"])))
        aps = np.array([float(r["ap"]) for r in rows_a])
        rocs = np.array([float(r["roc"]) for r in rows_a])
        r1s = np.array([float(r["rec_1pct"]) for r in rows_a])
        r5s = np.array([float(r["rec_5pct"]) for r in rows_a])
        alo, ahi = bca_ci(aps); rlo, rhi = bca_ci(rocs)
        sum_rows.append({
            "detector": DISPLAY[arm], "arm": arm, "n_cells": len(rows_a),
            "ap_mean": float(aps.mean()), "ap_ci_lo": alo, "ap_ci_hi": ahi,
            "roc_mean": float(rocs.mean()), "roc_ci_lo": rlo, "roc_ci_hi": rhi,
            "rec_1pct_mean": float(r1s.mean()), "rec_5pct_mean": float(r5s.mean()),
        })
        if arm != "a34_generic":
            diff = mamrd_aps - aps
            dlo, dhi = bca_ci(diff)
            _, pv = wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="exact")
            paired_rows.append({
                "contrast": f"MAMRD - {DISPLAY[arm]}",
                "dAP": float(diff.mean()), "ci_lo": dlo, "ci_hi": dhi,
                "wilcoxon_p": float(pv), "mamrd_wins_of_15": int(np.sum(diff > 0)),
            })

        # Per-attack Recall@5%FPR across 15 client-disjoint cells
        for k in range(1, 7):
            rec_k = []
            for r in rows_a:
                yt = np.array(json.loads(r["val_labels_json"]), dtype=int)
                sc_arr = np.array(json.loads(r["val_scores_json"]), dtype=float)
                ak = np.array(json.loads(r["val_atk_json"]), dtype=int)
                thr5 = float(np.quantile(sc_arr[yt == 0], 0.95))
                mk = ak == k
                if mk.any():
                    rec_k.append(float((sc_arr[mk] >= thr5).mean()))
            klo, khi = bca_ci(rec_k)
            atk_rows.append({
                "detector": DISPLAY[arm], "attack": f"FDI{k}", "n_cells": len(rec_k),
                "rec_at_5pct_fpr_mean": float(np.mean(rec_k)), "ci_lo": klo, "ci_hi": khi,
            })

    df_sum = pd.DataFrame(sum_rows)
    df_atk = pd.DataFrame(atk_rows)
    df_pair = pd.DataFrame(paired_rows)
    df_sum.to_csv(OUT / "c3_ecl_client_disjoint_summary.csv", index=False)
    df_atk.to_csv(OUT / "c3_ecl_client_disjoint_per_attack.csv", index=False)
    df_pair.to_csv(OUT / "c3_ecl_paired_contrasts.csv", index=False)

    log("\n--- C3 Client-Disjoint 5-Fold ECL Summary (15 cells/detector) ---")
    log(df_sum.to_string(index=False))
    log("\n--- C3 Paired Contrasts ---")
    log(df_pair.to_string(index=False))
    log("\n--- C3 Per-Attack Recall@5%FPR (FDI1 vs FDI2-FDI6) ---")
    log(df_atk[df_atk["detector"].isin(["MAMRD", "MSDM", "HORLN"])].to_string(index=False))

    (OUT / "c3_summary.json").write_text(json.dumps({
        "partitions": part_rows, "summary": sum_rows, "paired": paired_rows, "per_attack": atk_rows,
    }, indent=2))
    if not args.summarize_only:
        log(f"\n[DONE] C3 completed in {(time.time() - t0)/60.0:.2f} min -> {OUT}")
        (OUT / "run.log").write_text(scrub_paths("\n".join(log_lines) + "\n"))


if __name__ == "__main__":
    main()
