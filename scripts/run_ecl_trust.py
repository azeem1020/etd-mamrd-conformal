"""ECL cross-dataset trustworthy extension (Task #3, PLAN_TRUST_2026-06-02.md).

Replicates the LOCKED 30-min ECL protocol of run_ecl_hourly.py EXACTLY (so AP
reproduces), but (a) dumps per-instance test scores+labels+magnitude per cell, and
(b) runs 3 folds x 3 seeds per arm (the locked table was fold-0 only) so the conformal
layer has enough cells/positives for per-stratum CIs. Self-contained, resumable
(skips existing cells), GPU heat-gated.

ECL has NO calendar/weekday structure and carries injected FDI -> the SGCC weekday/
weekend phenotype is NOT meaningful here. Per the locked finding ("detectability =
MAGNITUDE, not direction") we stratify ECL by MAGNITUDE only (Heavy/Light by the
TRAIN-fit median of observed mean-consumption -> leakage-safe). Analysis reports
base-rate-normalized LIFT (same prevalence-confound control as SGCC).

Run smoke:  $P run_ecl_trust.py --smoke
Run full :  $P run_ecl_trust.py            (background; ~6 arms x 9 cells, heat-gated)
"""
from __future__ import annotations
import sys, csv, json, time, argparse, subprocess
from pathlib import Path
THIS = Path(__file__).resolve().parent
for p in (str(THIS), str(THIS.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score
from zkmix_lab.data import KFoldSpec
from zkmix_lab.training import TrainConfig, WeeklyGridDataset, seed_everything, train
from phased.models import build_arm
from run_local_grid import _stratified_three_way_split, _inference_batch
from msdm.fdi import ATTACKS
from msdm.train import train_msdm
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

ECL = ECL_PATH
OUT = (RESULTS_DIR / "ecl_trust"); OUT.mkdir(parents=True, exist_ok=True)
FREQ, PPD = "30min", 48
WIN = 7 * PPD
CAP = 25000
COOL_TO, GATE, MAXWAIT = 55, 75, 600
BUILD_ARMS = ["a34_generic", "horln", "wdcnn", "grud", "hybridattn"]   # via build_arm
# + "msdm" via train_msdm (matches locked MSDM 0.870)
CSV_COLS = ["arm", "seed", "fold", "n_test", "n_pos", "ap", "roc",
            "mag_median_tr", "val_labels_json", "val_scores_json", "val_mag_json"]


def gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        return int(out.decode().split("\n")[0])
    except Exception:
        return 0


def cool(target=COOL_TO, label=""):
    t = gpu_temp()
    if t <= target:
        return
    print(f"[heat] {label} cooling {t}->{target}C", flush=True)
    waited = 0
    while gpu_temp() > target and waited < MAXWAIT:
        time.sleep(20); waited += 20


def load_windows():
    df = pd.read_csv(ECL, sep=";", decimal=",", index_col=0); df.index = pd.to_datetime(df.index)
    res = df.resample(FREQ).mean().values.T.astype(np.float64)
    n, T = res.shape; nw = T // WIN
    out = [res[i, w*WIN:(w+1)*WIN] for i in range(n) for w in range(nw)
           if np.nansum(np.abs(res[i, w*WIN:(w+1)*WIN])) > 1e-6]
    return np.array(out)


def prep(W):
    m = np.nanmean(W, 1, keepdims=True); s = np.nanstd(W, 1, keepdims=True)
    W = np.where(W > m + 3*s, m + 3*s, W)
    lo = np.nanmin(W, 1, keepdims=True); hi = np.nanmax(W, 1, keepdims=True); d = hi - lo; d[d == 0] = 1
    return ((W - lo) / d).astype(np.float32)


def build_dataset():
    """EXACT replication of run_ecl_hourly data path (rng order matters)."""
    W = load_windows()
    rng = np.random.default_rng(0)
    if len(W) > CAP:
        W = W[rng.choice(len(W), CAP, False)]
    # FDI with a FRESH default_rng(0) (matches run_ecl_hourly.fdi call)
    frng = np.random.default_rng(0); y = np.zeros(len(W), int); Wo = W.copy()
    for i in frng.choice(len(W), int(0.10 * len(W)), False):
        Wo[i] = ATTACKS[frng.integers(6)](W[i], frng); y[i] = 1
    mag = np.nanmean(Wo, axis=1)            # observed magnitude (post-FDI, pre-norm)
    mag = np.where(np.isfinite(mag), mag, 0.0)
    Wp = prep(Wo)
    return Wp, y, mag


def split_idx(y, fold):
    folds = list(KFoldSpec(family="random", n_folds=5, seed=0).iter_folds(y))
    tr, val = folds[fold]
    _, es, te = _stratified_three_way_split(val, y, (0.33, 0.33, 0.34),
                                            rng=np.random.default_rng(fold))
    return tr, es, te


def dump(arm, s, f, te, yt, sc, mag, tr_mag_median):
    row = dict(arm=arm, seed=s, fold=f, n_test=len(te), n_pos=int(yt.sum()),
               ap=float(average_precision_score(yt, sc)),
               roc=float(roc_auc_score(yt, sc)),
               mag_median_tr=float(tr_mag_median),
               val_labels_json=json.dumps(yt.astype(int).tolist()),
               val_scores_json=json.dumps([round(float(x), 6) for x in sc]),
               val_mag_json=json.dumps([round(float(x), 6) for x in mag[te]]))
    fp = OUT / f"{arm}_{s}_{f}.csv"
    with open(fp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS); w.writeheader(); w.writerow(row)
    print(f"[{arm} s{s} f{f}] AP={row['ap']:.4f} ROC={row['roc']:.4f} "
          f"n={len(te)} pos={row['n_pos']} T={gpu_temp()}C", flush=True)


def run(seeds, folds, epochs, device, smoke):
    Wp, y, mag = build_dataset()
    print(f"[ecl] {Wp.shape} theft%={100*y.mean():.2f} mag[min/med/max]="
          f"{mag.min():.3f}/{np.median(mag):.3f}/{mag.max():.3f}", flush=True)
    Xg = Wp.reshape(len(Wp), 7, PPD)            # build_arm grid
    Xm = Wp[:, None, :]                          # MSDM (N,1,T)
    sc_scales = sorted({1, PPD // 2, PPD, 2 * PPD, 3 * PPD})
    sc_scales = [x for x in sc_scales if 0 < x < WIN]
    arms = (["wdcnn"] if smoke else BUILD_ARMS + ["msdm"])
    for arm in arms:
        for s in seeds:
            for f in folds:
                fp = OUT / f"{arm}_{s}_{f}.csv"
                if fp.exists() and not smoke:
                    print(f"[skip] {fp.name}", flush=True); continue
                while gpu_temp() > GATE:
                    time.sleep(20)
                tr, es, te = split_idx(y, f)
                tr_med = float(np.median(mag[tr]))    # leakage-safe magnitude threshold
                if arm == "msdm":
                    _, _, sc = train_msdm(Xm[tr], y[tr], Xm[te], y[te], scales=sc_scales,
                                          period=PPD, in_channels=1, epochs=epochs,
                                          batch=256, device=device, seed=s)
                else:
                    seed_everything(s)
                    model = build_arm(arm, weeks=7, days=PPD)
                    cfg = TrainConfig(epochs=epochs, patience=8, batch_size=256,
                                      pos_fraction=0.5, lr=3e-4, weight_decay=0.01,
                                      warmup_frac=0.05, min_lr_ratio=0.01,
                                      grad_clip_norm=1.0, device=device)
                    _, best = train(model, WeeklyGridDataset(Xg[tr], y[tr]),
                                    WeeklyGridDataset(Xg[es], y[es]), cfg)
                    sc = _inference_batch(best, Xg[te], torch.device(device))
                dump(arm, s, f, te, y[te], np.asarray(sc).ravel(), mag, tr_med)
                if smoke:
                    return
                time.sleep(20)
                cool(label=f"post {arm} s{s} f{f}")
    print("@@@ ECL TRUST DUMP DONE @@@", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.seeds, a.folds, (3 if a.smoke else a.epochs), a.device, a.smoke)
