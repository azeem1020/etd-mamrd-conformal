"""Phase D experiment runner.

Reuses the Phase C training infrastructure (zkmix_lab.training) unchanged.
Adds only the Phase D model-building logic and a manifest writer that records
results to results/phase_d_grid/ — separate from the Phase C manifest.

Cell key format
---------------
<arm>__<regime>__<split>__seed<S>__fold<F>__kf<K>

where arm ∈ {a1_cc_name_only, a2_ms_only, a3_cc_name_ms, a4_cc_name_d1,
             a5_cc_name_ms_nowide}.

Idempotency: re-running skips cells whose key already appears in manifest.csv.
"""
from __future__ import annotations

import json
import sys
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors  # peer group construction (Phase J)

# ---------------------------------------------------------------------------
# Resolve zkmix_lab on sys.path.
# ---------------------------------------------------------------------------

def _ensure_zkmix_on_path() -> None:
    lab = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "zkmix_lab")
    )
    if lab not in sys.path:
        sys.path.insert(0, lab)

_ensure_zkmix_on_path()

from zkmix_lab.data import KFoldSpec, load_sgcc, slice_regime   # type: ignore
from zkmix_lab.metrics import summarise, paired_bootstrap       # type: ignore
from zkmix_lab.calibration import calibrate_cell                # type: ignore
from zkmix_lab.conformal import expected_calibration_error_with_bins  # type: ignore
from zkmix_lab.training import (                                # type: ignore
    TrainConfig,
    WeeklyGridDataset,
    seed_everything,
    train,
)

from .models import ARM_BUILDERS, build_arm


# ---------------------------------------------------------------------------
# Phase J: Peer-Relative Consumption Encoding helpers
# ---------------------------------------------------------------------------

#: Arms that require the PRCE peer-residual augmentation (D*2 input stacking).
#: a9_lean does NOT need augmentation — it is the comparison baseline (use_prce=False).
#: Phase K: A20 (a20_gnn_peer) and A21 (a21_full_mitigation) also use the (N, W, 14)
#: input contract; the model swaps the branch internally via use_peer_gnn.
PRCE_ARMS: frozenset[str] = frozenset({"a17_prce", "a20_gnn_peer", "a21_full_mitigation"})

#: Phase K — arms whose training set is augmented with synthetic theft examples
#: (PhysSim, ``phased.augment.augment_training_set``).
AUGMENT_ARMS: frozenset[str] = frozenset({"a18_physsim_aug", "a21_full_mitigation"})

#: Phase K — arms whose backbone is initialised from a contrastive SSL pre-training
#: pass on the full unlabeled corpus before supervised fine-tuning.
SSL_ARMS: frozenset[str] = frozenset({"a19_ssl_identity", "a21_full_mitigation"})

# ---------------------------------------------------------------------------
# Phase L — Theft-Direction-Symmetric Multi-Task Network arm sets (PLAN_L)
# ---------------------------------------------------------------------------

#: Phase L — symmetric augmentation (Heavy/Light Risers + Droppers + Flat).
SYM_AUG_ARMS: frozenset[str] = frozenset({"a22_sym_aug", "a29_tds_mtn_plus"})

#: Phase L — SupCon archetype-aware pre-training (replaces SimCLR for these arms).
SUPCON_ARMS: frozenset[str] = frozenset({"a23_supcon", "a29_tds_mtn_plus"})

#: Phase L — arms whose forward pass needs the engineered-feature tensor.
ENG_FEAT_ARMS: frozenset[str] = frozenset({"a24_eng_features", "a29_tds_mtn_plus", "a30_eng_mamba"})

#: Phase L — arms that consume signed-peer features (own | sig_res | peer_std | peer_iqr).
SIGNED_PEER_ARMS: frozenset[str] = frozenset({"a25_signed_peer", "a29_tds_mtn_plus"})

#: Phase L — arms that use the MultiTaskZKMixV4 wrapper + train_multitask().
MULTITASK_ARMS: frozenset[str] = frozenset({"a26_multitask", "a29_tds_mtn_plus"})

#: Phase L — arms that compute Confident-Learning sample weights before training.
CL_ARMS: frozenset[str] = frozenset({"a27_conf_learn", "a29_tds_mtn_plus"})

#: Phase L — arms that use SAM optimizer + Test-Time Augmentation inference.
SAM_TTA_ARMS: frozenset[str] = frozenset({"a28_sam_tta", "a29_tds_mtn_plus"})

# ---------------------------------------------------------------------------
# Phase K: SSL pre-training cache
# ---------------------------------------------------------------------------
# We cache the SSL pre-training result per (corpus_hash, seed, n_epochs) so a
# single Kaggle run doesn't re-train SSL for every cell. The cache key is a
# stable function of the data and SSL hyper-parameters — not the downstream
# fold or arm — so all SSL_ARMS share the same checkpoint.

_SSL_CACHE_DIR_DEFAULT = Path(os.environ.get(
    "PHASE_K_SSL_CACHE",
    str(Path.home() / ".cache" / "phase_k_ssl"),
))


def _ssl_cache_key(X: np.ndarray, n_epochs: int, batch_size: int, lr: float, seed: int) -> str:
    """Stable hash key for SSL pre-training cache."""
    import hashlib
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X.shape).tobytes())
    # Sample a deterministic content sketch — first row, middle row, last row.
    n = X.shape[0]
    for i in (0, n // 2, n - 1):
        h.update(np.nan_to_num(X[i].astype(np.float32), nan=-1.0).tobytes())
    h.update(f"{n_epochs}|{batch_size}|{lr:.6e}|{seed}".encode())
    return h.hexdigest()[:16]


# SSL hyper-parameters (fixed across cells; cache key reflects these).
# Override via env vars: PHASE_K_SSL_EPOCHS, PHASE_K_SSL_BATCH, PHASE_K_SSL_LR, PHASE_K_SSL_SEED.
def _ssl_hp() -> dict:
    return dict(
        n_epochs=int(os.environ.get("PHASE_K_SSL_EPOCHS", "30")),
        batch_size=int(os.environ.get("PHASE_K_SSL_BATCH", "512")),
        lr=float(os.environ.get("PHASE_K_SSL_LR", "1e-3")),
        seed=int(os.environ.get("PHASE_K_SSL_SEED", "0")),
    )


def cfg_device(train_cfg) -> str:
    """Extract device string from a TrainConfig (zkmix_lab) instance."""
    dev = getattr(train_cfg, "device", "cuda")
    return str(dev)


def _ssl_init_model(
    model: "torch.nn.Module",
    X_reg: np.ndarray,
    regime_weeks: int,
    days: int,
    device: str = "cuda",
    cache_dir: Path | None = None,
) -> None:
    """Pre-train the A9-lean backbone with NT-Xent and load shape-compat params.

    Cache lookup avoids re-running the SSL pass for every cell — once computed
    for a given (data, hp) it is reused across all SSL_ARMS cells in a run.
    """
    cache_dir = cache_dir or _SSL_CACHE_DIR_DEFAULT
    cache_dir.mkdir(parents=True, exist_ok=True)
    hp = _ssl_hp()
    key = _ssl_cache_key(X_reg, **hp)
    ckpt_path = cache_dir / f"ssl_a9lean_{key}.pt"

    if not ckpt_path.exists():
        print(f"  [ssl] pre-training A9-lean backbone (cache miss: {key}) "
              f"epochs={hp['n_epochs']} ...", flush=True)
        from .models import build_a9_lean
        from .ssl import ssl_pretrain_backbone

        builder = lambda: build_a9_lean(weeks=regime_weeks, days=days)
        state = ssl_pretrain_backbone(
            X_reg, builder,
            n_epochs=hp["n_epochs"],
            batch_size=hp["batch_size"],
            lr=hp["lr"],
            seed=hp["seed"],
            device=device,
            verbose=True,
        )
        torch.save(state, ckpt_path)
        print(f"  [ssl] checkpoint saved to {ckpt_path}", flush=True)
    else:
        print(f"  [ssl] cache hit: {ckpt_path}", flush=True)

    state = torch.load(ckpt_path, map_location="cpu")
    from .ssl import load_state_dict_compatible
    loaded, skipped = load_state_dict_compatible(model, state, verbose=True)
    print(f"  [ssl] loaded {len(loaded)} backbone params; "
          f"skipped {len(skipped)} (shape mismatch — branch-specific).",
          flush=True)


def compute_peer_residuals(
    X_raw: np.ndarray,
    train_idx: np.ndarray,
    y: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Compute peer-relative consumption residuals for all N customers.

    For each customer c, finds k nearest **normal** (y=0) neighbours from
    ``train_idx`` using top-quartile monthly block means as the similarity
    space (robust to step-down theft — long-term thieves' peak months are
    their clean months, so full-window mean is not suppressed by theft).
    Returns r[c, w, d] = X_filled[c, w, d] − peer_baseline[c, w, d].

    Parameters
    ----------
    X_raw   : (N, W, D)  float32 with NaN at missing positions.
    train_idx : 1-D int array — indices of the current fold's training set.
    y       : (N,) int  — labels for ALL customers (0=normal, 1=theft).
    k       : number of nearest normal neighbours for the peer baseline.

    Returns
    -------
    peer_res : (N, W, D) float32 — peer-relative residual.
        NaN positions in the customer's own series are zeroed (no deviation
        claimed where data is missing).

    Notes
    -----
    * Peer groups are built exclusively from normal (y=0) customers in
      ``train_idx`` to prevent label leakage into the test fold.
    * k-NN feature: top-25% monthly block nanmeans (sorted descending, take
      top quarter). This is robust to step-down theft where the thief's
      full-window mean is suppressed but their peak months reflect normal use.
    * Peer baseline uses full nanmean over k peers (fold-safe, label-safe).
    * If a week is NaN for ALL k peers, peer_baseline = 0 at that week
      (conservative — no artificial signal injected).
    """
    N, W, D = X_raw.shape
    nb = W // 7   # 21 monthly blocks for W=147

    # ── 1. 21-dim monthly nanmean feature per customer (NaN-aware) ──────────
    x4 = X_raw[:, :nb * 7, :].reshape(N, nb, 7, D)          # (N, 21, 7, 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        monthly = np.nanmean(x4.reshape(N, nb, -1), axis=-1)  # (N, 21)
    monthly = np.nan_to_num(monthly, nan=0.0)

    # ── 1b. Peak-feature for k-NN: top-25% monthly blocks (descending sort) ─
    # Robust to step-down theft: a thief's highest-consumption months are their
    # pre-theft (clean) months, so this feature reflects their true normal level.
    top_q    = max(1, nb // 4)                                # ≈ 5 months / 21
    monthly_desc = np.sort(monthly, axis=1)[:, ::-1]         # (N, 21) descending
    knn_feat = monthly_desc[:, :top_q]                        # (N, top_q)

    # ── 2. Normal-only pool from training fold ───────────────────────────────
    normal_mask = y[train_idx] == 0
    normal_idx  = train_idx[normal_mask]                      # global indices

    k_eff = min(k, max(1, len(normal_idx)))

    # ── 3. k-NN lookup in peak-feature space ─────────────────────────────────
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(knn_feat[normal_idx])
    _, local_idx = nn.kneighbors(knn_feat)                    # (N, k_eff)
    global_idx   = normal_idx[local_idx]                      # (N, k_eff)

    # ── 4. Peer baseline: nanmean of k peers at each (week, day) ─────────────
    # X_raw[global_idx] → (N, k_eff, W, D); nanmean over axis=1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peer_baseline = np.nanmean(X_raw[global_idx], axis=1)  # (N, W, D)
    peer_baseline = np.nan_to_num(peer_baseline, nan=0.0)

    # ── 5. Residual: customer (0-filled) − peer baseline ─────────────────────
    X_filled  = np.where(np.isnan(X_raw), 0.0, X_raw)        # (N, W, D)
    peer_res  = X_filled - peer_baseline                      # (N, W, D)
    peer_res[np.isnan(X_raw)] = 0.0                           # mask own NaNs

    return peer_res.astype(np.float32)


# ---------------------------------------------------------------------------
# Cell definition
# ---------------------------------------------------------------------------

ArmName = Literal[
    "a1_cc_name_only",
    "a2_ms_only",
    "a3_cc_name_ms",
    "a4_cc_name_d1",
    "a5_cc_name_ms_nowide",
]


@dataclass(frozen=True)
class CellD:
    """One atomic Phase D experiment.

    Parameters match Phase C Cell (runner.py) for comparability.
    arm selects the ZKMixV4 variant.
    """
    arm:          str    # ArmName
    regime:       str    = "full"        # "full" | "high_miss" | "stable"
    split_family: str    = "random"      # "random" | "stratified"
    seed:         int    = 0
    fold:         int    = 0
    kfold_seed:   int    = 0             # fixed=0 matches Phase C kfold_seed convention

    def key(self) -> str:
        return (
            f"{self.arm}__{self.regime}__{self.split_family}"
            f"__seed{self.seed}__fold{self.fold}__kf{self.kfold_seed}"
        )


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------


def phase_d_grid(
    arms: list[str] | None = None,
    seeds: list[int] | None = None,
    n_folds: int = 3,
    kfold_seed: int = 0,
    regime: str = "full",
    split_family: str = "random",
) -> list[CellD]:
    """Return the complete Phase D cell list.

    Defaults match the pre-registration protocol:
      arms    : all 5 arms (A1–A5)
      seeds   : [0, 1, 2, 3, 4]
      n_folds : 3
    """
    if arms is None:
        arms = list(ARM_BUILDERS.keys())
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    cells: list[CellD] = []
    for arm in arms:
        for seed in seeds:
            for fold in range(n_folds):
                cells.append(CellD(
                    arm=arm,
                    regime=regime,
                    split_family=split_family,
                    seed=seed,
                    fold=fold,
                    kfold_seed=kfold_seed,
                ))
    return cells


# ---------------------------------------------------------------------------
# Stratified 3-way split — mirrors Phase C runner._stratified_three_way_split
# exactly so that Phase D test indices match Phase C test indices for
# per-sample paired bootstrap.
# ---------------------------------------------------------------------------


def _stratified_three_way_split(
    val_idx: np.ndarray,
    y: np.ndarray,
    fractions: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split val_idx into (early_stop, cal, test) preserving the theft prior."""
    f_es, f_cal, _ = fractions
    pos = val_idx[y[val_idx] == 1]
    neg = val_idx[y[val_idx] == 0]
    pos_perm = rng.permutation(pos)
    neg_perm = rng.permutation(neg)
    n_pos_es,  n_pos_cal = int(len(pos) * f_es), int(len(pos) * f_cal)
    n_neg_es,  n_neg_cal = int(len(neg) * f_es), int(len(neg) * f_cal)
    es  = np.concatenate([pos_perm[:n_pos_es],  neg_perm[:n_neg_es]])
    cal = np.concatenate([
        pos_perm[n_pos_es : n_pos_es + n_pos_cal],
        neg_perm[n_neg_es : n_neg_es + n_neg_cal],
    ])
    test = np.concatenate([
        pos_perm[n_pos_es + n_pos_cal :],
        neg_perm[n_neg_es + n_neg_cal :],
    ])
    es.sort(); cal.sort(); test.sort()
    return es, cal, test


def _inference_batch(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return sigmoid scores for X in batches (keeps VRAM stable)."""
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device, non_blocking=True)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores)


# ---------------------------------------------------------------------------
# Single-cell runner
# ---------------------------------------------------------------------------


def _run_cell(
    cell: CellD,
    data_dir: str | Path,
    train_cfg: TrainConfig,
) -> dict:
    """Train and evaluate one Phase D cell.

    Split protocol mirrors Phase C exactly so that (seed, fold, kfold_seed)
    produce the same test indices — required for per-sample paired bootstrap.

    Protocol:
      • Train       : 4 of 5 KFold folds (train_idx).
      • val_idx     : 1 held-out fold, stratified-split 33/33/34 into
                      (early_stop, cal, test).
      • Early stop  : model selection on early_stop slice.
      • Cal slice   : prior + temperature calibration fit here.
      • Test slice  : all reported metrics.  Never seen during training or
                      calibration.
    """
    data_dir = Path(data_dir)

    # Load data — same CSV paths as Phase C runner.
    X, y = load_sgcc(
        features_csv=data_dir / "after_preprocess_data.csv",
        labels_csv=data_dir / "label.csv",
    )
    X_reg, regime_weeks = slice_regime(X, cell.regime)        # (N, W', D)
    days = X_reg.shape[2]

    # K-fold split (must match Phase C kfold_seed=0 convention).
    spec = KFoldSpec(
        family=cell.split_family,
        n_folds=5,
        seed=cell.kfold_seed,
    )
    folds = list(spec.iter_folds(y))
    if cell.fold >= len(folds):
        raise IndexError(f"fold {cell.fold} out of range for {len(folds)} splits.")
    train_idx, val_idx = folds[cell.fold]

    # Stratified 3-way split of the held-out fold — RNG seeded identically to
    # Phase C: default_rng(kfold_seed * 7 + fold).
    rng = np.random.default_rng(cell.kfold_seed * 7 + cell.fold)
    es_idx, cal_idx, test_idx = _stratified_three_way_split(
        val_idx, y, fractions=(0.33, 0.33, 0.34), rng=rng,
    )

    # ── Phase J: peer-residual augmentation ─────────────────────────────────
    # If the arm uses PRCE, compute fold-safe peer residuals and stack them
    # onto the consumption tensor.  X_aug (N, W, D*2) is transparent to
    # WeeklyGridDataset and the training loop; ZKMixV4 splits internally.
    if cell.arm in PRCE_ARMS:
        peer_res = compute_peer_residuals(X_reg, train_idx, y, k=20)
        X_aug = np.concatenate([X_reg, peer_res], axis=-1)   # (N, W, D*2)
    else:
        X_aug = X_reg

    # ── Phase K: physics-sim theft augmentation ─────────────────────────────
    # For AUGMENT_ARMS, synthesise synthetic theft examples from normal customers
    # in the training fold and append them. Real labels untouched; only the
    # training set grows. Seeded per cell so augmentation is reproducible.
    train_X = X_aug[train_idx]
    train_y = y[train_idx]
    if cell.arm in AUGMENT_ARMS:
        from .augment import augment_training_set
        aug_rng = np.random.default_rng(cell.seed * 1009 + cell.fold * 31 + cell.kfold_seed)
        train_X, train_y = augment_training_set(
            train_X, train_y, rng=aug_rng, p_aug=0.40
        )
        print(f"  [augment] {cell.arm}: training set {len(X_aug[train_idx])} → "
              f"{len(train_X)} (synthetic {len(train_X) - len(X_aug[train_idx])}, "
              f"theft frac {train_y.mean():.3f})", flush=True)

    # Build datasets.
    train_ds      = WeeklyGridDataset(train_X, train_y)
    early_stop_ds = WeeklyGridDataset(X_aug[es_idx], y[es_idx])

    # Seed before model build so weights are deterministic for a given cell.seed.
    seed_everything(cell.seed)
    model = build_arm(cell.arm, weeks=regime_weeks, days=days)

    # ── Phase K: SSL backbone initialisation ────────────────────────────────
    # For SSL_ARMS, pre-train the A9-lean backbone with NT-Xent contrastive
    # learning on the FULL unlabeled X_reg (no labels), then load shape-compatible
    # parameters into the freshly-built model. Peer-GNN branch (A21) + fuse + head
    # stay randomly initialised. The pre-training is cached per (data, hp) so a
    # multi-cell run runs SSL once and reuses across cells.
    if cell.arm in SSL_ARMS:
        _ssl_init_model(model, X_reg, regime_weeks, days, device=cfg_device(train_cfg))

    # Train.
    cfg = train_cfg
    t0 = time.time()
    train_result, best_model = train(model, train_ds, early_stop_ds, cfg)
    wall_s = time.time() - t0

    device = torch.device(cfg.device)

    # Inference on calibration and test slices.
    cal_scores  = _inference_batch(best_model, X_aug[cal_idx],  device)
    test_scores = _inference_batch(best_model, X_aug[test_idx], device)
    cal_labels  = y[cal_idx]
    test_labels = y[test_idx]

    # Point metrics on test slice.
    metrics = summarise(test_labels, test_scores)

    # ECE (raw) on test slice.
    ece_raw, _ = expected_calibration_error_with_bins(test_scores, test_labels, n_bins=15)

    # Post-hoc calibration: prior-shift + temperature scaling.
    p_train  = max(min(float(cfg.pos_fraction), 1.0 - 1e-3), 1e-3)
    p_deploy = max(min(float(y[train_idx].mean()), 1.0 - 1e-3), 1e-3)
    cal_out  = calibrate_cell(
        cal_scores=cal_scores,
        cal_labels=cal_labels,
        test_scores=test_scores,
        p_train=p_train,
        p_deploy=p_deploy,
    )
    ece_cal, _ = expected_calibration_error_with_bins(
        cal_out.test_scores_calibrated, test_labels, n_bins=15
    )

    val_pr_auc_history = [
        float(rec.get("pr_auc", float("nan")))
        for rec in train_result.val_history
    ]

    return {
        "key":          cell.key(),
        "arm":          cell.arm,
        "regime":       cell.regime,
        "split_family": cell.split_family,
        "seed":         cell.seed,
        "fold":         cell.fold,
        "kfold_seed":   cell.kfold_seed,
        "pr_auc":       float(metrics["pr_auc"]),
        "auroc":        float(metrics["auroc"]),
        "f1_optimal":   float(metrics["f1_optimal"]),
        "precision_at_recall": float(metrics.get("precision_at_recall", float("nan"))),
        "ece_raw":      float(ece_raw),
        "ece_calibrated": float(ece_cal),
        "prior_shift":  float(cal_out.prior_shift),
        "temperature_T": float(cal_out.T),
        "best_val_pr_auc": float(train_result.best_val_pr_auc),
        "best_epoch":   int(train_result.best_epoch),
        "wall_s":       float(wall_s),
        "n_test":       int(len(test_labels)),
        "n_pos_test":   int(test_labels.sum()),
        "val_pr_auc_history_json": json.dumps(val_pr_auc_history),
        # Store test scores for downstream paired bootstrap vs Phase C A0.
        # Column names match Phase C manifest: val_labels_json / val_scores_json
        # (Phase C calls its test slice "val" in the manifest row).
        "val_labels_json":  json.dumps([int(v) for v in test_labels.tolist()]),
        "val_scores_json":  json.dumps([round(s, 6) for s in test_scores.tolist()]),
        "val_scores_calibrated_json": json.dumps(
            [round(s, 6) for s in cal_out.test_scores_calibrated.tolist()]
        ),
    }


# ---------------------------------------------------------------------------
# Grid runner (idempotent)
# ---------------------------------------------------------------------------


def run_grid(
    cells: list[CellD],
    data_dir: str | Path,
    manifest_path: str | Path,
    train_cfg: TrainConfig | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all cells; skip any already in manifest_path.

    Parameters
    ----------
    cells         : list of CellD to run.
    data_dir      : path to SGCC data files (contains data/ subdir).
    manifest_path : CSV path where results are appended.
    train_cfg     : TrainConfig (defaults to Phase C settings if None).
    verbose       : print progress.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    train_cfg = train_cfg or TrainConfig()

    # Load existing manifest.
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        done_keys: set[str] = set(existing["key"].tolist())
    else:
        existing = pd.DataFrame()
        done_keys = set()

    rows: list[dict] = []
    for i, cell in enumerate(cells):
        if cell.key() in done_keys:
            if verbose:
                print(f"[skip] {cell.key()}")
            continue
        if verbose:
            print(f"[{i+1}/{len(cells)}] running {cell.key()} ...", flush=True)
        try:
            row = _run_cell(cell, data_dir, train_cfg)
            rows.append(row)
            # Append to manifest immediately for crash resilience.
            new_df = pd.DataFrame([row])
            if manifest_path.exists():
                new_df.to_csv(manifest_path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(manifest_path, index=False)
            if verbose:
                print(
                    f"  → pr_auc={row['pr_auc']:.4f}  "
                    f"ece_cal={row['ece_calibrated']:.4f}  "
                    f"wall={row['wall_s']:.0f}s"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {cell.key()}: {exc}")

    final = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(rows)
    return final


# ---------------------------------------------------------------------------
# Post-hoc paired bootstrap (vs Phase C A0 arm)
# ---------------------------------------------------------------------------


def compute_verdicts(
    manifest_d: pd.DataFrame,
    manifest_c_path: str | Path,
    output_path: str | Path,
    n_replicates: int = 5000,
    alpha: float = 0.05,
) -> None:
    """Compute paired bootstrap verdicts for Phase D arms vs Phase C A0.

    Phase C A0 = zkmix_v3_name_nn arm from phase_c_grid/manifest.csv.
    For each Phase D arm, the paired comparison uses matched (seed, fold,
    kfold_seed) cells — one test-score vector per matched pair.

    Writes a human-readable verdict file to output_path.
    """
    manifest_c_path = Path(manifest_c_path)
    df_c = pd.read_csv(manifest_c_path)
    # Phase C A0 arm key prefix.
    a0 = df_c[df_c["backbone"].str.startswith("zkmix_v3_name_nn")
              & (df_c["regime"] == "full")
              & (df_c["split_family"] == "random")].copy()
    if a0.empty:
        print("WARNING: no Phase C A0 cells found in manifest.")
        return

    lines = ["# Phase D Paired-Bootstrap Verdicts\n"]
    for arm in manifest_d["arm"].unique():
        subset = manifest_d[manifest_d["arm"] == arm]
        # Pool test labels and scores across matched (seed, fold) pairs.
        all_y    = []
        all_s_a0 = []
        all_s_d  = []
        for _, row_d in subset.iterrows():
            # Find matched Phase C cell by (seed, fold) — same kfold_seed=0.
            matched = a0[
                (a0["seed"] == row_d["seed"])
                & (a0["fold"] == row_d["fold"])
            ]
            if matched.empty:
                continue
            row_c = matched.iloc[0]
            # Phase D stores test slice as val_labels_json / val_scores_json
            # (column names match Phase C convention).
            y_d  = np.array(json.loads(row_d["val_labels_json"]), dtype=np.int32)
            s_d  = np.array(json.loads(row_d["val_scores_json"]),  dtype=np.float64)
            s_c  = np.array(json.loads(row_c["val_scores_json"]),  dtype=np.float64)
            y_c  = np.array(json.loads(row_c["val_labels_json"]),  dtype=np.int32)
            # Both test slices must match — same (kfold_seed, fold, rng) produces
            # identical test_idx, so y_d == y_c element-wise.
            min_n = min(len(y_d), len(s_c))
            all_y.append(y_d[:min_n])
            all_s_d.append(s_d[:min_n])
            all_s_a0.append(s_c[:min_n])

        if not all_y:
            lines.append(f"## {arm}: no matched A0 cells found\n")
            continue

        y_pool  = np.concatenate(all_y)
        s_d_p   = np.concatenate(all_s_d)
        s_a0_p  = np.concatenate(all_s_a0)
        result  = paired_bootstrap(y_pool, s_a0_p, s_d_p,
                                   n_replicates=n_replicates, alpha=alpha)
        verdict = (
            "PASS" if result.ci_low > 0 and result.margin_point >= 0.01
            else "DIRECTIONAL" if result.ci_low > 0
            else "FAIL"
        )
        lines.append(
            f"## {arm}\n"
            f"  Δ (point)  = {result.margin_point:+.4f}\n"
            f"  Δ (mean)   = {result.delta_mean:+.4f}\n"
            f"  95% BCa CI = [{result.ci_low:+.4f}, {result.ci_high:+.4f}]\n"
            f"  Verdict    = {verdict}\n\n"
        )

    Path(output_path).write_text("\n".join(lines))
    print(f"Verdicts written to {output_path}")


__all__ = [
    "CellD",
    "phase_d_grid",
    "run_grid",
    "compute_verdicts",
]
