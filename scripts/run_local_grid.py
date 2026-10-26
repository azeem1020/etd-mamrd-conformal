#!/usr/bin/env python3
"""Local GPU grid runner for Phase D.

Stages (see PLAN_D_LOCAL_GRID_2026-05-27.md):
  Stage 1: A1, A2, A3  — 5 seeds × 3 folds = 15 cells each
  Stage 2: A6           — 15 cells (run only if Stage 1 shows A3 negative synergy)

Usage:
    # Stage 1 (all arms):
    conda run -n tgpu python run_local_grid.py

    # Specific arms:
    conda run -n tgpu python run_local_grid.py --arms a2_ms_only a3_cc_name_ms

    # Stage 2 (add A6):
    conda run -n tgpu python run_local_grid.py --arms a6_branch_selective

    # Verdicts only (after grid is done):
    conda run -n tgpu python run_local_grid.py --verdicts_only
"""
from __future__ import annotations
import sys, os, argparse, json, time
from pathlib import Path

HERE    = Path(__file__).parent.resolve()
LAB_DIR = (HERE / ".." / "zkmix_lab").resolve()
for p in [str(HERE), str(LAB_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import torch

from zkmix_lab.training import TrainConfig
from zkmix_lab.metrics  import summarise, paired_bootstrap
from zkmix_lab.calibration import calibrate_cell
from zkmix_lab.conformal   import expected_calibration_error_with_bins
from zkmix_lab.data        import KFoldSpec, slice_regime
from zkmix_lab.training    import WeeklyGridDataset, seed_everything, train

from phased.models    import build_arm, ARM_BUILDERS
from phased.runner_d  import (
    _stratified_three_way_split, _inference_batch,
    compute_peer_residuals, PRCE_ARMS,
    AUGMENT_ARMS, SSL_ARMS, _ssl_init_model,
    # Phase L hooks
    SYM_AUG_ARMS, SUPCON_ARMS, ENG_FEAT_ARMS, SIGNED_PEER_ARMS,
    MULTITASK_ARMS, CL_ARMS, SAM_TTA_ARMS,
)
from phased.augment   import augment_training_set, SYMMETRIC_ARCHETYPE_WEIGHTS
# Phase L imports (defer heavy modules to runtime where possible)
from phased.archetype     import compute_archetype_labels
from phased.eng_features  import compute_eng_features_standardised
from phased.signed_peer   import compute_signed_peer_features
from phased.multitask     import (
    PhaseLDataset, MultiTaskZKMixV4, MTTrainConfig, train_phase_l,
)
from phased.tta           import tta_inference
from repro_config import RESULTS_DIR, DATA_DIR, SGCC_PATH, ECL_PATH

# Phase F: per-arm TrainConfig overrides (loss_type etc.)
# Keys must match entries in ARM_BUILDERS.  Values are passed as **kwargs to TrainConfig.
ARM_TRAIN_OVERRIDES: dict[str, dict] = {
    "a10_focal": dict(loss_type="focal", focal_gamma=2.0, focal_alpha=0.5),
}


SGCC_CSV      = SGCC_PATH
RESULTS_DIR   = HERE.parent.parent / "results" / "phase_d_local"
MANIFEST_PATH = RESULTS_DIR / "local_manifest.csv"
PHASE_C_MANIFEST = HERE.parent.parent / "results" / "phase_c_grid" / "manifest.csv"


# ---------------------------------------------------------------------------
# Preprocessing (NaN-native, in memory)
# ---------------------------------------------------------------------------

def load_and_preprocess(csv_path: Path = SGCC_CSV) -> tuple[np.ndarray, np.ndarray]:
    """Load SGCC CSV and apply NaN-native preprocessing.

    Steps 2+3+4 only (Zheng 2018 §IV-A); Step 1 NaN-fill OMITTED so
    CC-NaME receives the raw missing-value pattern.
    """
    print(f"[data] Loading {csv_path} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(csv_path)
    labels = df["FLAG"].values.astype(np.int64)
    arr    = df.drop(columns=["CONS_NO", "FLAG"]).values.astype(np.float64)
    print(f"  loaded {df.shape} in {time.time()-t0:.1f}s  "
          f"NaN={np.isnan(arr).mean()*100:.1f}%  "
          f"theft={labels.mean()*100:.1f}%", flush=True)

    # Step 2: nan-aware outlier clip
    ceiling = np.nanmean(arr, axis=1, keepdims=True) + 2.0 * np.nanstd(arr, axis=1, keepdims=True)
    arr = np.where(arr > ceiling, ceiling, arr)

    # Step 3: nan-aware per-user min-max normalise
    row_min, row_max = np.nanmin(arr, axis=1, keepdims=True), np.nanmax(arr, axis=1, keepdims=True)
    denom = row_max - row_min
    denom[denom == 0] = 1.0
    arr = (arr - row_min) / denom                # NaN cells remain NaN

    # Step 4: truncate to multiple of 7
    n_keep = (arr.shape[1] // 7) * 7
    arr    = arr[:, :n_keep].astype(np.float32)
    X      = arr.reshape(len(labels), n_keep // 7, 7)
    print(f"  final shape: {X.shape}  NaN={np.isnan(X).mean()*100:.2f}%", flush=True)
    return X, labels


# ---------------------------------------------------------------------------
# Per-cell runner (matches Phase C protocol exactly)
# ---------------------------------------------------------------------------

def _phase_l_inference(
    model, X: np.ndarray, eng_feat: np.ndarray | None,
    device: torch.device, needs: dict[str, bool], batch_size: int = 1024,
) -> np.ndarray:
    """Inference path for Phase L arms (MT or eng-feat) without TTA.

    Handles three cases:
      * MultiTaskZKMixV4 with eng_feat   → backbone.embed(x, eng_feat=eng) → fuse → head
      * MultiTaskZKMixV4 without eng_feat → model.predict_logit(x)
      * plain ZKMixV4 with eng_feat       → model(x, eng_feat=eng)
    Returns sigmoid scores (N,).
    """
    model.eval()
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            ef = torch.from_numpy(eng_feat[i:i+batch_size]).to(device) if eng_feat is not None else None
            if needs["multitask"]:
                if ef is not None:
                    emb   = model.backbone.embed(xb, eng_feat=ef)
                    fused = model.backbone.fuse(emb)
                    logit = model.backbone.head(fused).squeeze(-1)
                else:
                    logit = model.predict_logit(xb)
            else:
                logit = model(xb, eng_feat=ef) if ef is not None else model(xb)
            out[i:i+batch_size] = torch.sigmoid(logit).cpu().numpy()
    return out


def _arm_uses(arm_name: str) -> dict[str, bool]:
    """Return a dict of which Phase L behaviours this arm needs."""
    return dict(
        # Phase K legacy
        prce_layout=(arm_name in PRCE_ARMS),
        legacy_aug=(arm_name in AUGMENT_ARMS),
        ssl_init=(arm_name in SSL_ARMS),
        # Phase L
        sym_aug=(arm_name in SYM_AUG_ARMS),
        supcon=(arm_name in SUPCON_ARMS),
        eng_feat=(arm_name in ENG_FEAT_ARMS),
        signed_peer=(arm_name in SIGNED_PEER_ARMS),
        multitask=(arm_name in MULTITASK_ARMS),
        cl=(arm_name in CL_ARMS),
        sam_tta=(arm_name in SAM_TTA_ARMS),
    )


def _supcon_init_model(model, X_reg, archetype_y, regime_weeks, D, device):
    """Phase L: SupCon archetype-supervised pre-training, cache-shared by key."""
    import hashlib, os, torch as _torch
    from pathlib import Path as _Path
    from phased.ssl import supcon_pretrain_backbone, load_state_dict_compatible
    from phased.models import build_a9_lean
    from phased.runner_d import _ssl_hp, _SSL_CACHE_DIR_DEFAULT

    cache_dir = _Path(os.environ.get("PHASE_K_SSL_CACHE", str(_SSL_CACHE_DIR_DEFAULT)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    hp = _ssl_hp()

    # Cache key includes ARCHETYPE label distribution so SupCon checkpoints are
    # distinct from plain SimCLR ones (which use the same data but no labels).
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X_reg.shape).tobytes())
    h.update(np.bincount(archetype_y, minlength=6).tobytes())
    h.update(f"SUPCON|{hp['n_epochs']}|{hp['batch_size']}|{hp['lr']:.6e}|{hp['seed']}".encode())
    key = h.hexdigest()[:16]
    ckpt = cache_dir / f"supcon_a9lean_{key}.pt"

    if not ckpt.exists():
        print(f"  [supcon] pre-training (cache miss: {key}) ep={hp['n_epochs']}", flush=True)
        builder = lambda: build_a9_lean(weeks=regime_weeks, days=D)
        state = supcon_pretrain_backbone(
            X_reg, archetype_y, builder,
            n_epochs=hp["n_epochs"], batch_size=hp["batch_size"],
            lr=hp["lr"], seed=hp["seed"],
            device=device, verbose=True,
        )
        _torch.save(state, ckpt)
        print(f"  [supcon] saved → {ckpt}", flush=True)
    else:
        print(f"  [supcon] cache hit → {ckpt}", flush=True)

    state = _torch.load(ckpt, map_location="cpu")
    loaded, skipped = load_state_dict_compatible(model, state, verbose=False)
    print(f"  [supcon] loaded {len(loaded)} params; skipped {len(skipped)}", flush=True)


def run_cell(
    arm_name: str,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    fold: int,
    kfold_seed: int = 0,
    epochs: int = 30,
    device: str = "cuda",
) -> dict:
    W, D = X.shape[1], X.shape[2]
    X_reg, regime_weeks = slice_regime(X, "full")
    needs = _arm_uses(arm_name)

    spec  = KFoldSpec(family="random", n_folds=5, seed=kfold_seed)
    folds = list(spec.iter_folds(y))
    train_idx, val_idx = folds[fold]
    rng = np.random.default_rng(kfold_seed * 7 + fold)
    es_idx, cal_idx, test_idx = _stratified_three_way_split(
        val_idx, y, fractions=(0.33, 0.33, 0.34), rng=rng
    )

    # ── Step 1: Archetype labels (needed for SupCon, multi-task) ─────────────
    if needs["supcon"] or needs["multitask"]:
        archetype_y = compute_archetype_labels(X_reg, y)
    else:
        archetype_y = None

    # ── Step 2: Engineered features (needed for A24, A29, A30) ──────────────
    # Leakage-safe: feature VALUES are per-customer (each customer's own series,
    # no cross-customer leak), but the z-score standardisation mean/std must be
    # fit on TRAIN customers only — otherwise test-set statistics leak into the
    # normalisation (skill XXIX.3.1 "leakage via normalisation"). Fit on
    # X_reg[train_idx], then apply to all rows.
    if needs["eng_feat"]:
        _, _ef_mean, _ef_std = compute_eng_features_standardised(X_reg[train_idx])
        eng_feat_all, _, _   = compute_eng_features_standardised(
            X_reg, mean=_ef_mean, std=_ef_std,
        )
    else:
        eng_feat_all = None

    # ── Step 3: Peer-feature stacking (input layout) ────────────────────────
    if needs["signed_peer"]:
        peer_block = compute_signed_peer_features(X_reg, train_idx, y, k=20)  # (N, W, 3*D)
        X_aug = np.concatenate([X_reg, peer_block], axis=-1)                  # (N, W, 4*D)
    elif needs["prce_layout"]:
        peer_res = compute_peer_residuals(X_reg, train_idx, y, k=20)
        X_aug = np.concatenate([X_reg, peer_res], axis=-1)                    # (N, W, 2*D)
    else:
        X_aug = X_reg                                                          # (N, W, D)

    # ── Step 4: Training subset extraction ───────────────────────────────────
    train_X = X_aug[train_idx]
    train_y = y[train_idx]
    train_arch_y = archetype_y[train_idx] if archetype_y is not None else None
    train_eng = eng_feat_all[train_idx]    if eng_feat_all is not None else None

    # ── Step 5: Training-set augmentation ────────────────────────────────────
    if needs["sym_aug"]:
        # Phase L: symmetric 5-archetype synthesis. Augmented rows inherit:
        # archetype label by archetype class from sample; eng_feat recomputed
        # from the augmented X (since features are functions of the series).
        aug_rng = np.random.default_rng(seed * 1009 + fold * 31 + kfold_seed)
        normal_idx = np.where(train_y == 0)[0]
        n_to_synth = aug_rng.binomial(n=len(normal_idx), p=0.40)
        if n_to_synth > 0:
            from phased.augment import synthesize_theft_examples
            src = aug_rng.choice(normal_idx, size=n_to_synth, replace=False)
            # Note: synthesize operates on the OWN-channel block; peer & eng feats
            # are recomputed below for augmented rows. We only manipulate own-cons.
            X_synth_own, _ = synthesize_theft_examples(
                train_X[src][..., :D], aug_rng, archetype_weights=SYMMETRIC_ARCHETYPE_WEIGHTS,
            )
            # Reuse the source row's peer block unchanged (peers are normal-only
            # so they're representative for the augmented sample too). Eng feat
            # is recomputed from the augmented own series.
            if needs["signed_peer"] or needs["prce_layout"]:
                peer_part = train_X[src][..., D:]
                X_synth = np.concatenate([X_synth_own, peer_part], axis=-1)
            else:
                X_synth = X_synth_own
            y_synth = np.ones(n_to_synth, dtype=np.int64)

            train_X = np.concatenate([train_X, X_synth], axis=0)
            train_y = np.concatenate([train_y, y_synth], axis=0)

            if train_arch_y is not None:
                # Assign archetype label per synthetic sample (5 classes evenly by source archetype)
                # We sample archetype index in {1..5} according to SYMMETRIC weights.
                arch_choices = np.array([1, 2, 3, 4, 5])   # Heavy_D, Light_D, Flat, Light_R, Heavy_R
                probs = np.array([SYMMETRIC_ARCHETYPE_WEIGHTS[k] for k in
                                  ["heavy_dropper","light_dropper","flat_stealer","light_riser","heavy_riser"]])
                probs = probs / probs.sum()
                synth_arch = aug_rng.choice(arch_choices, size=n_to_synth, p=probs)
                train_arch_y = np.concatenate([train_arch_y, synth_arch])

            if train_eng is not None:
                # Recompute eng-features from augmented OWN series
                synth_eng, _, _ = compute_eng_features_standardised(
                    X_synth_own,
                    mean=_ef_mean if eng_feat_all is not None else None,
                    std =_ef_std  if eng_feat_all is not None else None,
                )
                train_eng = np.concatenate([train_eng, synth_eng], axis=0)

            print(f"  [aug:sym] {arm_name}: {len(X_aug[train_idx])} → {len(train_X)} "
                  f"(theft frac {train_y.mean():.3f})", flush=True)
    elif needs["legacy_aug"]:
        # Phase K backwards-compat: A18 / A21 (Dropper + Flat only — no Risers).
        aug_rng = np.random.default_rng(seed * 1009 + fold * 31 + kfold_seed)
        train_X, train_y = augment_training_set(train_X, train_y, rng=aug_rng, p_aug=0.40)

    # ── Step 6: Build model ──────────────────────────────────────────────────
    seed_everything(seed)
    model = build_arm(arm_name, weeks=regime_weeks, days=D)

    # ── Step 7: Pre-training init ────────────────────────────────────────────
    if needs["supcon"]:
        _supcon_init_model(model, X_reg, archetype_y, regime_weeks, D, device=device)
    elif needs["ssl_init"]:
        _ssl_init_model(model, X_reg, regime_weeks, D, device=device)

    # ── Step 8: Confident-Learning weights ───────────────────────────────────
    if needs["cl"]:
        from phased.conf_learn import compute_cl_weights
        # CL uses only own-consumption (not peer features) — pass first D channels.
        cl_w, _ = compute_cl_weights(
            train_X[..., :D], train_y.astype(np.int64),
            n_splits=3, n_epochs=10, device=device, seed=seed,
        )
    else:
        cl_w = np.ones(len(train_y), dtype=np.float32)

    # ── Step 9: Dispatch to training loop ────────────────────────────────────
    is_phase_l = (needs["multitask"] or needs["eng_feat"]
                   or needs["cl"] or needs["sam_tta"])

    train_overrides = ARM_TRAIN_OVERRIDES.get(arm_name, {})

    t0 = time.time()
    if is_phase_l:
        # Custom Phase L training loop.
        if needs["multitask"]:
            from phased.multitask import MultiTaskZKMixV4
            model_to_train = MultiTaskZKMixV4(model, n_archetypes=6)
        else:
            model_to_train = model

        # ES dataset uses uniform weights and original labels (no CL on val).
        es_arch = archetype_y[es_idx] if archetype_y is not None else None
        es_eng  = eng_feat_all[es_idx] if eng_feat_all is not None else None

        train_ds = PhaseLDataset(
            features=train_X, binary_labels=train_y,
            archetype_labels=train_arch_y if needs["multitask"] else None,
            eng_features=train_eng if needs["eng_feat"] else None,
            weights=cl_w,
        )
        early_ds = PhaseLDataset(
            features=X_aug[es_idx], binary_labels=y[es_idx],
            archetype_labels=es_arch if needs["multitask"] else None,
            eng_features=es_eng if needs["eng_feat"] else None,
            weights=None,
        )

        cfg = MTTrainConfig(
            epochs=epochs, patience=8, batch_size=256, pos_fraction=0.5,
            lr=3e-4, weight_decay=0.01, warmup_frac=0.05, min_lr_ratio=0.01,
            grad_clip_norm=1.0, device=device, seed=seed,
            aux_lambda=0.30, class_balanced=True, cb_beta=0.9999,
            use_sam=needs["sam_tta"], sam_rho=0.05,
            n_archetypes=6,
            **train_overrides,
        )
        train_result, best_model = train_phase_l(
            model_to_train, train_ds, early_ds, cfg,
            is_multitask=needs["multitask"],
            use_eng_feat=needs["eng_feat"],
        )
    else:
        # Standard zkmix_lab training path (A22, A23, A25).
        train_ds = WeeklyGridDataset(train_X, train_y)
        early_ds = WeeklyGridDataset(X_aug[es_idx], y[es_idx])
        cfg = TrainConfig(
            epochs=epochs, patience=10,
            batch_size=128, eval_batch_size=256, pos_fraction=0.5,
            optimizer="AdamW", lr=3e-4, weight_decay=1e-4, eval_every=3,
            warmup_frac=0.05, min_lr_ratio=0.01,
            grad_clip_norm=1.0, device=device,
            **train_overrides,
        )
        train_result, best_model = train(model, train_ds, early_ds, cfg)
    wall = time.time() - t0

    # ── Step 10: Inference ───────────────────────────────────────────────────
    dev = torch.device(device)
    if needs["sam_tta"]:
        # TTA: 5 views, averaged sigmoid. Use own-consumption only for views.
        own_cal  = X_aug[cal_idx][..., :D]
        own_test = X_aug[test_idx][..., :D]
        peer_cal_block  = X_aug[cal_idx][..., D:]  if X_aug.shape[-1] > D else None
        peer_test_block = X_aug[test_idx][..., D:] if X_aug.shape[-1] > D else None
        eng_cal  = eng_feat_all[cal_idx]  if eng_feat_all is not None else None
        eng_test = eng_feat_all[test_idx] if eng_feat_all is not None else None
        cal_scores  = tta_inference(best_model, own_cal,  device=dev,
                                     n_views=5, seed=seed,
                                     peer_features=peer_cal_block,
                                     eng_features=eng_cal)
        test_scores = tta_inference(best_model, own_test, device=dev,
                                     n_views=5, seed=seed,
                                     peer_features=peer_test_block,
                                     eng_features=eng_test)
    else:
        # Standard inference. If MT: use predict_logit; if eng_feat: pass it.
        if needs["multitask"] or needs["eng_feat"]:
            cal_scores  = _phase_l_inference(best_model, X_aug[cal_idx],
                                              eng_feat_all[cal_idx] if eng_feat_all is not None else None,
                                              dev, needs)
            test_scores = _phase_l_inference(best_model, X_aug[test_idx],
                                              eng_feat_all[test_idx] if eng_feat_all is not None else None,
                                              dev, needs)
        else:
            cal_scores  = _inference_batch(best_model, X_aug[cal_idx],  dev)
            test_scores = _inference_batch(best_model, X_aug[test_idx], dev)
    cal_labels  = y[cal_idx]
    test_labels = y[test_idx]

    metrics  = summarise(test_labels, test_scores)
    ece_raw, _ = expected_calibration_error_with_bins(test_scores, test_labels)

    p_train  = max(min(0.5, 1-1e-3), 1e-3)
    p_deploy = max(min(float(y[train_idx].mean()), 1-1e-3), 1e-3)
    cal_out  = calibrate_cell(
        cal_scores=cal_scores, cal_labels=cal_labels,
        test_scores=test_scores, p_train=p_train, p_deploy=p_deploy,
    )
    ece_cal, _ = expected_calibration_error_with_bins(cal_out.test_scores_calibrated, test_labels)

    key = f"{arm_name}__full__random__seed{seed}__fold{fold}__kf{kfold_seed}"
    return {
        "key":          key,
        "arm":          arm_name,
        "regime":       "full",
        "split_family": "random",
        "seed":         seed,
        "fold":         fold,
        "kfold_seed":   kfold_seed,
        "pr_auc":       float(metrics["pr_auc"]),
        "auroc":        float(metrics["auroc"]),
        "f1_optimal":   float(metrics["f1_optimal"]),
        "precision_at_recall": float(metrics.get("precision_at_recall", float("nan"))),
        "ece_raw":      float(ece_raw),
        "ece_calibrated": float(ece_cal),
        "best_val_pr_auc": float(train_result.best_val_pr_auc),
        "best_epoch":   int(train_result.best_epoch),
        "wall_s":       float(wall),
        "n_test":       int(len(test_labels)),
        "val_labels_json":  json.dumps([int(v) for v in test_labels.tolist()]),
        "val_scores_json":  json.dumps([round(s, 6) for s in test_scores.tolist()]),
    }


# ---------------------------------------------------------------------------
# Grid runner (idempotent)
# ---------------------------------------------------------------------------

def run_grid(
    arms: list[str],
    X: np.ndarray,
    y: np.ndarray,
    manifest_path: Path,
    seeds: list[int] | None = None,
    kfold_seeds: list[int] | None = None,
    n_folds: int = 3,
    epochs: int = 30,
    device: str = "cuda",
) -> pd.DataFrame:
    seeds       = seeds       or [0, 1, 2, 3, 4]
    kfold_seeds = kfold_seeds or [0]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[str] = set()
    if manifest_path.exists():
        done_keys = set(pd.read_csv(manifest_path)["key"].tolist())

    total = len(arms) * len(seeds) * n_folds * len(kfold_seeds)
    done_before = len(done_keys)
    i = 0
    for arm in arms:
        for kf in kfold_seeds:
            for seed in seeds:
                for fold in range(n_folds):
                    i += 1
                    key = f"{arm}__full__random__seed{seed}__fold{fold}__kf{kf}"
                    if key in done_keys:
                        print(f"[{i}/{total}] skip {key}", flush=True)
                        continue
                    print(f"[{i}/{total}] {arm} kf={kf} seed={seed} fold={fold} ...",
                          end=" ", flush=True)
                    try:
                        row = run_cell(arm, X, y, seed, fold,
                                       kfold_seed=kf, epochs=epochs, device=device)
                        df_row = pd.DataFrame([row])
                        if manifest_path.exists():
                            df_row.to_csv(manifest_path, mode="a", header=False, index=False)
                        else:
                            df_row.to_csv(manifest_path, index=False)
                        done_keys.add(key)
                        print(f"PR-AUC={row['pr_auc']:.4f}  ep={row['best_epoch']}  "
                              f"wall={row['wall_s']:.0f}s", flush=True)
                    except Exception as e:
                        print(f"ERROR: {e}", flush=True)

    df = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    n_new = len(df) - done_before
    print(f"\n[grid] {n_new} new cells written to {manifest_path}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Verdict computation (paired bootstrap vs Phase C A0)
# ---------------------------------------------------------------------------

def _print_phase_d_summary(df: pd.DataFrame) -> None:
    """Summary stats when Phase C reference is unavailable (e.g. on Kaggle)."""
    if df.empty:
        print("[verdicts] No results yet."); return
    print("\n[verdicts] Phase D/E arm summary (no Phase C reference available):")
    print(df.groupby("arm")["pr_auc"].agg(["count", "mean", "std"]).round(4).to_string())
    a0_rows = df[df["arm"] == "a0_nanative"]["pr_auc"]
    if len(a0_rows) > 0:
        a0_mean = a0_rows.mean()
        print(f"\n[verdicts] A0-nanative reference (in-run): mean={a0_mean:.4f}")
        for arm in sorted(df["arm"].unique()):
            if arm == "a0_nanative": continue
            arm_rows = df[df["arm"] == arm]["pr_auc"]
            if len(arm_rows) > 0:
                d = arm_rows.mean() - a0_mean
                print(f"  {arm}: Δ={d:+.4f}")

def compute_verdicts(manifest_d: pd.DataFrame, phase_c_manifest_path: Path) -> None:
    if not phase_c_manifest_path.exists():
        print(f"\n[verdicts] Phase C manifest not found at {phase_c_manifest_path} — skipping Phase C comparison.")
        _print_phase_d_summary(manifest_d)
        return
    df_c = pd.read_csv(phase_c_manifest_path)
    a0   = df_c[
        (df_c["backbone"] == "zkmix_v3_name_nn") &
        (df_c["regime"]   == "full") &
        (df_c["split_family"] == "random") &
        (df_c["kfold_seed_used"] == 0)
    ].copy()
    print(f"\n[verdicts] Phase C A0 reference: {len(a0)} cells  "
          f"mean={a0.pr_auc.mean():.4f}  std={a0.pr_auc.std():.4f}")

    print("\n" + "="*80)
    print("PHASE D LOCAL GRID VERDICTS")
    print("="*80)

    summary_rows = []
    for arm in sorted(manifest_d["arm"].unique()):
        sub = manifest_d[manifest_d["arm"] == arm]
        mean_d = sub["pr_auc"].mean()
        std_d  = sub["pr_auc"].std()
        print(f"\n{arm}  ({len(sub)} cells)  mean={mean_d:.4f}  std={std_d:.4f}")

        # Pool matched (seed, fold) pairs for bootstrap
        all_y, all_s_d, all_s_c = [], [], []
        for _, row_d in sub.iterrows():
            matched = a0[(a0["seed"] == row_d["seed"]) & (a0["fold"] == row_d["fold"])]
            if matched.empty:
                continue
            row_c = matched.iloc[0]
            y_d   = np.array(json.loads(row_d["val_labels_json"]), dtype=np.int32)
            s_d   = np.array(json.loads(row_d["val_scores_json"]),  dtype=np.float64)
            s_c   = np.array(json.loads(row_c["val_scores_json"]),  dtype=np.float64)
            n     = min(len(y_d), len(s_c))
            all_y.append(y_d[:n]);  all_s_d.append(s_d[:n]);  all_s_c.append(s_c[:n])

        if not all_y:
            print("  No matched A0 cells found — cannot compute bootstrap.")
            continue

        y_pool = np.concatenate(all_y)
        s_d_p  = np.concatenate(all_s_d)
        s_c_p  = np.concatenate(all_s_c)
        res    = paired_bootstrap(y_pool, s_c_p, s_d_p, n_replicates=5000, alpha=0.05)

        verdict = (
            "PASS ✓"     if res.ci_low > 0 and res.margin_point >= 0.01
            else "DIR ↑" if res.ci_low > 0
            else "FAIL ✗" if res.ci_high < 0
            else "NULL —"
        )
        print(f"  vs A0: Δ_point={res.margin_point:+.4f}  "
              f"95%CI=[{res.ci_low:+.4f},{res.ci_high:+.4f}]  {verdict}")
        print(f"  (A0 pooled mean={s_c_p.mean():.4f}  "
              f"A0 PR-AUC matched cells: "
              f"{a0[(a0['seed'].isin(sub['seed'])) & (a0['fold'].isin(sub['fold']))]['pr_auc'].mean():.4f})")

        summary_rows.append({
            "arm": arm, "n_cells": len(sub),
            "pr_auc_mean": round(mean_d, 4), "pr_auc_std": round(std_d, 4),
            "delta": round(res.margin_point, 4),
            "ci_low": round(res.ci_low, 4), "ci_high": round(res.ci_high, 4),
            "verdict": verdict,
        })

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Arm':<28} {'N':>3} {'PR-AUC':>7} {'±std':>6} {'Δ vs A0':>8} "
          f"{'95% CI':>18} {'Verdict'}")
    print("-"*80)
    print(f"{'A0 Phase-C (reference)':<28} {'15':>3} {a0.pr_auc.mean():>7.4f} "
          f"{a0.pr_auc.std():>6.4f} {'—':>8} {'—':>18}")
    for r in summary_rows:
        ci_str = f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
        print(f"{r['arm']:<28} {r['n_cells']:>3} {r['pr_auc_mean']:>7.4f} "
              f"{r['pr_auc_std']:>6.4f} {r['delta']:>+8.4f} {ci_str:>18} {r['verdict']}")

    # Additive decomposition (if all three arms present)
    arms_present = {r["arm"]: r for r in summary_rows}
    if "a1_cc_name_only" in arms_present and "a2_ms_only" in arms_present:
        d_cc = arms_present["a1_cc_name_only"]["delta"]
        d_ms = arms_present["a2_ms_only"]["delta"]
        print(f"\nAdditive decomposition:")
        print(f"  CC-NaME marginal  Δ(A1−A0) = {d_cc:+.4f}")
        print(f"  MS-Lap  marginal  Δ(A2−A0) = {d_ms:+.4f}")
        if "a3_cc_name_ms" in arms_present:
            d_a3 = arms_present["a3_cc_name_ms"]["delta"]
            syn  = d_a3 - (d_cc + d_ms)
            print(f"  Combined          Δ(A3−A0) = {d_a3:+.4f}")
            print(f"  Synergy A3−(A1+A2−A0)     = {syn:+.4f}  "
                  f"({'positive ✓' if syn >= 0 else 'negative — see Stage 2'})")
        if "a6_branch_selective" in arms_present:
            d_a6 = arms_present["a6_branch_selective"]["delta"]
            print(f"  A6 branch-selective         = {d_a6:+.4f}")

    # Save verdict file
    verdict_path = RESULTS_DIR / "local_verdicts.md"
    with open(verdict_path, "w") as f:
        f.write(f"# Phase D Local Grid Verdicts — {pd.Timestamp.now().date()}\n\n")
        f.write("| Arm | N | PR-AUC | ±std | Δ vs A0 | 95% CI | Verdict |\n")
        f.write("|-----|---|--------|------|---------|--------|---------|\n")
        f.write(f"| A0 Phase-C | 15 | {a0.pr_auc.mean():.4f} | "
                f"{a0.pr_auc.std():.4f} | — | — | — |\n")
        for r in summary_rows:
            f.write(f"| {r['arm']} | {r['n_cells']} | {r['pr_auc_mean']:.4f} | "
                    f"{r['pr_auc_std']:.4f} | {r['delta']:+.4f} | "
                    f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] | {r['verdict']} |\n")
    print(f"\nVerdicts written to {verdict_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase D local GPU grid runner.")
    p.add_argument("--arms", nargs="+", default=None,
                   help="Arms to run (default: a1_cc_name_only a2_ms_only a3_cc_name_ms)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds (default: 0 1 2 3 4)")
    p.add_argument("--kfold_seeds", nargs="+", type=int, default=None,
                   help="KFold split seeds (default: [0]; use '0 1 2 3 4' for 75-cell Kaggle run)")
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None,
                   help="Manifest CSV path (default: results/phase_d_local/local_manifest.csv)")
    p.add_argument("--verdicts_only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}" + (f" — {torch.cuda.get_device_name(0)}"
                                   if device == "cuda" else ""), flush=True)

    manifest_path = Path(args.manifest) if args.manifest else MANIFEST_PATH

    if args.verdicts_only:
        if not manifest_path.exists():
            print(f"ERROR: no manifest at {manifest_path}"); return
        df = pd.read_csv(manifest_path)
        compute_verdicts(df, PHASE_C_MANIFEST)
        return

    # Default arms: Stage 1 (A1, A2, A3).  A6 added manually after Stage 1 analysis.
    default_arms = ["a1_cc_name_only", "a2_ms_only", "a3_cc_name_ms"]
    arms  = args.arms  or default_arms
    seeds = args.seeds or [0, 1, 2, 3, 4]

    # Validate arm names (a6_branch_selective registered at import time if available)
    valid = set(ARM_BUILDERS.keys())
    bad   = [a for a in arms if a not in valid]
    if bad:
        print(f"ERROR: unknown arms {bad}.  Valid: {sorted(valid)}")
        return

    # Preprocess data once, share across all cells.
    X, y = load_and_preprocess()

    kfold_seeds = args.kfold_seeds or [0]
    t_start = time.time()
    df = run_grid(arms, X, y, manifest_path,
                  seeds=seeds, kfold_seeds=kfold_seeds, n_folds=args.n_folds,
                  epochs=args.epochs, device=device)
    print(f"\n[total] {time.time()-t_start:.0f}s", flush=True)

    compute_verdicts(df, PHASE_C_MANIFEST)


if __name__ == "__main__":
    main()
