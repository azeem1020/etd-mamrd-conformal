"""Experiment runner — grid + per-cell training + conformal calibration.

A single cell evaluates one (backbone, training-mode, split, seed, fold).
Per-cell metrics written to ``manifest.csv``:

* ranking      : PR-AUC, AUROC, MAP@100, MAP@200
* calibration  : ECE (expected calibration error, 15 bins)
* conformal    : marginal coverage, mean set size, abstain rate
                 (APS @ α=0.10 calibrated on val[:half]; evaluated on val[half:])
* deployment   : wall seconds, peak VRAM (best effort), gate mean (selective only)
* paired       : val indices, val labels, val sigmoid scores
                 (sufficient for the downstream paired-bootstrap aggregator)

The runner is *idempotent*: rerunning ``run_grid`` skips any cell whose
``key`` already appears in the manifest.  This lets Kaggle workers run in
parallel across GPUs against the same manifest path and survive timeouts.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch

from dataclasses import replace as dc_replace

from .calibration import calibrate_cell
from .conformal import (
    APSCalibrator,
    APSResult,
    MondrianAPSCalibrator,
    expected_calibration_error_with_bins,
)
from .data import KFoldSpec, load_sgcc, make_synthetic, slice_regime
from .metrics import paired_bootstrap, pr_auc, summarise
from .models import SelectiveConfig, SelectiveModel, build_model
from .training import TrainConfig, WeeklyGridDataset, seed_everything, train


# ---------------------------------------------------------------------------
# Grid specification
# ---------------------------------------------------------------------------


TrainingMode = Literal["plain", "selective"]


@dataclass(frozen=True)
class Cell:
    """One atomic experiment — (backbone, mode, regime, split, seed, fold).

    ``regime`` selects a weeks-axis slice of the SGCC matrix:
        * "full"      : weeks 0..147   (default)
        * "high_miss" : weeks 0..104   (2014-01 → 2015-12, severe NaNs)
        * "stable"    : weeks 104..147 (2016-01 → 2016-10, low NaNs)

    ``kfold_seed``: if ``-1`` (the historical default), the KFold split
    inherits ``seed`` — every model-init seed produces a *different* fold
    partition.  Setting it to a fixed value (e.g. 0) freezes the partition
    across the seed dimension, which is the precondition for averaging
    sigmoid scores across seeds in :func:`seed_ensemble`.

    Backwards compatibility: ``kfold_seed=-1`` produces the same key as the
    legacy ``Cell`` (no suffix), so the existing 540-cell manifest is
    still readable.  Cells with ``kfold_seed >= 0`` get a ``__kfN`` suffix
    so they cannot accidentally collide with legacy keys.
    """

    backbone: str
    mode: TrainingMode
    regime: str
    split_family: str
    seed: int
    fold: int
    kfold_seed: int = -1
    loss_type: str = "bce"
    # Cheap-ablation knobs (Phase 1 of the post-v3-ablation breakthrough hunt).
    # Defaults preserve legacy keys.  Each non-default value gets a stable
    # textual suffix in ``key()`` so a manifest can mix variants safely.
    label_smoothing: float = 0.0
    cutmix_prob: float = 0.0
    asl_gamma_pos: float = 2.0
    asl_gamma_neg: float = 0.0
    asl_margin: float = 0.0

    def effective_kfold_seed(self) -> int:
        return self.seed if self.kfold_seed < 0 else self.kfold_seed

    def key(self) -> str:
        base = (
            f"{self.backbone}__{self.mode}__{self.regime}__{self.split_family}"
            f"__seed{self.seed}__fold{self.fold}"
        )
        if self.kfold_seed >= 0:
            base += f"__kf{self.kfold_seed}"
        if self.loss_type != "bce":
            base += f"__{self.loss_type}"
        # Defaults: ε=0, cutmix=0, γ⁺=2, γ⁻=0, margin=0 — i.e. legacy ASL.
        # Only emit suffixes when a knob differs from default, so the legacy
        # manifest keys stay bit-identical.
        if self.label_smoothing > 0.0:
            base += f"__ls{self.label_smoothing:g}"
        if self.cutmix_prob > 0.0:
            base += f"__cm{self.cutmix_prob:g}"
        if self.loss_type == "asl":
            if self.asl_gamma_pos != 2.0:
                base += f"__gp{self.asl_gamma_pos:g}"
            if self.asl_gamma_neg != 0.0:
                base += f"__gn{self.asl_gamma_neg:g}"
            if self.asl_margin > 0.0:
                base += f"__m{self.asl_margin:g}"
        return base


@dataclass(frozen=True)
class GridSpec:
    """Default headline grid for the trustworthy-AI framework comparison.

    Scope (post 2026-05-16 expansion to Option 3):
    * plain mode, full regime    : all 10 backbones × 2 splits × 3 seeds × 5 folds = 300
    * plain mode, high_miss      : headline trio          × 2 splits × 3 seeds × 5 folds = 90
    * plain mode, stable         : headline trio          × 2 splits × 3 seeds × 5 folds = 90
    * selective mode, full       : selective trio (2 backbones)  × 2 × 3 × 5  = 60
                                                                       TOTAL  = 540 cells
    """

    backbones: tuple[str, ...] = (
        # Headline + ablation + ZKMix family.
        "zheng_baseline",
        "zkmix_v2_frozen",
        "zkmix_v2_learnable",
        "zkmix_v2_no_kernel_stats",
        # Modern mixers.
        "tsmixer",
        "timemixer",
        # Classical baselines (reviewer-objection coverage).
        "vanilla_cnn",
        "bilstm",
        "tcn",
        "transformer",
    )
    selective_backbones: tuple[str, ...] = (
        "zheng_baseline",
        "zkmix_v2_learnable",
    )
    regime_ablation_backbones: tuple[str, ...] = (
        "zheng_baseline",
        "zkmix_v2_learnable",
        "zkmix_v2_no_kernel_stats",
    )
    extra_regimes: tuple[str, ...] = ("high_miss", "stable")
    split_families: tuple[str, ...] = ("random", "stratified")
    seeds: tuple[int, ...] = (0, 1, 2)
    n_folds: int = 5
    # Breakthrough-attempt extras (2026-05-17).  Default values preserve
    # the legacy 540-cell behaviour.
    kfold_seed: int = -1                       # -1 ⇒ inherit from cell.seed
    loss_type: str = "bce"                     # "bce" | "focal"

    def cells(self) -> list[Cell]:
        out: list[Cell] = []
        kf = self.kfold_seed
        lt = self.loss_type
        # 1) Plain mode, full regime: every backbone.
        for bb, sp, sd in product(self.backbones, self.split_families, self.seeds):
            for f in range(self.n_folds):
                out.append(Cell(bb, "plain", "full", sp, sd, f, kfold_seed=kf, loss_type=lt))
        # 2) Plain mode, extra regimes: only the headline trio.
        for bb, rg, sp, sd in product(
            self.regime_ablation_backbones,
            self.extra_regimes,
            self.split_families,
            self.seeds,
        ):
            for f in range(self.n_folds):
                out.append(Cell(bb, "plain", rg, sp, sd, f, kfold_seed=kf, loss_type=lt))
        # 3) Selective mode, full regime only.
        for bb, sp, sd in product(self.selective_backbones, self.split_families, self.seeds):
            for f in range(self.n_folds):
                out.append(Cell(bb, "selective", "full", sp, sd, f, kfold_seed=kf, loss_type=lt))
        return out


# ---------------------------------------------------------------------------
# Per-cell run
# ---------------------------------------------------------------------------


@dataclass
class CellOutcome:
    cell: Cell
    # Ranking metrics on the test slice.
    pr_auc: float
    auroc: float
    map_100: float | None
    map_200: float | None
    # Operational metrics (single-threshold) on the test slice.
    precision_at_recall_90: float
    f1_optimal: float
    accuracy_at_prior: float
    # Calibration on the test slice.
    ece: float
    # Per-bin reliability diagram payload (list of dicts) — exact contents
    # described in ``conformal.expected_calibration_error_with_bins``.
    reliability_bins: list[dict[str, float]] = field(default_factory=list)
    # Marginal APS (single τ̂).
    aps_alpha: float = float("nan")
    aps_tau: float = float("nan")
    aps_coverage: float = float("nan")
    aps_coverage_pos: float = float("nan")
    aps_coverage_neg: float = float("nan")
    aps_mean_set_size: float = float("nan")
    aps_abstain_rate: float = float("nan")
    # Mondrian APS (per-class τ̂_0, τ̂_1).
    mondrian_tau_neg: float = float("nan")
    mondrian_tau_pos: float = float("nan")
    mondrian_coverage: float = float("nan")
    mondrian_coverage_pos: float = float("nan")
    mondrian_coverage_neg: float = float("nan")
    mondrian_mean_set_size: float = float("nan")
    mondrian_abstain_rate: float = float("nan")
    # Selective head outputs (when mode == "selective").
    gate_mean: float | None = None
    gate_below_05_rate: float | None = None
    # Architecture cost.
    n_params: int = 0
    # Training trajectory — early-stop selected epoch and per-epoch curves.
    best_epoch: int = -1
    best_val_pr_auc: float = float("nan")
    train_loss_history: list[float] = field(default_factory=list)
    val_pr_auc_history: list[float] = field(default_factory=list)
    # Bookkeeping on the test slice (the slice every reported metric refers to).
    n_val: int = 0
    n_pos_val: int = 0
    duration_s: float = 0.0
    infer_ms_per_sample: float = float("nan")
    val_indices: list[int] = field(default_factory=list)
    val_labels: list[int] = field(default_factory=list)
    val_scores: list[float] = field(default_factory=list)
    val_gates: list[float] | None = None
    # Calibration slice — needed for any post-hoc α-sweep coverage analysis.
    cal_indices: list[int] = field(default_factory=list)
    cal_labels: list[int] = field(default_factory=list)
    cal_scores: list[float] = field(default_factory=list)
    # Post-hoc calibration (prior correction + temperature scaling) —
    # fixes the conformal-degeneracy + selective-gate-saturation problem
    # observed in the 540-cell run.  Stored both as the fitted scalars
    # (prior_shift, temperature_T) and as the calibrated probabilities
    # for the test slice, so downstream analysis can recompute any
    # trustworthy-AI metric without re-running.
    prior_shift_constant: float = float("nan")
    temperature_T: float = float("nan")
    nll_cal_before: float = float("nan")
    nll_cal_after: float = float("nan")
    ece_calibrated: float = float("nan")
    aps_coverage_calibrated: float = float("nan")
    aps_coverage_pos_calibrated: float = float("nan")
    aps_coverage_neg_calibrated: float = float("nan")
    aps_mean_set_size_calibrated: float = float("nan")
    aps_abstain_rate_calibrated: float = float("nan")
    mondrian_coverage_calibrated: float = float("nan")
    mondrian_coverage_pos_calibrated: float = float("nan")
    mondrian_coverage_neg_calibrated: float = float("nan")
    mondrian_mean_set_size_calibrated: float = float("nan")
    val_scores_calibrated: list[float] = field(default_factory=list)

    def manifest_row(self) -> dict[str, object]:
        return {
            "key": self.cell.key(),
            "backbone": self.cell.backbone,
            "mode": self.cell.mode,
            "regime": self.cell.regime,
            "split_family": self.cell.split_family,
            "seed": self.cell.seed,
            "fold": self.cell.fold,
            "pr_auc": self.pr_auc,
            "auroc": self.auroc,
            "map_100": self.map_100,
            "map_200": self.map_200,
            "precision_at_recall_90": self.precision_at_recall_90,
            "f1_optimal": self.f1_optimal,
            "accuracy_at_prior": self.accuracy_at_prior,
            "ece": self.ece,
            "aps_alpha": self.aps_alpha,
            "aps_tau": self.aps_tau,
            "aps_coverage": self.aps_coverage,
            "aps_coverage_pos": self.aps_coverage_pos,
            "aps_coverage_neg": self.aps_coverage_neg,
            "aps_mean_set_size": self.aps_mean_set_size,
            "aps_abstain_rate": self.aps_abstain_rate,
            "mondrian_tau_neg": self.mondrian_tau_neg,
            "mondrian_tau_pos": self.mondrian_tau_pos,
            "mondrian_coverage": self.mondrian_coverage,
            "mondrian_coverage_pos": self.mondrian_coverage_pos,
            "mondrian_coverage_neg": self.mondrian_coverage_neg,
            "mondrian_mean_set_size": self.mondrian_mean_set_size,
            "mondrian_abstain_rate": self.mondrian_abstain_rate,
            "gate_mean": self.gate_mean,
            "gate_below_05_rate": self.gate_below_05_rate,
            "n_params": self.n_params,
            "best_epoch": self.best_epoch,
            "best_val_pr_auc": self.best_val_pr_auc,
            "n_val": self.n_val,
            "n_pos_val": self.n_pos_val,
            "duration_s": self.duration_s,
            "infer_ms_per_sample": self.infer_ms_per_sample,
            "val_indices_json": json.dumps(self.val_indices),
            "val_labels_json": json.dumps(self.val_labels),
            "val_scores_json": json.dumps([round(s, 6) for s in self.val_scores]),
            "val_gates_json": json.dumps(
                None if self.val_gates is None else [round(g, 6) for g in self.val_gates]
            ),
            "cal_indices_json": json.dumps(self.cal_indices),
            "cal_labels_json": json.dumps(self.cal_labels),
            "cal_scores_json": json.dumps([round(s, 6) for s in self.cal_scores]),
            "train_loss_history_json": json.dumps(
                [round(v, 6) for v in self.train_loss_history]
            ),
            "val_pr_auc_history_json": json.dumps(
                [round(v, 6) for v in self.val_pr_auc_history]
            ),
            "reliability_bins_json": json.dumps(self.reliability_bins),
            # Post-hoc calibration (prior + temperature scaling).
            "prior_shift_constant": self.prior_shift_constant,
            "temperature_T": self.temperature_T,
            "nll_cal_before": self.nll_cal_before,
            "nll_cal_after": self.nll_cal_after,
            "ece_calibrated": self.ece_calibrated,
            "aps_coverage_calibrated": self.aps_coverage_calibrated,
            "aps_coverage_pos_calibrated": self.aps_coverage_pos_calibrated,
            "aps_coverage_neg_calibrated": self.aps_coverage_neg_calibrated,
            "aps_mean_set_size_calibrated": self.aps_mean_set_size_calibrated,
            "aps_abstain_rate_calibrated": self.aps_abstain_rate_calibrated,
            "mondrian_coverage_calibrated": self.mondrian_coverage_calibrated,
            "mondrian_coverage_pos_calibrated": self.mondrian_coverage_pos_calibrated,
            "mondrian_coverage_neg_calibrated": self.mondrian_coverage_neg_calibrated,
            "mondrian_mean_set_size_calibrated": self.mondrian_mean_set_size_calibrated,
            "val_scores_calibrated_json": json.dumps(
                [round(s, 6) for s in self.val_scores_calibrated]
            ),
            "kfold_seed_used": self.cell.effective_kfold_seed(),
            "loss_type": self.cell.loss_type,
            # Phase 1 cheap-ablation knobs — recorded so downstream analysis
            # can recover the full identity of any cell beyond (backbone,
            # loss_type) without having to re-parse the key string.
            "label_smoothing": self.cell.label_smoothing,
            "cutmix_prob": self.cell.cutmix_prob,
            "asl_gamma_pos": self.cell.asl_gamma_pos,
            "asl_gamma_neg": self.cell.asl_gamma_neg,
            "asl_margin": self.cell.asl_margin,
        }


def _build_for_mode(name: str, mode: TrainingMode, weeks: int, days: int) -> torch.nn.Module:
    backbone = build_model(name, weeks=weeks, days=days)
    if mode == "plain":
        return backbone
    if mode == "selective":
        return SelectiveModel(backbone, cfg=SelectiveConfig())
    raise ValueError(f"Unknown mode {mode!r}.")


def _inference(
    model: torch.nn.Module, X: np.ndarray, device: torch.device, mode: TrainingMode,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(sigmoid_scores, gates_or_None)`` for an array of inputs.

    Batched to keep VRAM stable on a 6 GiB GPU even for 8 k validation rows
    × 1029 timepoints.
    """
    model.eval()
    scores: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    n = X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device, non_blocking=True)
            out = model(xb)
            if isinstance(out, tuple):
                logit, gate = out
                scores.append(torch.sigmoid(logit).cpu().numpy())
                gates.append(gate.cpu().numpy())
            else:
                scores.append(torch.sigmoid(out).cpu().numpy())
    return (np.concatenate(scores), (np.concatenate(gates) if gates else None))


def _stratified_three_way_split(
    val_idx: np.ndarray, y: np.ndarray, fractions: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``val_idx`` into (early_stop, cal, test) preserving the theft
    prior in each piece.  Fractions sum to 1.0; each class is split
    independently by those fractions.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0; got {fractions}.")
    f_es, f_cal, _ = fractions
    pos = val_idx[y[val_idx] == 1]
    neg = val_idx[y[val_idx] == 0]
    pos_perm = rng.permutation(pos)
    neg_perm = rng.permutation(neg)
    n_pos_es, n_pos_cal = int(len(pos) * f_es), int(len(pos) * f_cal)
    n_neg_es, n_neg_cal = int(len(neg) * f_es), int(len(neg) * f_cal)
    es = np.concatenate([pos_perm[:n_pos_es], neg_perm[:n_neg_es]])
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


def run_cell(
    cell: Cell,
    X: np.ndarray,
    y: np.ndarray,
    train_cfg: TrainConfig | None = None,
    weeks: int = 147,
    days: int = 7,
    aps_alpha: float = 0.10,
    val_fractions: tuple[float, float, float] = (0.33, 0.33, 0.34),
    p_train_prior: float | None = None,
    p_deploy_prior: float | None = None,
) -> CellOutcome:
    """Execute one (backbone, mode, regime, split, seed, fold) cell end-to-end.

    Protocol (fixes the test-set-leakage bug present prior to 2026-05-16):

      • Train       : 4 folds.
      • val_idx     : 1 held-out fold, split stratified-three-ways into
                      (early_stop, cal, test) by ``val_fractions``.
      • Early stop  : best epoch chosen on the early_stop slice ONLY.
      • Cal slice   : APS τ̂ fit here.  Mondrian τ̂_0, τ̂_1 fit here too.
      • Test slice  : every reported metric (ranking, calibration,
                      coverage) computed here.  Never seen during
                      training selection or calibration.

    Seed discipline: ``cell.seed`` is set BEFORE the model is built so
    initial weights are deterministic across (re-)runs of the same key.
    """
    base_cfg = train_cfg or TrainConfig(seed=cell.seed)
    # Immutable replace — don't mutate caller's TrainConfig.
    cfg = dc_replace(
        base_cfg,
        selective=(cell.mode == "selective"),
        seed=cell.seed,
        loss_type=cell.loss_type,
        label_smoothing=cell.label_smoothing,
        cutmix_prob=cell.cutmix_prob,
        asl_gamma_pos=cell.asl_gamma_pos,
        asl_gamma_neg=cell.asl_gamma_neg,
        asl_margin=cell.asl_margin,
    )

    # Apply regime slice.
    X_reg, regime_weeks = slice_regime(X, cell.regime)

    spec = KFoldSpec(  # type: ignore[arg-type]
        family=cell.split_family, n_folds=5, seed=cell.effective_kfold_seed(),
    )
    folds = list(spec.iter_folds(y))
    if cell.fold >= len(folds):
        raise IndexError(f"fold {cell.fold} out of range for {len(folds)} splits.")
    train_idx, val_idx = folds[cell.fold]

    # Stratified 3-way split of the validation fold.  Use the EFFECTIVE
    # kfold seed so that two cells differing only in `seed` (model-init)
    # land on identical (early_stop, cal, test) partitions — required for
    # post-hoc seed ensembling.
    rng = np.random.default_rng(cell.effective_kfold_seed() * 7 + cell.fold)
    es_idx, cal_idx, test_idx = _stratified_three_way_split(
        val_idx, y, fractions=val_fractions, rng=rng,
    )

    train_ds = WeeklyGridDataset(X_reg[train_idx], y[train_idx])
    early_stop_ds = WeeklyGridDataset(X_reg[es_idx], y[es_idx])

    # Seed BEFORE the model is built so initial weights are deterministic.
    seed_everything(cfg.seed)
    model = _build_for_mode(cell.backbone, cell.mode, weeks=regime_weeks, days=days)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    t0 = time.time()
    train_result, best_model = train(model, train_ds, early_stop_ds, cfg)
    duration = time.time() - t0

    device = torch.device(cfg.device)
    cal_scores, _ = _inference(best_model, X_reg[cal_idx], device, cell.mode)
    # Time only the test-slice inference — represents deployment cost
    # per sample without contamination from the calibration pass.
    t_inf = time.time()
    test_scores, test_gates = _inference(best_model, X_reg[test_idx], device, cell.mode)
    # On CUDA, _inference already moved outputs to host via .cpu(); the
    # implicit sync there is enough.  Explicit synchronize is a no-op on
    # CPU; we gate it on device.type so we don't crash on a build without
    # CUDA installed.
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_dt = time.time() - t_inf
    infer_ms_per_sample = (
        1000.0 * infer_dt / max(1, len(test_idx))
    )
    cal_labels = y[cal_idx]
    test_labels = y[test_idx]

    # Standard ranking + operational metrics on the test slice.
    metrics = summarise(test_labels, test_scores)
    ece, reliability_bins = expected_calibration_error_with_bins(test_scores, test_labels)

    # APS marginal: fit on cal slice, evaluate on test slice.
    cal_probs_2 = APSCalibrator.to_binary_probs(cal_scores)
    test_probs_2 = APSCalibrator.to_binary_probs(test_scores)
    aps: APSResult = (
        APSCalibrator(alpha=aps_alpha).fit(cal_probs_2, cal_labels)
        .evaluate(test_probs_2, test_labels)
    )

    # Mondrian APS: per-class τ̂, gives class-conditional coverage.
    mondrian: APSResult = (
        MondrianAPSCalibrator(alpha=aps_alpha, n_classes=2)
        .fit(cal_probs_2, cal_labels)
        .evaluate(test_probs_2, test_labels)
    )
    mondrian_tau = mondrian.tau_per_class
    if len(mondrian_tau) != 2:
        mondrian_tau = (float("nan"), float("nan"))

    # Selective-head gate diagnostics.
    if test_gates is not None:
        gate_mean = float(test_gates.mean())
        gate_below_05 = float((test_gates < 0.5).mean())
        val_gates_list: list[float] | None = test_gates.tolist()
    else:
        gate_mean = None
        gate_below_05 = None
        val_gates_list = None

    val_pr_auc_history = [
        float(rec.get("pr_auc", float("nan")))
        for rec in train_result.val_history
    ]

    # ============= Post-hoc calibration =============
    #
    # We trained with BinaryStratifiedSampler at pos_fraction = cfg.pos_fraction
    # (default 0.5 — balanced batches).  The deploy prior on this fold is
    # the empirical positive rate of TRAIN labels (≈ 0.085 on SGCC).  We
    # apply prior correction + temperature scaling to the test slice and
    # recompute every trustworthy-AI metric on the calibrated scores.
    p_train = cfg.pos_fraction if p_train_prior is None else float(p_train_prior)
    p_deploy = (
        float(y[train_idx].mean()) if p_deploy_prior is None else float(p_deploy_prior)
    )
    # Guard the boundaries — a degenerate fold (all-positive or all-negative
    # train slice) would crash prior_shift; fall back to 0.5 / 0.085 then.
    p_train_safe  = max(min(p_train,  1.0 - 1e-3), 1e-3)
    p_deploy_safe = max(min(p_deploy, 1.0 - 1e-3), 1e-3)
    cal_out = calibrate_cell(
        cal_scores=cal_scores,
        cal_labels=cal_labels,
        test_scores=test_scores,
        p_train=p_train_safe,
        p_deploy=p_deploy_safe,
    )
    test_scores_cal = cal_out.test_scores_calibrated

    # Recompute trustworthy-AI artefacts on the calibrated probabilities.
    ece_cal, _ = expected_calibration_error_with_bins(test_scores_cal, test_labels)
    cal_probs_2_cal  = APSCalibrator.to_binary_probs(cal_out.cal_scores_calibrated)
    test_probs_2_cal = APSCalibrator.to_binary_probs(test_scores_cal)
    aps_cal: APSResult = (
        APSCalibrator(alpha=aps_alpha).fit(cal_probs_2_cal, cal_labels)
        .evaluate(test_probs_2_cal, test_labels)
    )
    mondrian_cal: APSResult = (
        MondrianAPSCalibrator(alpha=aps_alpha, n_classes=2)
        .fit(cal_probs_2_cal, cal_labels)
        .evaluate(test_probs_2_cal, test_labels)
    )

    return CellOutcome(
        cell=cell,
        pr_auc=metrics["pr_auc"],
        auroc=metrics["auroc"],
        map_100=metrics.get("map@100"),
        map_200=metrics.get("map@200"),
        precision_at_recall_90=metrics["precision_at_recall"],
        f1_optimal=metrics["f1_optimal"],
        accuracy_at_prior=metrics["accuracy_at_prior"],
        ece=ece,
        reliability_bins=reliability_bins,
        aps_alpha=aps.alpha,
        aps_tau=aps.tau,
        aps_coverage=aps.coverage_marginal,
        aps_coverage_pos=aps.coverage_pos,
        aps_coverage_neg=aps.coverage_neg,
        aps_mean_set_size=aps.mean_set_size,
        aps_abstain_rate=aps.abstain_rate,
        mondrian_tau_neg=float(mondrian_tau[0]),
        mondrian_tau_pos=float(mondrian_tau[1]),
        mondrian_coverage=mondrian.coverage_marginal,
        mondrian_coverage_pos=mondrian.coverage_pos,
        mondrian_coverage_neg=mondrian.coverage_neg,
        mondrian_mean_set_size=mondrian.mean_set_size,
        mondrian_abstain_rate=mondrian.abstain_rate,
        gate_mean=gate_mean,
        gate_below_05_rate=gate_below_05,
        n_params=n_params,
        best_epoch=int(train_result.best_epoch),
        best_val_pr_auc=float(train_result.best_val_pr_auc),
        train_loss_history=list(map(float, train_result.train_history)),
        val_pr_auc_history=val_pr_auc_history,
        n_val=int(len(test_idx)),
        n_pos_val=int(test_labels.sum()),
        duration_s=duration,
        infer_ms_per_sample=float(infer_ms_per_sample),
        val_indices=test_idx.tolist(),
        val_labels=test_labels.tolist(),
        val_scores=test_scores.tolist(),
        val_gates=val_gates_list,
        cal_indices=cal_idx.tolist(),
        cal_labels=cal_labels.tolist(),
        cal_scores=cal_scores.tolist(),
        # Post-hoc calibration.
        prior_shift_constant=float(cal_out.prior_shift),
        temperature_T=float(cal_out.T),
        nll_cal_before=float(cal_out.nll_cal_before),
        nll_cal_after=float(cal_out.nll_cal_after),
        ece_calibrated=float(ece_cal),
        aps_coverage_calibrated=float(aps_cal.coverage_marginal),
        aps_coverage_pos_calibrated=float(aps_cal.coverage_pos),
        aps_coverage_neg_calibrated=float(aps_cal.coverage_neg),
        aps_mean_set_size_calibrated=float(aps_cal.mean_set_size),
        aps_abstain_rate_calibrated=float(aps_cal.abstain_rate),
        mondrian_coverage_calibrated=float(mondrian_cal.coverage_marginal),
        mondrian_coverage_pos_calibrated=float(mondrian_cal.coverage_pos),
        mondrian_coverage_neg_calibrated=float(mondrian_cal.coverage_neg),
        mondrian_mean_set_size_calibrated=float(mondrian_cal.mean_set_size),
        val_scores_calibrated=test_scores_cal.tolist(),
    )


# ---------------------------------------------------------------------------
# Grid driver
# ---------------------------------------------------------------------------


def run_grid(
    out_dir: str | Path,
    spec: GridSpec | None = None,
    train_cfg: TrainConfig | None = None,
    *,
    use_synthetic: bool = False,
    weeks: int = 147,
    days: int = 7,
    only_keys: Iterable[str] | None = None,
    backbone_filter: Iterable[str] | None = None,
    manifest_name: str = "manifest.csv",
    features_csv: str | Path = "data/after_preprocess_data.csv",
    labels_csv: str | Path = "data/label.csv",
) -> Path:
    """Run every cell in ``spec`` and append rows to ``manifest_name``.

    ``backbone_filter`` lets a Kaggle worker take only a subset of cells —
    useful for the two-GPU parallel chain in the launcher notebook.

    ``manifest_name`` lets each Kaggle worker own a private CSV
    (``manifest_gpu0.csv`` / ``manifest_gpu1.csv``) so the two workers
    never contend on the same file handle.  A separate ``merge_manifests``
    helper concatenates them into the canonical ``manifest.csv`` after
    the sweep completes.
    """
    spec = spec or GridSpec()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / manifest_name

    X, y = (
        make_synthetic(weeks=weeks, days=days)
        if use_synthetic
        else load_sgcc(features_csv=features_csv, labels_csv=labels_csv,
                       weeks=weeks, days=days)
    )

    done_keys: set[str] = set()
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        done_keys = set(pd.read_csv(manifest_path)["key"].tolist())

    only = set(only_keys) if only_keys is not None else None
    bb_only = set(backbone_filter) if backbone_filter is not None else None

    for cell in spec.cells():
        if bb_only is not None and cell.backbone not in bb_only:
            continue
        if only is not None and cell.key() not in only:
            continue
        if cell.key() in done_keys:
            continue
        outcome = run_cell(cell, X, y, train_cfg=train_cfg, weeks=weeks, days=days)
        row = outcome.manifest_row()
        pd.DataFrame([row]).to_csv(
            manifest_path, mode="a", header=not manifest_path.exists(), index=False
        )
    return manifest_path


# ---------------------------------------------------------------------------
# Aggregator — paired bootstrap CIs across (a, b) backbone pairs
# ---------------------------------------------------------------------------


def aggregate(
    manifest_path: str | Path,
    pairs: Iterable[tuple[str, str]] | None = None,
    n_replicates: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
    mode_filter: str = "plain",
    regime_filter: str | None = None,
) -> pd.DataFrame:
    """Compute paired bootstrap CIs for every (a, b) backbone pair within
    each (regime, split_family) — restricted to ``mode_filter`` cells.

    ``regime_filter``:
        None    → one row per (regime, split, a, b) covering every regime
                  present in the manifest.
        "full"  → only the full-regime rows (the headline lock target).
    """
    df = pd.read_csv(manifest_path)
    if df.empty:
        raise RuntimeError("Manifest is empty; nothing to aggregate.")
    df = df[df["mode"] == mode_filter].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No rows with mode == {mode_filter!r} in manifest.")
    if regime_filter is not None:
        df = df[df["regime"] == regime_filter].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(f"No rows with regime == {regime_filter!r}.")

    # Distinguish (backbone, loss_type) as the comparison atom when the
    # manifest mixes loss types for the same backbone (e.g. zkmix_v3 with
    # both BCE and focal arms).  This is backwards-compatible: when
    # loss_type is missing or uniformly 'bce', the arm-level key collapses
    # to the backbone name and the legacy behaviour is preserved.
    if "loss_type" not in df.columns:
        df["loss_type"] = "bce"
    df["loss_type"] = df["loss_type"].fillna("bce")
    # Fill cheap-ablation knobs with defaults if absent (legacy manifests).
    for col, default in (
        ("label_smoothing", 0.0),
        ("cutmix_prob",     0.0),
        ("asl_gamma_pos",   2.0),
        ("asl_gamma_neg",   0.0),
        ("asl_margin",      0.0),
    ):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    def _arm_label(r: pd.Series) -> str:
        tag = r["backbone"]
        if r["loss_type"] != "bce":
            tag += f"_{r['loss_type']}"
        if r["label_smoothing"] > 0:
            tag += f"_ls{r['label_smoothing']:g}"
        if r["cutmix_prob"] > 0:
            tag += f"_cm{r['cutmix_prob']:g}"
        if r["loss_type"] == "asl":
            if r["asl_gamma_pos"] != 2.0:
                tag += f"_gp{r['asl_gamma_pos']:g}"
            if r["asl_gamma_neg"] != 0.0:
                tag += f"_gn{r['asl_gamma_neg']:g}"
            if r["asl_margin"] > 0.0:
                tag += f"_m{r['asl_margin']:g}"
        return tag
    df["_arm"] = df.apply(_arm_label, axis=1)

    if pairs is None:
        arms = sorted(df["_arm"].unique())
        pair_keys = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1 :]]
    else:
        # Caller may pass either backbone names or arm names; treat as arm names.
        pair_keys = list(pairs)

    out: list[dict[str, object]] = []
    for regime in sorted(df["regime"].unique()):
        df_r = df[df["regime"] == regime]
        for split_family in sorted(df_r["split_family"].unique()):
            sub = df_r[df_r["split_family"] == split_family]
            for a, b in pair_keys:
                sub_a = sub[sub["_arm"] == a]
                sub_b = sub[sub["_arm"] == b]
                if sub_a.empty or sub_b.empty:
                    continue
                labels_pool: list[int] = []
                scores_a_pool: list[float] = []
                scores_b_pool: list[float] = []
                sub_b_idx = sub_b.set_index(["seed", "fold"]).sort_index()
                for (sd, fd), row_a in sub_a.set_index(["seed", "fold"]).sort_index().iterrows():
                    if (sd, fd) not in sub_b_idx.index:
                        continue
                    row_b = sub_b_idx.loc[(sd, fd)]
                    # A duplicate (seed, fold) within the same arm is a manifest bug
                    # — should never happen now that the arm key includes loss_type.
                    if isinstance(row_b, pd.DataFrame):
                        continue
                    if json.loads(row_a["val_indices_json"]) != json.loads(row_b["val_indices_json"]):
                        continue
                    labels_pool.extend(json.loads(row_a["val_labels_json"]))
                    scores_a_pool.extend(json.loads(row_a["val_scores_json"]))
                    scores_b_pool.extend(json.loads(row_b["val_scores_json"]))
                if len(labels_pool) == 0:
                    continue
                y_arr = np.asarray(labels_pool)
                a_arr = np.asarray(scores_a_pool)
                b_arr = np.asarray(scores_b_pool)
                res = paired_bootstrap(
                    y_arr, a_arr, b_arr,
                    n_replicates=n_replicates, alpha=alpha, seed=seed,
                )
                # Decompose the arm back into (backbone, loss_type) for downstream.
                row_a0 = sub_a.iloc[0]; row_b0 = sub_b.iloc[0]
                out.append({
                    "arm_a": a,
                    "arm_b": b,
                    "backbone_a": row_a0["backbone"],
                    "loss_type_a": row_a0["loss_type"],
                    "backbone_b": row_b0["backbone"],
                    "loss_type_b": row_b0["loss_type"],
                    "regime": regime,
                    "split_family": split_family,
                    "mode": mode_filter,
                    "metric_a_point": float(pr_auc(y_arr, a_arr)),
                    "metric_b_point": float(pr_auc(y_arr, b_arr)),
                    "delta_point": res.margin_point,
                    "delta_mean": res.delta_mean,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "n_replicates": res.n_replicates,
                })
    return pd.DataFrame(out)


def conformal_summary(manifest_path: str | Path) -> pd.DataFrame:
    """Per-(backbone, mode, regime, split) coverage + set size + ECE.

    Marginal coverage should be ≥ 0.90 = 1 − α on every row.  Mondrian
    columns track class-conditional coverage; the Mondrian guarantee is
    that ``mondrian_coverage_pos`` and ``mondrian_coverage_neg`` are
    each ≥ 0.90 separately (rather than only the marginal average).
    """
    df = pd.read_csv(manifest_path)
    g = df.groupby(["backbone", "mode", "regime", "split_family"], as_index=False).agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        auroc_mean=("auroc", "mean"),
        precision_at_recall_90_mean=("precision_at_recall_90", "mean"),
        f1_optimal_mean=("f1_optimal", "mean"),
        ece_mean=("ece", "mean"),
        aps_coverage_mean=("aps_coverage", "mean"),
        aps_coverage_pos_mean=("aps_coverage_pos", "mean"),
        aps_coverage_neg_mean=("aps_coverage_neg", "mean"),
        aps_mean_set_size_mean=("aps_mean_set_size", "mean"),
        aps_abstain_rate_mean=("aps_abstain_rate", "mean"),
        mondrian_coverage_mean=("mondrian_coverage", "mean"),
        mondrian_coverage_pos_mean=("mondrian_coverage_pos", "mean"),
        mondrian_coverage_neg_mean=("mondrian_coverage_neg", "mean"),
        mondrian_mean_set_size_mean=("mondrian_mean_set_size", "mean"),
        mondrian_abstain_rate_mean=("mondrian_abstain_rate", "mean"),
        n_params_mean=("n_params", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        infer_ms_per_sample_mean=("infer_ms_per_sample", "mean"),
        n_cells=("pr_auc", "count"),
    )
    return g.round(4)


# ---------------------------------------------------------------------------
# Per-worker manifest merge
# ---------------------------------------------------------------------------


def merge_manifests(
    out_dir: str | Path,
    pattern: str = "manifest_gpu*.csv",
    combined_name: str = "manifest.csv",
) -> Path:
    """Concatenate every ``manifest_gpu*.csv`` in ``out_dir`` into one CSV.

    Duplicates on ``key`` are dropped, keeping the first occurrence.  This
    is the canonical way to coalesce the parallel-worker outputs after a
    Kaggle 2× T4 sweep so the rest of the analysis pipeline can keep
    pointing at a single ``manifest.csv``.

    Raises if no per-worker files exist (the caller would otherwise carry
    on against an empty combined file).
    """
    out_dir = Path(out_dir)
    parts = sorted(out_dir.glob(pattern))
    if not parts:
        raise FileNotFoundError(
            f"No files matching {pattern!r} under {out_dir}; "
            "did any worker write a manifest?"
        )
    frames = [pd.read_csv(p) for p in parts]
    combined = pd.concat(frames, ignore_index=True)
    if "key" in combined.columns:
        combined = combined.drop_duplicates(subset=["key"], keep="first")
    combined_path = out_dir / combined_name
    combined.to_csv(combined_path, index=False)
    return combined_path


# ---------------------------------------------------------------------------
# Selective head: gate-threshold sweep
# ---------------------------------------------------------------------------


def selective_curve(
    manifest_path: str | Path,
    *,
    gate_thresholds: Iterable[float] = (
        0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    ),
    regime: str = "full",
) -> pd.DataFrame:
    """For each (backbone, split_family) selective row, sweep the gate
    threshold and report ``(coverage, pr_auc_on_selected)``.

    Definition of coverage here: fraction of test examples whose gate
    output ≥ threshold (i.e. *not* abstained).  PR-AUC is computed on
    the selected subset only — abstained examples are removed.

    Returns one row per (backbone, split_family, seed, fold, threshold).
    Aggregate downstream with groupby for plotting.
    """
    df = pd.read_csv(manifest_path)
    df = df[(df["mode"] == "selective") & (df["regime"] == regime)]
    if df.empty:
        raise RuntimeError(
            f"No selective rows for regime={regime!r} in {manifest_path}."
        )
    out: list[dict[str, object]] = []
    for _, row in df.iterrows():
        gates = json.loads(row["val_gates_json"])
        if gates is None:
            continue
        labels = np.asarray(json.loads(row["val_labels_json"]))
        scores = np.asarray(json.loads(row["val_scores_json"]))
        gates_arr = np.asarray(gates)
        for thr in gate_thresholds:
            mask = gates_arr >= float(thr)
            coverage = float(mask.mean())
            sub_y = labels[mask]
            sub_s = scores[mask]
            # Need at least one positive AND one negative on the selected
            # subset to define PR-AUC.
            if sub_y.size == 0 or sub_y.sum() == 0 or sub_y.sum() == sub_y.size:
                pra = float("nan")
            else:
                pra = float(pr_auc(sub_y, sub_s))
            out.append({
                "backbone": row["backbone"],
                "split_family": row["split_family"],
                "regime": regime,
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "gate_threshold": float(thr),
                "coverage": coverage,
                "pr_auc_selected": pra,
                "n_selected": int(mask.sum()),
                "n_pos_selected": int(sub_y.sum()),
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Seed ensemble — average sigmoid scores across model-init seeds
# ---------------------------------------------------------------------------


def seed_ensemble(
    manifest_path: str | Path,
    *,
    use_calibrated: bool = True,
    regime: str | None = None,
    mode_filter: str = "plain",
) -> pd.DataFrame:
    """Average per-example sigmoid scores across model-init seeds.

    Precondition (enforced by ``val_indices_json`` equality)
    --------------------------------------------------------
    Two cells can only be ensembled if they evaluated on the **same test
    examples**.  This holds iff they share ``(backbone, mode, regime,
    split_family, fold, kfold_seed)`` and differ only in ``seed`` (the
    model-init seed).  Cells with ``kfold_seed=-1`` (the legacy default)
    use ``seed`` to drive the KFold split too — so different seeds give
    different val partitions and **cannot** be ensembled.  Such groups
    are silently skipped (PR-AUC = NaN in the result).

    Returns
    -------
    DataFrame with one row per (backbone, mode, regime, split_family,
    fold, kfold_seed) — the ensembled PR-AUC, AUROC, and number of seeds
    that contributed.

    ``use_calibrated``: when True, ensemble the calibrated sigmoid scores
    (post prior-shift + temperature scaling) instead of the raw model
    outputs.  Ensembling-after-calibration is generally better because
    the calibrated probabilities live on a comparable scale across seeds.
    """
    df = pd.read_csv(manifest_path)
    df = df[df["mode"] == mode_filter]
    if regime is not None:
        df = df[df["regime"] == regime]
    if df.empty:
        raise RuntimeError(
            f"No rows matching mode={mode_filter!r} regime={regime!r} in manifest."
        )

    score_col = "val_scores_calibrated_json" if use_calibrated else "val_scores_json"
    if score_col not in df.columns:
        raise RuntimeError(
            f"Manifest is missing column {score_col!r}; was it produced by a "
            "pre-calibration-fix run?  Re-run with the current runner."
        )

    # `kfold_seed_used` is filled in by manifest_row; legacy rows lack it.
    if "kfold_seed_used" not in df.columns:
        df["kfold_seed_used"] = -1
    # `loss_type` distinguishes (e.g.) zkmix_v3 with BCE from zkmix_v3
    # with focal — they must NOT be ensembled together.
    if "loss_type" not in df.columns:
        df["loss_type"] = "bce"
    df["loss_type"] = df["loss_type"].fillna("bce")

    group_cols = ["backbone", "loss_type", "mode", "regime", "split_family", "fold", "kfold_seed_used"]
    out: list[dict[str, object]] = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if len(grp) < 2:
            continue
        # Require identical val_indices across the seeds being averaged.
        idx_sets = grp["val_indices_json"].unique()
        if len(idx_sets) != 1:
            # Different val partitions ⇒ can't ensemble.  Skip silently.
            continue
        val_idx = json.loads(idx_sets[0])
        labels = np.asarray(json.loads(grp["val_labels_json"].iloc[0]))
        scores = np.stack(
            [np.asarray(json.loads(s)) for s in grp[score_col].to_list()],
            axis=0,
        )
        if np.isnan(scores).any():
            # Calibrated column may carry NaNs on a degenerate cal slice.
            continue
        ensembled = scores.mean(axis=0)
        # Need both classes present to define PR-AUC.
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        out.append({
            "backbone":        keys[0],
            "loss_type":       keys[1],
            "mode":            keys[2],
            "regime":          keys[3],
            "split_family":    keys[4],
            "fold":            int(keys[5]),
            "kfold_seed":      int(keys[6]),
            "n_seeds":         int(len(grp)),
            "n_test":          int(len(labels)),
            "pr_auc_ensemble": float(pr_auc(labels, ensembled)),
            "pr_auc_seed_mean":float(grp["pr_auc"].mean()),
            "pr_auc_seed_std": float(grp["pr_auc"].std()),
        })
    return pd.DataFrame(out)


__all__ = [
    "Cell",
    "CellOutcome",
    "GridSpec",
    "run_cell",
    "run_grid",
    "aggregate",
    "conformal_summary",
    "merge_manifests",
    "selective_curve",
    "seed_ensemble",
]
