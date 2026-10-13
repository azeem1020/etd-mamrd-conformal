"""SCT-ETD — Selective Conformal Electricity-Theft Detection.

A PhD-grade extension of Zheng et al. 2018 (IEEE TII) Wide-and-Deep CNN,
re-cast as a selective multi-class problem with conformal coverage guarantees
and a cost-aware operating policy.

Submodules
----------
data            Leakage-aware splits, 5-class soft relabelling, regime detector.
representation  Mask-aware 3-channel encoder + anti-mean kernel (preserves Zheng S3).
models          Wide-Deep-Patch encoder, 5-way selective head, domain-adversarial trunk.
calibration     Isotonic + temperature + Adaptive Prediction Sets (APS) conformal.
policy          Net-value optimiser + active-learning re-queue.
eval            Selective metrics (RC-curve, AURC), conformal coverage audit, saliency.

See PAPER1_ENERGY_AI_PLAN.md for the Energy & AI submission carve-out.
"""
__version__ = "0.0.1"
