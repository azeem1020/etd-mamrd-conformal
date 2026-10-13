"""Integration tests for ZKMixV4.

Tests verified:
  G1. Scalar fallback (use_cc_name=False, dilations=(1,)) ≈ Phase C NaME-NN structure.
  G2. CC-NaME at init is bit-identical to scalar NaME.
  G3. Output shape and no-NaN guarantee.
  G4. All arm builders produce correct shapes.
  G5. Gradient flows to all trainable parameters.
  G6. Token interpretability accessor returns correct shape.
  G7. model.embed() and model.forward() are consistent.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import torch.nn as nn

from phased.models import (
    ZKMixV4, ZKMixV4Config,
    build_a1, build_a2, build_a3, build_a4, build_a5,
    build_arm, ARM_BUILDERS,
)

W, D = 147, 7
torch.manual_seed(0)


def _batch(n: int = 4, nan_frac: float = 0.35, d_mult: int = 1) -> torch.Tensor:
    x = torch.randn(n, W, D * d_mult)
    mask = torch.rand(n, W, D * d_mult) < nan_frac
    x[mask] = float("nan")
    return x


# ---------------------------------------------------------------------------
# G1 / G2: Equivalence guarantees
# ---------------------------------------------------------------------------

class TestEquivalenceGuarantees:

    def test_g2_cc_name_init_equals_scalar(self):
        """G2: At init, CC-NaME enabled must produce same forward as scalar NaME.

        The context encoder output layer is zero-inited → δτ=0 → τ_n = base_token.

        Because m_cc has extra encoder parameters that consume RNG state during
        __init__, we can't rely on the same seed producing identical shared-layer
        weights.  Instead we build both models with the same seed, then explicitly
        copy the shared (non-NaME) weights from scalar to cc, ensuring the only
        difference is the NaME module itself.
        """
        cfg_cc     = ZKMixV4Config(use_cc_name=True,  use_ms_kernel=False, ms_dilations=(1,))
        cfg_scalar = ZKMixV4Config(use_cc_name=False, use_ms_kernel=False, ms_dilations=(1,))
        torch.manual_seed(5)
        m_cc = ZKMixV4(cfg_cc)
        torch.manual_seed(5)
        m_scalar = ZKMixV4(cfg_scalar)

        # Explicitly copy all shared non-NaME parameters.
        shared_keys = [k for k in m_scalar.state_dict() if not k.startswith("name_module")]
        sd_cc = m_cc.state_dict()
        sd_sc = m_scalar.state_dict()
        with torch.no_grad():
            for k in shared_keys:
                sd_cc[k].copy_(sd_sc[k])
        m_cc.load_state_dict(sd_cc)

        x = _batch()
        m_cc.eval(); m_scalar.eval()
        with torch.no_grad():
            out_cc     = m_cc(x)
            out_scalar = m_scalar(x)
        assert torch.allclose(out_cc, out_scalar, atol=1e-5, rtol=0), \
            "CC-NaME at init must produce same output as scalar NaME (shared weights matched)."

    def test_single_scale_matches_structure(self):
        """Single-scale (dilations=(1,)) must have same channel dim as Phase C CNN branch."""
        cfg = ZKMixV4Config(use_ms_kernel=False, ms_dilations=(1,))
        m   = ZKMixV4(cfg)
        # The CNN branch must output beta=60 dim.
        x4d = torch.randn(2, 1, W, D)
        out = m.cnn_branch(x4d)
        assert out.shape == (2, cfg.beta), \
            f"Single-scale CNN branch must output (N, beta={cfg.beta}); got {tuple(out.shape)}."


# ---------------------------------------------------------------------------
# G3: Output shape and no-NaN
# ---------------------------------------------------------------------------

class TestOutputContract:

    def test_forward_shape(self):
        m = build_a3()
        x = _batch()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (4,), f"Forward must return (N,); got {tuple(out.shape)}."

    def test_no_nan_in_output(self):
        m = build_a3()
        x = _batch(nan_frac=0.5)
        with torch.no_grad():
            out = m(x)
        assert torch.isfinite(out).all(), "Model output must be finite even with 50% NaN input."

    def test_embed_shape(self):
        m   = build_a3()
        x   = _batch()
        emb = m.embed(x)
        assert emb.shape == (4, m.embed_dim), \
            f"embed() must return (N, {m.embed_dim}); got {tuple(emb.shape)}."

    def test_embed_forward_consistency(self):
        """forward(x) must equal head(fuse(embed(x))).

        Must be tested in eval() mode so Dropout is deterministic.
        """
        m = build_a3()
        m.eval()
        x = _batch()
        with torch.no_grad():
            out_forward = m(x)
            out_manual  = m.head(m.fuse(m.embed(x))).squeeze(-1)
        assert torch.allclose(out_forward, out_manual, atol=1e-5), \
            "forward() must equal head(fuse(embed(x)))."

    def test_all_arms_correct_shape(self):
        for name in ARM_BUILDERS:
            try:
                m   = build_arm(name)
            except ImportError:
                continue  # Requires compiled mamba_ssm kernel
            cfg = getattr(m, 'cfg', None)
            d_mult = 1
            if cfg is not None:
                if getattr(cfg, 'use_signed_peer', False):
                    d_mult = 4
                elif getattr(cfg, 'use_prce', False):
                    d_mult = 2
            x   = _batch(d_mult=d_mult)
            eng = torch.randn(4, cfg.eng_features_n) if (cfg is not None and getattr(cfg, 'use_eng_features', False)) else None
            out = m(x, eng_feat=eng) if eng is not None else m(x)
            assert out.shape == (4,), \
                f"Arm {name}: forward must return (N,); got {tuple(out.shape)}."

    def test_wrong_input_dims_raises(self):
        m = build_a3()
        with pytest.raises(ValueError):
            m(torch.randn(4, W, D, 1))   # 4D input, not 3D


# ---------------------------------------------------------------------------
# G4: All arm builders
# ---------------------------------------------------------------------------

class TestArmBuilders:

    def test_a1_cc_name_only_has_scalar_ms(self):
        """A1 should use CC-NaME and single-scale (1,) Laplacian."""
        m = build_a1()
        assert m.cfg.use_cc_name is True
        assert tuple(m.cfg.ms_dilations) == (1,) or not m.cfg.use_ms_kernel

    def test_a2_ms_only_has_scalar_name(self):
        """A2 should use scalar NaME and multi-scale Laplacian."""
        m = build_a2()
        assert m.cfg.use_cc_name is False
        assert m.cfg.use_ms_kernel is True

    def test_a3_full_phase_d(self):
        """A3 must have both CC-NaME and multi-scale enabled."""
        m = build_a3()
        assert m.cfg.use_cc_name is True
        assert m.cfg.use_ms_kernel is True
        assert 4 in m.cfg.ms_dilations

    def test_a4_cc_name_single_scale(self):
        """A4: CC-NaME + d=(1,) — confirms multi-scale contribution when vs A3."""
        m = build_a4()
        assert m.cfg.use_cc_name is True
        assert 2 not in m.cfg.ms_dilations and 4 not in m.cfg.ms_dilations

    def test_a5_no_wide_branch(self):
        """A5 must not include wide branch output in embed dim."""
        m = build_a5()
        # embed_dim for A5 = beta + mixer_out_dim (no alpha contribution)
        expected_dim = ZKMixV4Config().beta + ZKMixV4Config().mixer_out_dim
        assert m.embed_dim == expected_dim, \
            f"A5 embed_dim should be {expected_dim}; got {m.embed_dim}."


# ---------------------------------------------------------------------------
# G5: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_all_trainable_params_receive_gradient(self):
        """Every parameter with requires_grad=True must have non-None grad."""
        m = build_a3()
        # Break zero-init in CC-NaME encoder.
        with torch.no_grad():
            m.name_module.encoder.net[-1].weight.fill_(0.01)
        x = _batch(n=8, nan_frac=0.3)
        m(x).sum().backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Parameter {name} has no gradient."

    def test_frozen_stencils_no_gradient(self):
        """Frozen stencil weights must have requires_grad=False."""
        m = build_a3()
        for i, branch in enumerate(m.cnn_branch.branches):
            assert not branch.stencil.weight.requires_grad, \
                f"Scale branch {i} stencil must be frozen."


# ---------------------------------------------------------------------------
# G6: Token interpretability
# ---------------------------------------------------------------------------

class TestTokenInterpretability:

    def test_cc_name_tokens_shape(self):
        m = build_a3()
        x = _batch()
        with torch.no_grad():
            tokens = m.cc_name_tokens(x)
        assert tokens.shape == (4,), \
            f"cc_name_tokens must return (N,); got {tuple(tokens.shape)}."

    def test_scalar_fallback_tokens_constant(self):
        """For scalar NaME (cc_name=False), all tokens should equal base_token."""
        m = build_a2()   # A2: scalar NaME
        x = _batch()
        with torch.no_grad():
            tokens = m.cc_name_tokens(x)
        base = m.name_module.base_token.item()
        assert torch.allclose(tokens, torch.full_like(tokens, base), atol=1e-6), \
            "Scalar NaME tokens must all equal base_token."


# ---------------------------------------------------------------------------
# Smoke: forward pass on CPU and (if available) CUDA
# ---------------------------------------------------------------------------

class TestDeviceSmoke:

    def test_cpu_forward(self):
        m = build_a3().cpu()
        x = _batch()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (4,)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available.")
    def test_cuda_forward(self):
        m = build_a3().cuda()
        x = _batch().cuda()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (4,)
        assert out.device.type == "cuda"
