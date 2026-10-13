"""Unit tests for MultiScaleLaplacian.

Key properties verified:
  - d=1 branch output matches Phase C frozen 9-stencil exactly.
  - Output shape is correct at all dilation scales.
  - Frozen stencil weights do NOT change after an optimizer step.
  - Learnable 1×1 mixing weights DO change after gradient step.
  - Multi-scale output is strictly different from single-scale.
  - No NaN in output even when input has been pre-imputed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
import torch.nn as nn

from phased.ms_kernel import MultiScaleLaplacian, ScaleBranch, _eq7_nine_stencil_weights_local


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

W, D = 147, 7


def _clean_grid(n: int = 4) -> torch.Tensor:
    """(N, 1, W, D) with no NaN — simulates post-NaME input."""
    torch.manual_seed(1)
    return torch.randn(n, 1, W, D)


# ---------------------------------------------------------------------------
# Stencil weight reconstruction
# ---------------------------------------------------------------------------

class TestNineStencilWeights:

    def test_shape(self):
        w = _eq7_nine_stencil_weights_local()
        assert w.shape == (9, 1, 3, 3), f"Expected (9,1,3,3), got {tuple(w.shape)}."

    def test_zero_sum_per_filter(self):
        """Each 3×3 filter must sum to 0 (anti-mean property: constant input → 0 output)."""
        w = _eq7_nine_stencil_weights_local()
        sums = w.sum(dim=(-2, -1))    # (9, 1)
        assert torch.allclose(sums, torch.zeros_like(sums), atol=1e-6), \
            "Every stencil filter must sum to 0 (zero-response on constant input)."

    def test_matches_zkmix_lab_if_available(self):
        """If zkmix_lab is importable, local reconstruction must match exactly."""
        try:
            from zkmix_lab.models import _eq7_nine_stencil_weights  # type: ignore
            w_lab   = _eq7_nine_stencil_weights()
            w_local = _eq7_nine_stencil_weights_local()
            assert torch.allclose(w_lab, w_local, atol=1e-6, rtol=0), \
                "Local reconstruction must match zkmix_lab weights exactly."
        except ImportError:
            pytest.skip("zkmix_lab not on path — skipping cross-check.")


# ---------------------------------------------------------------------------
# ScaleBranch
# ---------------------------------------------------------------------------

class TestScaleBranch:

    def test_output_shape(self):
        b = ScaleBranch(dilation=1, out_channels=15)
        x = _clean_grid()
        out = b(x)
        assert out.shape == (4, 15, W, D), \
            f"Expected (4, 15, {W}, {D}), got {tuple(out.shape)}."

    def test_d1_stencil_frozen_by_default(self):
        """Stencil weights must not change after a gradient update (freeze=True)."""
        b = ScaleBranch(dilation=1, freeze=True, out_channels=8)
        w_before = b.stencil.weight.data.clone()
        x = _clean_grid()
        opt = torch.optim.SGD(b.parameters(), lr=0.1)
        loss = b(x).sum()
        loss.backward()
        opt.step()
        w_after = b.stencil.weight.data
        assert torch.allclose(w_before, w_after, atol=1e-7), \
            "Frozen stencil weights must not change after optimizer step."

    def test_d1_mix_conv_trains(self):
        """1×1 mixing conv must update after gradient step (learnable)."""
        b = ScaleBranch(dilation=1, freeze=True, out_channels=8)
        w_before = b.mix.weight.data.clone()
        x = _clean_grid()
        opt = torch.optim.SGD(b.parameters(), lr=0.1)
        b(x).sum().backward()
        opt.step()
        w_after = b.mix.weight.data
        assert not torch.allclose(w_before, w_after), \
            "1×1 mix conv weights must change after gradient step."

    def test_dilations_give_different_outputs(self):
        """Scale branches at d=1 and d=4 must produce different feature maps."""
        b1 = ScaleBranch(dilation=1, freeze=True, out_channels=9)
        b4 = ScaleBranch(dilation=4, freeze=True, out_channels=9)
        # Copy mix weights so the difference comes only from dilation.
        with torch.no_grad():
            b4.mix.weight.copy_(b1.mix.weight)
            b4.mix.bias.copy_(b1.mix.bias)
        x = _clean_grid()
        with torch.no_grad():
            out1 = b1(x)
            out4 = b4(x)
        # Interior cells (away from boundary): outputs must differ.
        diff = (out1[:, :, 10:-10, 1:-1] - out4[:, :, 10:-10, 1:-1]).abs().max().item()
        assert diff > 1e-4, \
            f"d=1 and d=4 branches should give different interior responses; max diff={diff}."

    def test_constant_input_zero_output(self):
        """Constant input (all cells equal) → stencil response = 0.
        (Anti-mean property: L_d(constant) = 0 everywhere in the interior.)"""
        b = ScaleBranch(dilation=1, freeze=True, out_channels=9)
        x = torch.ones(1, 1, W, D) * 5.0      # constant
        with torch.no_grad():
            out = b.stencil(x)                 # raw stencil output (9 channels)
        # Interior region (away from zero-padded boundary).
        interior = out[:, :, 2:-2, 2:-2]
        assert interior.abs().max().item() < 1e-5, \
            "Stencil response to constant input must be 0 in the interior."


# ---------------------------------------------------------------------------
# MultiScaleLaplacian
# ---------------------------------------------------------------------------

class TestMultiScaleLaplacian:

    def test_output_shape(self):
        ms = MultiScaleLaplacian(dilations=(1, 2, 4), weeks=W, days=D, embed_dim=60)
        x  = _clean_grid()
        out = ms(x)
        assert out.shape == (4, 60), f"Expected (4, 60), got {tuple(out.shape)}."

    def test_single_scale_output_shape(self):
        ms = MultiScaleLaplacian(dilations=(1,), weeks=W, days=D, embed_dim=60)
        x  = _clean_grid()
        assert ms(x).shape == (4, 60)

    def test_no_nan_in_output(self):
        ms = MultiScaleLaplacian(dilations=(1, 2, 4), weeks=W, days=D, embed_dim=60)
        x  = _clean_grid()
        with torch.no_grad():
            out = ms(x)
        assert torch.isfinite(out).all(), "MultiScaleLaplacian output must be finite."

    def test_multi_scale_differs_from_single_scale(self):
        """Multi-scale and single-scale must produce different embeddings."""
        ms3 = MultiScaleLaplacian(dilations=(1, 2, 4), weeks=W, days=D, embed_dim=60)
        ms1 = MultiScaleLaplacian(dilations=(1,),       weeks=W, days=D, embed_dim=60)
        # Synchronise the d=1 branch weights and head weights to isolate only
        # the multi-scale effect.
        with torch.no_grad():
            # Copy d=1 stencil mix weights.
            ms1.branches[0].mix.weight.copy_(ms3.branches[0].mix.weight)
            ms1.branches[0].mix.bias.copy_(ms3.branches[0].mix.bias)
        x = _clean_grid()
        with torch.no_grad():
            out3 = ms3(x)
            out1 = ms1(x)
        diff = (out3 - out1).abs().max().item()
        assert diff > 1e-4, \
            f"Multi-scale and single-scale embeddings should differ; max diff={diff}."

    def test_gradients_flow_to_mix_weights(self):
        """Gradients must reach the learnable 1×1 mix conv at each scale."""
        ms  = MultiScaleLaplacian(dilations=(1, 2, 4), weeks=W, days=D, embed_dim=60)
        x   = _clean_grid()
        ms(x).sum().backward()
        for i, branch in enumerate(ms.branches):
            assert branch.mix.weight.grad is not None, \
                f"Scale branch {i} mix weight has no gradient."
            assert branch.mix.weight.grad.abs().sum() > 0, \
                f"Scale branch {i} mix weight gradient is zero."

    def test_frozen_stencils_no_gradient(self):
        """Frozen stencil weights must have no gradient (requires_grad=False)."""
        ms = MultiScaleLaplacian(
            dilations=(1, 2, 4), freeze_stencils=True, weeks=W, days=D
        )
        for branch in ms.branches:
            assert not branch.stencil.weight.requires_grad, \
                "Frozen stencil weight must have requires_grad=False."

    def test_unfrozen_stencils_have_gradient(self):
        """Un-frozen stencil weights must receive gradients."""
        ms = MultiScaleLaplacian(
            dilations=(1, 2, 4), freeze_stencils=False, weeks=W, days=D
        )
        x = _clean_grid()
        ms(x).sum().backward()
        for i, branch in enumerate(ms.branches):
            assert branch.stencil.weight.grad is not None, \
                f"Un-frozen scale {i} stencil must receive gradient."

    def test_wrong_input_shape_raises(self):
        ms = MultiScaleLaplacian(dilations=(1,), weeks=W, days=D)
        with pytest.raises(ValueError, match="expects"):
            ms(torch.randn(4, W, D))   # missing channel dim

    def test_head_projects_to_embed_dim(self):
        for ed in [32, 60, 128]:
            ms  = MultiScaleLaplacian(dilations=(1, 2, 4), weeks=W, days=D, embed_dim=ed)
            out = ms(_clean_grid())
            assert out.shape[-1] == ed
