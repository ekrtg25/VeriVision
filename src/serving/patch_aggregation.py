"""
Aggregation of the DINOv2ForensicStudent's two heads (global CLS logit +
dense patch-anomaly map) into a single perceptual signal.

Why this exists
----------------
The global CLS classifier behaves like a *mean-pooled* representation over
the whole 518x518 image. When 75-80% of the frame is untouched real photo
and only a small region is spliced/generated, the CLS token's fake signal
gets diluted -> low P(AI), even though the patch-anomaly head correctly
lights up on the forged region.

This is a Multiple-Instance-Learning problem: the image-level label should
follow "fake if ANY patch is confidently fake", not "fake if the AVERAGE
patch looks fake". Top-k pooling over patch logits, combined with the CLS
logit via a noisy-OR in probability space, fixes this without retraining
the backbone.
"""

from dataclasses import dataclass

import torch


@dataclass
class PerceptualAggregate:
    p_global: float        # P(AI) from the CLS token alone
    p_local: float         # P(AI) from the top-k most anomalous patches
    p_fused: float         # combined perceptual probability
    fused_logit: float     # logit(p_fused) - fed into the cross-expert fusion
    loc_map: torch.Tensor  # (grid, grid) per-patch anomaly probabilities, for heatmaps


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def aggregate_perceptual(
    cls_logit: torch.Tensor,
    loc_map_logits: torch.Tensor,
    topk_ratio: float = 0.05,
    noisy_or_weight: float = 0.5,
) -> PerceptualAggregate:
    """
    cls_logit: 0-dim tensor, raw logit from `classifier` head.
    loc_map_logits: (1, 1, grid, grid) raw logits from `patch_anomaly` head.
    topk_ratio: fraction of patches used for the local score. Default 5%:
        for a 37x37=1369 patch grid that's ~68 patches - small enough to
        catch a localized splice, large enough to survive single-patch
        noise (a lone mis-fired patch shouldn't flip the verdict).
    noisy_or_weight: blend factor between the noisy-OR estimate and a
        straight local/global mix, kept tunable for calibration.
    """
    flat_patch_logits = loc_map_logits.flatten()
    k = max(1, int(flat_patch_logits.numel() * topk_ratio))
    topk_vals, _ = torch.topk(flat_patch_logits, k)
    local_logit = topk_vals.mean()

    p_global = torch.sigmoid(cls_logit)
    p_local = torch.sigmoid(local_logit)

    # "Fake if fake globally OR fake locally" - a confident local splice
    # must not be able to be out-voted by a large clean background.
    p_noisy_or = 1 - (1 - p_global) * (1 - p_local)
    p_weighted = noisy_or_weight * p_local + (1 - noisy_or_weight) * p_global

    p_fused = torch.maximum(p_noisy_or, p_weighted)
    fused_logit = _logit(p_fused)

    loc_map_probs = torch.sigmoid(loc_map_logits).squeeze(0).squeeze(0)

    return PerceptualAggregate(
        p_global=float(p_global),
        p_local=float(p_local),
        p_fused=float(p_fused),
        fused_logit=float(fused_logit),
        loc_map=loc_map_probs.detach(),
    )
