"""
Robust Patch Aggregation for DINOv2ForensicStudent.
Prevents Noisy-OR inflation on authentic images with natural textures.
"""

from dataclasses import dataclass
import torch


@dataclass
class PerceptualAggregate:
    p_global: float
    p_local: float
    p_fused: float
    fused_logit: float
    loc_map: torch.Tensor


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def aggregate_perceptual(
    cls_logit: torch.Tensor,
    loc_map_logits: torch.Tensor,
    topk_ratio: float = 0.02,        
    local_activation_threshold: float = 0.65,
) -> PerceptualAggregate:
    flat_patch_logits = loc_map_logits.flatten()
    k = max(1, int(flat_patch_logits.numel() * topk_ratio))
    topk_vals, _ = torch.topk(flat_patch_logits, k)
    local_logit = topk_vals.mean()

    p_global = torch.sigmoid(cls_logit)
    p_local = torch.sigmoid(local_logit)

    if p_local > local_activation_threshold and p_global < 0.5:
        p_fused = 0.4 * p_global + 0.6 * p_local
    else:
        p_fused = 0.85 * p_global + 0.15 * p_local

    fused_logit = _logit(p_fused)
    loc_map_probs = torch.sigmoid(loc_map_logits).squeeze(0).squeeze(0)

    return PerceptualAggregate(
        p_global=float(p_global),
        p_local=float(p_local),
        p_fused=float(p_fused),
        fused_logit=float(fused_logit),
        loc_map=loc_map_probs.detach(),
    )