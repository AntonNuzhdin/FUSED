"""The FUSED training objective.

    L = lambda_seg * L_BCE + lambda_edge * L_EDG + lambda_cls * L_CE
        + lambda_moe * L_MoE + lambda_div * L_div

with lambda_seg = lambda_cls = 1, lambda_edge = 0.5 and
lambda_moe = lambda_div = 0.01.

The localization terms (L_BCE and L_EDG) are applied to manipulated images only;
authentic images are supervised by the classification term alone. L_EDG is an
edge-weighted BCE normalized by the number of boundary pixels. L_MoE is the
router load-balancing term and L_div the pooled-latent diversity penalty.
"""

import torch
import torch.nn.functional as F


def edge_bce(seg_logits, masks, edges):
    """Boundary-weighted BCE, normalized by the number of edge pixels."""
    logits = seg_logits.float()
    masks = masks.float()
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear",
                               align_corners=False)
    weight = edges.float()
    if weight.shape[-2:] != masks.shape[-2:]:
        weight = F.interpolate(weight, size=masks.shape[-2:], mode="nearest")
    return F.binary_cross_entropy_with_logits(
        logits, masks, weight=weight, reduction="sum"
    ) / (weight.sum() + 1e-6)


def fused_loss(
    result,
    batch,
    lambda_seg=1.0,
    lambda_edge=0.5,
    lambda_cls=1.0,
    lambda_moe=0.01,
    lambda_div=0.01,
):
    """Compute the total loss and a dict of detached component scalars.

    ``result`` is the model output; ``batch`` provides ``seg_labels`` (B, 1, H, W),
    ``edge`` (B, 1, H, W) and ``member`` (B,) where member > 0 marks a manipulated
    image.
    """
    seg_logits = result["logits"]
    masks = batch["seg_labels"]
    edges = batch["edge"]
    member = batch["member"]

    fake_idx = (member > 0).nonzero(as_tuple=True)[0]
    if fake_idx.numel() > 0:
        l_seg = F.binary_cross_entropy_with_logits(seg_logits[fake_idx], masks[fake_idx])
        l_edge = edge_bce(seg_logits[fake_idx], masks[fake_idx], edges[fake_idx])
    else:
        l_seg = seg_logits.new_tensor(0.0)
        l_edge = seg_logits.new_tensor(0.0)

    l_cls = F.cross_entropy(result["cls_logits"], (member > 0).long())
    l_moe = result.get("moe_aux_loss", seg_logits.new_tensor(0.0))
    l_div = result.get("diversity_loss", seg_logits.new_tensor(0.0))

    total = (lambda_seg * l_seg
             + lambda_edge * l_edge
             + lambda_cls * l_cls
             + lambda_moe * l_moe
             + lambda_div * l_div)

    components = {
        "l_seg": l_seg.detach(),
        "l_edge": l_edge.detach(),
        "l_cls": l_cls.detach(),
        "l_moe": l_moe.detach() if torch.is_tensor(l_moe) else torch.tensor(float(l_moe)),
        "l_div": l_div.detach() if torch.is_tensor(l_div) else torch.tensor(float(l_div)),
    }
    return total, components
