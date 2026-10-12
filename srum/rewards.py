"""Global–local dual reward utilities for SRUM.

Paper loss (offline self-rewarding):
    L = L_r + lambda_c * L_ref
    L_r   = E[ alpha * R ⊙ ||v_θ - (ε - x_gt)||_2^2 ]
    L_ref = E[ ||v_θ - (ε - x_gt)||_2^2 ]
with lambda_c = 0.5 by default.

`alpha` is the global layout score in [0, 1].
`R` is a spatial local reward map filled from region scores in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch


@dataclass
class DualRewardConfig:
    lambda_c: float = 0.5
    local_score_min: float = -1.0
    local_score_max: float = 1.0
    global_score_min: float = 0.0
    global_score_max: float = 1.0


@dataclass
class RegionScore:
    region_id: str
    bbox_xyxy: Sequence[float]  # normalized [x0,y0,x1,y1] in [0,1]
    score: float


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def aggregate_dual_reward(
    global_score: float,
    regions: Iterable[RegionScore],
    height: int,
    width: int,
    device: Optional[torch.device] = None,
    cfg: Optional[DualRewardConfig] = None,
) -> tuple[torch.Tensor, float]:
    """Build spatial reward map R and scalar alpha.

    Returns
    -------
    reward_map : Tensor[H, W]
        Local scores painted into boxes (default 0 elsewhere).
    alpha : float
        Global gating weight in [0, 1].
    """
    cfg = cfg or DualRewardConfig()
    device = device or torch.device("cpu")
    alpha = clamp(float(global_score), cfg.global_score_min, cfg.global_score_max)
    reward_map = torch.zeros(height, width, device=device, dtype=torch.float32)

    for region in regions:
        x0, y0, x1, y1 = region.bbox_xyxy
        ix0 = int(clamp(x0, 0.0, 1.0) * (width - 1))
        iy0 = int(clamp(y0, 0.0, 1.0) * (height - 1))
        ix1 = int(clamp(x1, 0.0, 1.0) * (width - 1)) + 1
        iy1 = int(clamp(y1, 0.0, 1.0) * (height - 1)) + 1
        score = clamp(float(region.score), cfg.local_score_min, cfg.local_score_max)
        reward_map[iy0:iy1, ix0:ix1] = score

    return reward_map, alpha


def reward_weighted_flow_loss(
    velocity_pred: torch.Tensor,
    velocity_target: torch.Tensor,
    reward_map: torch.Tensor,
    alpha: float,
    lambda_c: float = 0.5,
) -> torch.Tensor:
    """Combine reward-weighted loss and reference constraint.

    velocity_* : [B, C, H, W] (or broadcastable to reward_map spatial size)
    reward_map : [H, W] or [B, H, W]
    """
    if reward_map.dim() == 2:
        reward_map = reward_map.unsqueeze(0)
    while reward_map.dim() < velocity_pred.dim():
        reward_map = reward_map.unsqueeze(1)

    per_pixel = (velocity_pred - velocity_target).pow(2)
    # broadcast reward over channels
    weighted = (alpha * reward_map) * per_pixel
    l_r = weighted.mean()
    l_ref = per_pixel.mean()
    return l_r + lambda_c * l_ref


def summarize_region_scores(regions: List[RegionScore]) -> dict:
    if not regions:
        return {"count": 0, "mean": 0.0}
    scores = [r.score for r in regions]
    return {
        "count": len(scores),
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }
