"""SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models."""

from .rewards import DualRewardConfig, aggregate_dual_reward
from .prompts import GLOBAL_LAYOUT_PROMPT, LOCAL_REGION_PROMPT

__all__ = [
    "DualRewardConfig",
    "aggregate_dual_reward",
    "GLOBAL_LAYOUT_PROMPT",
    "LOCAL_REGION_PROMPT",
]
