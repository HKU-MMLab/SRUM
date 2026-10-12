#!/usr/bin/env python3
"""Offline SRUM self-rewarding training skeleton.

This script documents the paper recipe and provides a clear extension
point on top of the BAGEL training stack under ``Bagel/train``.

Paper setting (author response):
  - Offline: generate ~6K candidates, freeze understanding evaluator,
    cache rewards, then update stage-4 generation/flow parameters.
  - Loss: L = L_r + 0.5 * L_ref
  - Optim: AdamW, lr=2.5e-5, warmup=500, grad clip=1.0

Usage (high level):
  1. Prepare candidate parquet/json with prompt, image path, region boxes.
  2. Run reward scoring with a frozen UMM understanding checkpoint.
  3. Launch BAGEL FSDP training with reward-weighted loss hook.

This file intentionally keeps heavy model wiring optional so the private
repo stays lightweight and reproducible once checkpoints are available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from .prompts import GLOBAL_LAYOUT_PROMPT, LOCAL_REGION_PROMPT
from .rewards import DualRewardConfig, RegionScore, aggregate_dual_reward


@dataclass
class SRUMTrainConfig:
    candidate_manifest: str = "data/candidates.jsonl"
    reward_cache: str = "data/reward_cache.jsonl"
    output_dir: str = "outputs/srum"
    lambda_c: float = 0.5
    lr: float = 2.5e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    num_candidates: int = 6000
    seed: int = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SRUM offline self-rewarding trainer (skeleton)")
    p.add_argument("--candidate-manifest", type=str, default="data/candidates.jsonl")
    p.add_argument("--reward-cache", type=str, default="data/reward_cache.jsonl")
    p.add_argument("--output-dir", type=str, default="outputs/srum")
    p.add_argument("--lambda-c", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true", help="Only validate manifests / write config")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def materialize_example_reward(row: Dict[str, Any], cfg: DualRewardConfig) -> Dict[str, Any]:
    """Build a cached reward record from a manifest row.

    Expected row keys:
      prompt, image_path, height, width,
      global_score, regions: [{id, bbox, score}, ...]
    """
    regions = [
        RegionScore(
            region_id=str(r.get("id", i)),
            bbox_xyxy=r["bbox"],
            score=float(r["score"]),
        )
        for i, r in enumerate(row.get("regions", []))
    ]
    reward_map, alpha = aggregate_dual_reward(
        global_score=float(row.get("global_score", 0.0)),
        regions=regions,
        height=int(row.get("height", 64)),
        width=int(row.get("width", 64)),
        cfg=cfg,
    )
    return {
        "prompt": row.get("prompt"),
        "image_path": row.get("image_path"),
        "alpha": alpha,
        "reward_map_mean": float(reward_map.mean().item()),
        "num_regions": len(regions),
        "global_prompt": GLOBAL_LAYOUT_PROMPT.format(prompt=row.get("prompt", "")),
        "local_prompt_template": LOCAL_REGION_PROMPT,
    }


def main() -> int:
    args = parse_args()
    cfg = SRUMTrainConfig(
        candidate_manifest=args.candidate_manifest,
        reward_cache=args.reward_cache,
        output_dir=args.output_dir,
        lambda_c=args.lambda_c,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    rows = load_jsonl(Path(cfg.candidate_manifest))
    print(f"[SRUM] candidates loaded: {len(rows)} from {cfg.candidate_manifest}")
    print(f"[SRUM] config written to {out / 'train_config.json'}")
    print("[SRUM] evaluator must stay frozen; only generation/flow params are updated.")

    if args.dry_run:
        reward_cfg = DualRewardConfig(lambda_c=cfg.lambda_c)
        preview = [materialize_example_reward(r, reward_cfg) for r in rows[:3]]
        preview_path = out / "reward_preview.json"
        preview_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
        print(f"[SRUM] dry-run preview -> {preview_path}")
        return 0

    print(
        "[SRUM] Full training hooks into Bagel/train (FSDP). "
        "Wire reward_weighted_flow_loss into the flow-matching step, "
        "then launch with your cluster script."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
