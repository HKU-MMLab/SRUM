#!/usr/bin/env python3
"""Download BAGEL / SRUM checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="HF repo id, e.g. Wayne-King/SRUM_BAGEL_7B_MoT",
    )
    parser.add_argument("--out", required=True, help="Local directory")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=args.patterns,
        cache_dir=str(out / "cache"),
    )
    print(f"Downloaded {args.repo} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
