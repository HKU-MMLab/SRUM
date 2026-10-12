# Method

SRUM is an **offline self-rewarding** post-training recipe for unified multimodal models (UMMs). The understanding module scores generated images; the generation / flow module is updated. The evaluator stays frozen (no evaluator gradients).

## Four steps

1. **Candidates.** Sample ~6K images from the base UMM for each training prompt (paper setting).
2. **Frozen evaluator.** Keep the understanding side fixed. It only produces rewards.
3. **Dual reward.**
   - Global score **α** ∈ [0, 1]: whole-image layout / semantic alignment.
   - Local map **R**: region scores ∈ [−1, 1] painted into boxes for object-level fidelity.
4. **Optimize generation.** Reward-weighted flow matching plus a reference constraint:

```text
L = L_r + λ_c · L_ref ,   λ_c = 0.5
```

`L_r` weights the flow target by `α · R`. `L_ref` is the unweighted reference term (not an extra KL divergence; see paper / author response).

## Defaults (paper)

| Item | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | 2.5×10<sup>−5</sup> |
| Warmup | 500 steps |
| Grad clip | 1.0 |
| λ<sub>c</sub> | 0.5 |

## Code map

| File | Role |
| --- | --- |
| `srum/prompts.py` | Global / local scoring prompt templates |
| `srum/rewards.py` | Build `R`, gate with `α`, combine `L_r` + `L_ref` |
| `srum/train_srum.py` | Offline skeleton + dry-run on JSONL manifests |
| `Bagel/train/` | FSDP training stack to hook for full runs |

## Dry run

```bash
python -m srum.train_srum --dry-run \
  --candidate-manifest data/candidates.example.jsonl \
  --output-dir outputs/srum_dry
```

Expected manifest fields per line: `prompt`, `image_path`, `height`, `width`, `global_score`, `regions` (`id`, `bbox`, `score`).
