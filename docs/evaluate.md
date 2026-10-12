# Evaluation

Numbers in the README are **paper cells**. This guide lists the benches we report and the entry points shipped with the BAGEL stack.

## Main (paper)

| Benchmark | What it measures | How to run |
| --- | --- | --- |
| [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) | Composition (color, shape, spatial, numeracy, texture, …) | Follow the official CompBench protocol on generated images. |
| T2I-ReasonBench | Reasoning-informed T2I (entity, idiom, scientific, textual) | Protocol as in the SRUM paper / HF card. |

Reported BAGEL + SRUM (CoT) overall on CompBench: **88.37** (base Bagel **82.18**).  
ReasonBench average: **46.75** (base **43.82**).

## Related suites in `Bagel/eval`

These scripts mirror the upstream BAGEL evaluation layout:

| Suite | Script / folder | Notes |
| --- | --- | --- |
| GenEval | `Bagel/scripts/eval/run_geneval.sh` | Object-centric composition |
| WISE | `Bagel/scripts/eval/run_wise.sh` | World-knowledge T2I |
| ImgEdit | `Bagel/scripts/eval/run_imgedit.sh` | Editing quality |
| GEdit | `Bagel/scripts/eval/run_gedit.sh` | Editing (GEdit) |
| KRIS | `Bagel/scripts/eval/run_kris.sh` | Knowledge / reasoning edits |
| RISE | `Bagel/scripts/eval/run_rise.sh` | Reasoning edits |
| VLM suite | `Bagel/scripts/eval/run_eval_vlm.sh` | Understanding side checks |

Also useful when comparing UMM post-training broadly: [DPG-Bench](https://github.com/TencentQQGYLab/ELLA) (dense prompts).

## Suggested workflow

1. Generate images with `models/SRUM_BAGEL_7B_MoT` (same CFG defaults as the README).
2. Score CompBench / ReasonBench with the official evaluators.
3. Optionally run BAGEL `scripts/eval/*` for GenEval / WISE / editing suites.
4. Do not mix cells across benches that share similar-looking digits — report each suite separately (same discipline as MIST’s protocol notes).

## Links

- Paper: https://arxiv.org/abs/2510.12784  
- Model card (full tables): https://huggingface.co/Wayne-King/SRUM_BAGEL_7B_MoT  
- Method notes: [method.md](method.md)
