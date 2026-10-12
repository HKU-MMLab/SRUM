<div align="center">

# SRUM

### Fine-Grained Self-Rewarding for Unified Multimodal Models

**ECCV 2026**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.12784)
[![Model](https://img.shields.io/badge/Model-Hugging%20Face-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/Wayne-King/SRUM_BAGEL_7B_MoT)
[![Project](https://img.shields.io/badge/Project-Page-0f6e56)](./docs/index.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](./LICENSE)

**Weiyang Jin**<sup>*</sup> · **Yuwei Niu**<sup>*</sup> · Jiaqi Liao · Chengqi Duan · Aoxue Li · Shenghua Gao · **Xihui Liu**

The University of Hong Kong  
Contact: [xihuiliu@hku.hk](mailto:xihuiliu@hku.hk)

</div>

---

## Highlights

SRUM turns a unified multimodal model’s **understanding** module into an internal rewarder for its **generation** module — no extra human labels.

- **Self-rewarding post-training** for UMMs (BAGEL and beyond)
- **Global–local dual rewards** for layout semantics and object-level fidelity
- Strong gains on composition & reasoning T2I benchmarks

| Benchmark | Base | +SRUM |
|-----------|------|-------|
| T2I-CompBench (overall) | 82.18 | **88.37** |
| T2I-ReasonBench (avg.) | 43.82 | **46.75** |

## News

- **2026-10** — Code release under [`HKU-MMLab/SRUM`](https://github.com/HKU-MMLab/SRUM).
- **2025-11** — Model & report on Hugging Face: [`Wayne-King/SRUM_BAGEL_7B_MoT`](https://huggingface.co/Wayne-King/SRUM_BAGEL_7B_MoT).

## Quick Start

```bash
git clone https://github.com/HKU-MMLab/SRUM.git
cd SRUM
pip install -r requirements.txt
```

Download checkpoints:

```bash
# base BAGEL
python scripts/download_ckpt.py \
  --repo ByteDance-Seed/BAGEL-7B-MoT \
  --out models/BAGEL-7B-MoT

# SRUM-tuned BAGEL
python scripts/download_ckpt.py \
  --repo Wayne-King/SRUM_BAGEL_7B_MoT \
  --out models/SRUM_BAGEL_7B_MoT
```

Inference uses the BAGEL interface — point the model path to `models/SRUM_BAGEL_7B_MoT`:

```bash
cd Bagel
# app.py / inference.ipynb / inferencer.py
```

<details>
<summary><b>Inference hyperparameters</b></summary>

| Param | Role | Typical |
|-------|------|---------|
| `cfg_text_scale` | Text guidance (`1.0` = off) | `4.0–8.0` |
| `cfg_image_scale` | Image preservation | `1.0–2.0` |
| `cfg_interval` | Denoising fraction with CFG | `[0.4, 1.0]` |
| `timestep_shift` | Early layout vs late detail | — |
| `num_timesteps` | Denoising steps | `50` |
| `cfg_renorm_type` | `global` / `channel` / `text_channel` | `global` (T2I) |

</details>

## Method

Offline self-rewarding (paper setting):

1. Sample candidates from the base UMM (~6K).
2. Freeze the understanding module as evaluator (**no evaluator gradients**).
3. Score with global layout weight **α** and local region map **R**.
4. Update generation / flow parameters with

```text
L = L_r + λ_c · L_ref ,   λ_c = 0.5
```

Defaults: **AdamW**, lr **2.5×10<sup>−5</sup>**, warmup **500**, grad clip **1.0**.

| Module | Path |
|--------|------|
| Scoring prompts | `srum/prompts.py` |
| Dual-reward helpers | `srum/rewards.py` |
| Training skeleton | `srum/train_srum.py` |
| BAGEL stack | `Bagel/` |

## Evaluation

We mainly report:

| Benchmark | Focus | Link |
|-----------|--------|------|
| **T2I-CompBench** | Compositional T2I (color, shape, spatial, numeracy, …) | [GitHub](https://github.com/Karine-Huang/T2I-CompBench) |
| **T2I-ReasonBench** | Reasoning-informed T2I (entity, idiom, science, text) | paper / release notes |

Related generation / editing benches used in the broader UMM ecosystem (optional extras when comparing to BAGEL / RecA-style setups):

| Benchmark | Focus | Link |
|-----------|--------|------|
| GenEval | Object-focused compositional generation | [GitHub](https://github.com/djghosh13/geneval) |
| DPG-Bench | Dense prompt following | [GitHub](https://github.com/TencentQQGYLab/ELLA) |
| WISE | World-knowledge T2I | [GitHub](https://github.com/PKU-YuanGroup/WISE) |
| ImgEdit / GEdit | Image editing quality | see BAGEL `Bagel/eval` |

Eval helpers live under `Bagel/eval/` (GenEval, WISE, ImgEdit, …). CompBench-style scoring follows the official T2I-CompBench protocol.

## Results

**Composition (T2I-CompBench overall)**

| Model | Overall |
|-------|---------|
| Bagel | 82.18 |
| Bagel (CoT) | 84.46 |
| **Bagel + SRUM (CoT)** | **88.37** |

**Reasoning-informed (T2I-ReasonBench)**

| Model | Average |
|-------|---------|
| Bagel | 43.82 |
| Bagel + SFT | 45.37 |
| **Bagel + SRUM** | **46.75** |

Full tables: [Hugging Face model card](https://huggingface.co/Wayne-King/SRUM_BAGEL_7B_MoT) · [arXiv](https://arxiv.org/abs/2510.12784)

## Repository

```text
SRUM/
├── Bagel/          # inference & training (BAGEL-compatible)
├── srum/           # self-rewarding modules
├── scripts/        # checkpoint download helpers
├── docs/           # project page
├── assets/
└── citation.bib
```

## Citation

```bibtex
@inproceedings{jin2026srum,
  title={Srum: Fine-grained self-rewarding for unified multimodal models},
  author={Jin, Weiyang and Niu, Yuwei and Liao, Jiaqi and Duan, Chengqi and Li, Aoxue and Gao, Shenghua and Liu, Xihui},
  booktitle={European Conference on Computer Vision},
  pages={675--691},
  year={2026},
  organization={Springer}
}
```

## Acknowledgments

Built on [BAGEL](https://github.com/ByteDance-Seed/Bagel). Project page inspired by [RecA](https://github.com/horizonwind2004/reconstruction-alignment).

## License

Apache 2.0. BAGEL components follow their upstream terms.
