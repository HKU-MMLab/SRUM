<p align="center">
  <h1 align="center">SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models</h1>
  <p align="center"><b>ECCV 2026</b></p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.12784"><img src="https://img.shields.io/badge/Paper-arXiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://huggingface.co/Wayne-King/SRUM_BAGEL_7B_MoT"><img src="https://img.shields.io/badge/Model-HuggingFace-fcd022?style=for-the-badge&logo=huggingface&logoColor=000" alt="Model"></a>
  <a href="./docs/index.html"><img src="https://img.shields.io/badge/Project-Page-00CED1?style=for-the-badge" alt="Project"></a>
</p>

<p align="center">
  <b>Weiyang Jin*</b>, <b>Yuwei Niu*</b>, Jiaqi Liao, Chengqi Duan, Aoxue Li, Shenghua Gao, <b>Xihui Liu</b><br/>
  The University of Hong Kong &nbsp;·&nbsp; contact: <a href="mailto:xihuiliu@hku.hk">xihuiliu@hku.hk</a>
</p>

---

## Overview

**SRUM** is a self-rewarding post-training framework for Unified Multimodal Models (UMMs). It closes the gap where a model’s strong **understanding** fails to transfer to **generation**, by using the model’s own understanding module as an internal evaluator that rewards the generation module — **without extra human labels**.

A **global–local dual reward** provides multi-scale guidance:

- **Global reward** — overall visual semantics and layout
- **Local reward** — fine-grained, object-level fidelity

On BAGEL, SRUM boosts **T2I-CompBench** from **82.18 → 88.37** and **T2I-ReasonBench** from **43.82 → 46.75**.

This repository is the **ECCV 2026** reconstruction of SRUM: BAGEL-based inference/training code, self-rewarding modules, paper PDFs, and a project page.

## 🔥 News

- **2026-10**: Reorganized private release under `HKU-MMLab/SRUM`.
- **2025-11**: Model & report released on Hugging Face (`Wayne-King/SRUM_BAGEL_7B_MoT`).

## 🔧 Quick Start

### 1. Environment

```bash
git clone https://github.com/HKU-MMLab/SRUM.git
cd SRUM
pip install -r requirements.txt
# Optional: follow Bagel flash-attn install notes in Bagel/README.md
```

### 2. Download checkpoints

```bash
# Base BAGEL
python scripts/download_ckpt.py --repo ByteDance-Seed/BAGEL-7B-MoT --out models/BAGEL-7B-MoT

# SRUM-tuned BAGEL
python scripts/download_ckpt.py --repo Wayne-King/SRUM_BAGEL_7B_MoT --out models/SRUM_BAGEL_7B_MoT
```

### 3. Inference

Use the BAGEL inferencer with the SRUM checkpoint (same interface as BAGEL):

```bash
cd Bagel
# see inference.ipynb / app.py — point model path to ../models/SRUM_BAGEL_7B_MoT
```

### Inference hyperparameters (from BAGEL / SRUM HF card)

| Param | Role | Typical |
|-------|------|---------|
| `cfg_text_scale` | Text guidance | `4.0–8.0` (`1.0` = off) |
| `cfg_image_scale` | Image preservation | `1.0–2.0` |
| `cfg_interval` | Denoising fraction with CFG | `[0.4, 1.0]` |
| `timestep_shift` | Step schedule (layout vs detail) | — |
| `num_timesteps` | Denoising steps | `50` |
| `cfg_renorm_type` | `global` / `channel` / `text_channel` | `global` for T2I |

## 🧠 Method (self-rewarding)

Offline recipe (see paper + `paper/rebuttal.tex`):

1. Sample candidates from the base UMM (~6K in the paper setting).
2. Freeze the understanding side as evaluator (**no evaluator gradients**).
3. Score each candidate with **global** layout score \(\alpha\) and **local** region map \(R\).
4. Optimize generation / flow parameters with

\[
L = L_r + \lambda_c L_{\mathrm{ref}},\quad \lambda_c=0.5
\]

where \(L_r\) is the reward-weighted flow matching loss and \(L_{\mathrm{ref}}\) is the reference constraint (not an extra KL term).

Training defaults used in the paper response: AdamW, lr \(2.5\times10^{-5}\), warmup 500, grad clip 1.0.

Code entry points:

- `srum/prompts.py` — global / local scoring prompts
- `srum/rewards.py` — reward aggregation helpers
- `srum/train_srum.py` — offline self-reward training skeleton on top of Bagel

## 🍭 Results

### Composition (T2I-CompBench overall)

| Unified model | Overall |
|---------------|---------|
| Bagel | 82.18 |
| Bagel (CoT) | 84.46 |
| **Bagel+SRUM (CoT)** | **88.37** |

### Reasoning-informed (T2I-ReasonBench)

| Model | Average |
|-------|---------|
| Bagel | 43.82 |
| Bagel+SFT | 45.37 |
| **Bagel+SRUM** | **46.75** |

Full tables: Hugging Face model card / paper PDFs under `paper/`.

## 📁 Repository layout

```
SRUM-ECCV2026/
├── README.md
├── requirements.txt
├── Bagel/                 # BAGEL inference & training stack (upstream-compatible)
├── srum/                  # SRUM self-rewarding modules
├── scripts/               # download / demo helpers
├── docs/                  # project homepage
├── paper/                 # ECCV PDFs + rebuttal notes
└── assets/
```

## 📄 Paper

- arXiv: https://arxiv.org/abs/2510.12784
- Local PDFs (also on [GitHub Release v0.1.0](https://github.com/HKU-MMLab/SRUM/releases/tag/v0.1.0)): `paper/SRUM_ECCV2026_main.pdf`, `paper/SRUM_ECCV2026_supp.pdf`
- Rebuttal notes: [`paper/rebuttal.tex`](./paper/rebuttal.tex)

## ✍️ Citation

```bibtex
@inproceedings{jin2026srum,
  title     = {SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models},
  author    = {Jin, Weiyang and Niu, Yuwei and Liao, Jiaqi and Duan, Chengqi and Li, Aoxue and Gao, Shenghua and Liu, Xihui},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 🙏 Acknowledgments

Built on **[BAGEL](https://github.com/ByteDance-Seed/Bagel)**. Project page style inspired by **[RecA](https://github.com/horizonwind2004/reconstruction-alignment)**.

## 📜 License

Apache 2.0 (see `LICENSE`). BAGEL components follow their upstream license terms.
