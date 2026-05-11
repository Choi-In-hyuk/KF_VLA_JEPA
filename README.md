# KF-VLA-JEPA: Inference-Time Temporal Smoothing via Learned Kalman Filter

> Training-free improvement for Vision-Language-Action models by applying a Learned LDS Kalman Filter to `embodied_action_tokens` at inference time.

---

## Overview

VLA-JEPA's QwenVL encoder independently processes each frame, causing `embodied_action_tokens` to fluctuate across timesteps even when the scene barely changes. This high-frequency variation propagates to the DiT action head, creating inconsistency between action chunks.

We address this by applying a **Learned LDS Kalman Filter** between QwenVL and DiT — no model retraining required.

```
Images + Language
       ↓
    QwenVL (Qwen3-VL-2B)
       ↓
embodied_action_tokens  [B, 32, 2048]
       ↓
━━━━━━━━━━━━━━━━━━━━━━━
  Learned KF Smoothing        ← inserted here
  (PCA → KF predict/update → PCA decode → residual correction)
━━━━━━━━━━━━━━━━━━━━━━━
       ↓
    DiT (Flow-matching)
       ↓
  action chunk  [B, 7, action_dim]
```

## Key Results (LIBERO Benchmark)

| Suite | Baseline | KF (ours) | Δ | p-value |
|-------|----------|-----------|---|---------|
| LIBERO-Spatial | 95.20% ± 0.49% | **97.13% ± 0.34%** | **+1.93%p** | **0.010\*** |
| LIBERO-Object | 99.93% ± 0.09% | 99.87% ± 0.09% | −0.07%p | 0.519 (ceiling) |
| LIBERO-Goal | 97.07% ± 0.90% | **97.73% ± 0.19%** | +0.67%p | 0.363 |
| LIBERO-Long | 95.33% ± 0.41% | TBD | TBD | TBD |

- **Statistically significant** improvement on LIBERO-Spatial (p=0.010)
- **Variance reduced** across all suites (e.g., LIBERO-Goal std: 0.90% → 0.19%)
- EMA smoothing (α=0.5, 0.7) shows no improvement or slight degradation — validating the importance of learned dynamics

---

## Method

### Offline (one-time per suite)

1. Extract `embodied_action_tokens` from 40 demo trajectories per task
2. Learn PCA encoder **E** (2048d → 64d) via Truncated SVD
3. Learn AR(1) transition matrix **A** (64×64) via least squares

### Online (every inference call)

$$\hat{z}_t^- = Az_{t-1}, \quad P_t^- = AP_{t-1}A^\top + Q$$

$$K_t = P_t^-(P_t^- + R)^{-1}$$

$$z_t = \hat{z}_t^- + K_t(z_t^{\text{obs}} - \hat{z}_t^-), \quad P_t = (I-K_t)P_t^-$$

Filtered latent $z_t$ is decoded back to 2048d and applied as a residual correction to all 32 tokens.

---

## Setup

```bash
git clone https://github.com/Choi-In-hyuk/KF_VLA_JEPA
cd KF_VLA_JEPA

conda create -n vla_jepa python=3.10 -y
conda activate vla_jepa

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

**LIBERO environment** (separate conda env):
```bash
# Follow official LIBERO installation: https://github.com/Lifelong-Robot-Learning/LIBERO
```

---

## Usage

### Step 1: Extract tokens from demo data

```bash
python experiments/vla_jepa/extract_tokens.py \
    --ckpt_path <VLA-JEPA checkpoint> \
    --libero_dir <LIBERO dataset dir> \
    --out_dir <token output dir>
```

### Step 2: Train LDS

```bash
python experiments/vla_jepa/train_lds_tokens.py \
    --token_dir <token output dir> \
    --ckpt_dir <LDS output dir> \
    --latent_dim 64
```

### Step 3: Run evaluation (Baseline / KF / EMA)

```bash
# Full pipeline: token extraction → LDS training → evaluation
bash run_all_suites.sh

# LIBERO-PRO perturbation evaluation
bash run_libero_pro.sh
```

The scripts automatically skip completed steps and support resuming.

---

## Server Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--lds_path` | Path to trained LDS `.npz` file (enables KF) | None |
| `--kf_q` | KF process noise | 0.1 |
| `--kf_r` | KF observation noise | 5.0 |
| `--ema_alpha` | EMA smoothing coefficient (enables EMA, mutually exclusive with KF) | None |

---

## Repository Structure

```
KF_VLA_JEPA/
├── src/models/kf/learned_lds.py          # LearnedLDS: PCA + AR(1)
├── experiments/vla_jepa/
│   ├── extract_tokens.py                  # Token extraction from demos
│   └── train_lds_tokens.py               # LDS offline training
├── starVLA/model/framework/VLA_JEPA.py   # KF/EMA integration
├── deployment/model_server/server_policy.py
├── examples/LIBERO/
│   ├── eval_libero.py
│   └── model2libero_interface.py
├── run_all_suites.sh                      # Full LIBERO pipeline
├── run_libero_pro.sh                      # LIBERO-PRO pipeline
└── results/results.md                     # Experimental results & analysis
```

---

## Acknowledgement

This work builds on [VLA-JEPA](https://arxiv.org/abs/2602.10098) (Sun et al., 2026) and the [starVLA](https://github.com/starVLA/starVLA) framework.
