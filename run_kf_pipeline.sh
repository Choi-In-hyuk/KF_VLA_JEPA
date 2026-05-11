#!/bin/bash
# Full KF pipeline: extract tokens → train LDS → eval with KF
# Run from: /home/choi/VLA-JEPA
#   bash run_kf_pipeline.sh

set -e

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}:$(pwd):/home/choi/vjepa2

PYTHON=/home/choi/miniconda3/envs/vla_jepa/bin/python
CKPT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
TOKEN_DIR=/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/libero_spatial/tokens_vla_jepa
LDS_CKPT_DIR=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_spatial_lds_tokens
LDS_PATH=${LDS_CKPT_DIR}/lds_64.npz

# ── Step 1: Extract tokens ──────────────────────────────────────────
echo "================================================================"
echo "Step 1: Extracting embodied_action_tokens from LIBERO-Spatial..."
echo "================================================================"
${PYTHON} /home/choi/vjepa2/experiments/vla_jepa/extract_tokens.py \
    --ckpt_path ${CKPT} \
    --out_dir ${TOKEN_DIR}

# ── Step 2: Train LDS ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Step 2: Training LDS on token sequences..."
echo "================================================================"
${PYTHON} /home/choi/vjepa2/experiments/vla_jepa/train_lds_tokens.py \
    --token_dir ${TOKEN_DIR} \
    --ckpt_dir ${LDS_CKPT_DIR} \
    --latent_dim 64

# ── Step 3: Eval with KF ────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Step 3: Evaluating VLA-JEPA + KF on libero_spatial..."
echo "================================================================"
bash examples/LIBERO/eval_libero_kf.sh libero_spatial ${LDS_PATH}

echo ""
echo "================================================================"
echo "Pipeline complete!"
echo "Baseline : results/libero_spatial/LIBERO_checkpoints_VLA-JEPA-LIBERO.pt/eval.log"
echo "KF result: results/libero_spatial/KF_lds_64/eval.log"
echo "================================================================"
