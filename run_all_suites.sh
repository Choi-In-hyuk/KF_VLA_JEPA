#!/bin/bash
# 4개 suite × (baseline + KF + EMA 0.5 + EMA 0.7) × 3 seeds
# - 이미 완료된 결과는 자동 skip
#
# Usage: bash run_all_suites.sh

set -eo pipefail

# ── 설정 ────────────────────────────────────────────────────────────
EMA_ALPHAS=(0.5 0.7)
NUM_TRIALS=50
SEEDS=(7 42 123)
PORT=15087

PYTHON=/home/choi/miniconda3/envs/vla_jepa/bin/python
CKPT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:$(pwd):$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1

declare -A LIBERO_DIRS
LIBERO_DIRS["libero_spatial"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/libero_spatial/libero_spatial"
LIBERO_DIRS["libero_object"]="/home/choi/LGHA/LIBERO/libero/datasets/libero_object"
LIBERO_DIRS["libero_goal"]="/home/choi/LGHA/LIBERO/libero/datasets/libero_goal"
LIBERO_DIRS["libero_10"]="/home/choi/LGHA/LIBERO/libero/datasets/libero_10"

declare -A TOKEN_DIRS
TOKEN_DIRS["libero_spatial"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/libero_spatial/tokens_vla_jepa"
TOKEN_DIRS["libero_object"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/libero_object/tokens_vla_jepa"
TOKEN_DIRS["libero_goal"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/libero_goal/tokens_vla_jepa"
TOKEN_DIRS["libero_10"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/datasets/libero_10/tokens_vla_jepa"

declare -A LDS_PATHS
LDS_PATHS["libero_spatial"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_spatial_lds_tokens/lds_64.npz"
LDS_PATHS["libero_object"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_object_lds_tokens/lds_64.npz"
LDS_PATHS["libero_goal"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_goal_lds_tokens/lds_64.npz"
LDS_PATHS["libero_10"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_10_lds_tokens/lds_64.npz"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# ── 헬퍼: 서버 실행 ─────────────────────────────────────────────────
# mode: baseline | kf | ema
# alpha: EMA alpha (mode=ema일 때만 사용)
start_server() {
    local mode=$1
    local suite=$2
    local server_log=$3
    local alpha=${4:-""}

    fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 2
    rm -f "${server_log}"

    case ${mode} in
        kf)
            ${PYTHON} ./deployment/model_server/server_policy.py \
                --ckpt_path ${CKPT} --port ${PORT} --use_bf16 --cuda 0 \
                --lds_path ${LDS_PATHS[$suite]} \
                --kf_q 0.1 --kf_r 5.0 > "${server_log}" 2>&1 &
            ;;
        ema)
            ${PYTHON} ./deployment/model_server/server_policy.py \
                --ckpt_path ${CKPT} --port ${PORT} --use_bf16 --cuda 0 \
                --ema_alpha ${alpha} > "${server_log}" 2>&1 &
            ;;
        *)
            ${PYTHON} ./deployment/model_server/server_policy.py \
                --ckpt_path ${CKPT} --port ${PORT} --use_bf16 --cuda 0 > "${server_log}" 2>&1 &
            ;;
    esac

    local elapsed=0
    until grep -q "server listening" "${server_log}" 2>/dev/null; do
        sleep 2; elapsed=$((elapsed + 2))
        if [ $elapsed -ge 120 ]; then
            echo "ERROR: Server failed to start."; cat "${server_log}"
            kill $(pgrep -f "server_policy.*${PORT}") 2>/dev/null || true
            exit 1
        fi
    done
}

# ── 헬퍼: eval 1회 실행 ─────────────────────────────────────────────
run_eval() {
    local suite=$1
    local mode=$2
    local run_id=$3
    local alpha=${4:-""}
    local seed=${SEEDS[$((run_id - 1))]}

    # 폴더명 결정
    local folder
    if [ "${mode}" = "ema" ]; then
        local alpha_str=$(echo ${alpha} | tr '.' 'p')
        folder="EMA_${alpha_str}_run${run_id}"
    else
        folder="${mode}_run${run_id}"
    fi

    local out_dir="results/${suite}/${folder}"
    local result_log="${out_dir}/eval.log"

    # 이미 완료된 경우 skip
    if [ -f "${result_log}" ] && grep -q "Total success rate" "${result_log}" 2>/dev/null; then
        local sr=$(grep "Total success rate" "${result_log}" | tail -1 | grep -oP '\d+\.\d+')
        echo "  [SKIP] ${suite}/${folder} (${sr})"
        return 0
    fi

    mkdir -p "${out_dir}"
    echo ""
    echo "  ── [${suite}] ${folder}  seed=${seed} ──"

    local server_log="/tmp/vla_server_${suite}_${mode}_${run_id}.log"
    start_server ${mode} ${suite} ${server_log} ${alpha}
    echo "  Server up."

    ${PYTHON} ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path ${CKPT} \
        --args.host "127.0.0.1" \
        --args.port ${PORT} \
        --args.task-suite-name "${suite}" \
        --args.num-trials-per-task ${NUM_TRIALS} \
        --args.video-out-path "${out_dir}" \
        --args.with_state "true" \
        --args.seed ${seed} \
        2>&1 | tee "${result_log}"

    kill $(pgrep -f "server_policy.*${PORT}") 2>/dev/null || true
    sleep 2
}

# ════════════════════════════════════════════════════════════════════
# 메인 루프
# ════════════════════════════════════════════════════════════════════
for suite in "${SUITES[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "SUITE: ${suite}"
    echo "════════════════════════════════════════════════════════"

    TOKEN_DIR=${TOKEN_DIRS[$suite]}
    LDS_PATH=${LDS_PATHS[$suite]}
    LDS_CKPT_DIR=$(dirname ${LDS_PATH})

    # Step 1: 토큰 추출
    if [ -d "${TOKEN_DIR}/train" ] && [ "$(ls ${TOKEN_DIR}/train/*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[SKIP] 토큰 이미 존재: ${TOKEN_DIR}"
    else
        echo "[Step 1] 토큰 추출: ${suite}"
        ${PYTHON} /home/choi/vjepa2/experiments/vla_jepa/extract_tokens.py \
            --ckpt_path ${CKPT} \
            --libero_dir ${LIBERO_DIRS[$suite]} \
            --out_dir ${TOKEN_DIR}
    fi

    # Step 2: LDS 학습
    if [ -f "${LDS_PATH}" ]; then
        echo "[SKIP] LDS 이미 존재: ${LDS_PATH}"
    else
        echo "[Step 2] LDS 학습: ${suite}"
        ${PYTHON} /home/choi/vjepa2/experiments/vla_jepa/train_lds_tokens.py \
            --token_dir ${TOKEN_DIR} \
            --ckpt_dir ${LDS_CKPT_DIR} \
            --latent_dim 64
    fi

    # Step 3: 평가
    if [ "${suite}" = "libero_10" ]; then
        echo "[Step 3] 평가: baseline/KF × 3 seeds (libero_10 EMA 스킵 — PRO 우선)"
        for i in 1 2 3; do run_eval ${suite} "baseline" $i; done
        for i in 1 2 3; do run_eval ${suite} "kf"       $i; done
    else
        echo "[Step 3] 평가: baseline/KF/EMA(0.5,0.7) × 3 seeds"
        for i in 1 2 3; do run_eval ${suite} "baseline" $i; done
        for i in 1 2 3; do run_eval ${suite} "kf"       $i; done
        for alpha in "${EMA_ALPHAS[@]}"; do
            for i in 1 2 3; do run_eval ${suite} "ema" $i ${alpha}; done
        done
    fi
done

# ════════════════════════════════════════════════════════════════════
# 최종 결과 집계
# ════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════"
echo "최종 결과 요약"
echo "════════════════════════════════════════════════════════"

${PYTHON} - <<'PYEOF'
import numpy as np
from scipy import stats
import re, os

def get_sr(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        for line in reversed(f.readlines()):
            m = re.search(r'Total success rate.*?(\d+\.\d+)', line)
            if m: return float(m.group(1))
    return None

suites  = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
methods = [
    ("baseline", "Baseline",    lambda s,i: f"baseline_run{i}"),
    ("kf",       "KF (lds_64)", lambda s,i: f"kf_run{i}"),
    ("ema_0p5",  "EMA (α=0.5)", lambda s,i: f"EMA_0p5_run{i}"),
    ("ema_0p7",  "EMA (α=0.7)", lambda s,i: f"EMA_0p7_run{i}"),
]

hdr = f"{'Suite':<18} {'Method':<16} {'Run1':>7} {'Run2':>7} {'Run3':>7} {'Mean':>8} {'Std':>6} {'vs Base':>9} {'p':>7}"
print(f"\n{hdr}")
print("─" * len(hdr))

for suite in suites:
    b_rates = None
    for key, label, folder_fn in methods:
        rates = [get_sr(f"results/{suite}/{folder_fn(suite, i)}/eval.log") for i in range(1,4)]
        rates = [r for r in rates if r is not None]

        if not rates:
            print(f"  {suite:<16} {label:<16} {'N/A':>7}")
            continue

        mean = np.mean(rates) * 100
        std  = np.std(rates)  * 100
        r_strs = [f"{r*100:.1f}%" for r in rates] + ["  N/A"] * (3 - len(rates))

        delta_str, p_str = "", ""
        if b_rates is not None and len(rates) == 3 and len(b_rates) == 3:
            _, p = stats.ttest_ind(rates, b_rates)
            delta_str = f"{mean - np.mean(b_rates)*100:+.2f}%p"
            p_str = f"{p:.3f}{'*' if p < 0.05 else ' '}"

        print(f"  {suite:<16} {label:<16} {r_strs[0]:>7} {r_strs[1]:>7} {r_strs[2]:>7} {mean:>7.2f}% {std:>5.2f}% {delta_str:>9} {p_str:>7}")

        if key == "baseline":
            b_rates = rates
    print()
PYEOF
