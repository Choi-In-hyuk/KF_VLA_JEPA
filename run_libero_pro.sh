#!/bin/bash
# LIBERO-PRO: 퍼터베이션 평가 (baseline / KF / EMA 0.5 / EMA 0.7) × 4 perturbations × 3 seeds
# - LIBERO-PRO bddl/init 파일 기반, 별도 HDF5 데이터셋 불필요
# - LDS는 원본 suite에서 학습된 것 재사용 (run_all_suites.sh 완료 후 실행)
#
# Usage: bash run_libero_pro.sh

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

# LIBERO-PRO 사용 (perturbed suite 이름 인식)
export LIBERO_HOME=/home/choi/LIBERO-PRO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:/home/choi/VLA-JEPA:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1

# ── perturbed suite → base suite 매핑 (LDS 재사용) ─────────────────
# base suite LDS 경로
declare -A LDS_PATHS
LDS_PATHS["libero_spatial"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_spatial_lds_tokens/lds_64.npz"
LDS_PATHS["libero_object"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_object_lds_tokens/lds_64.npz"
LDS_PATHS["libero_goal"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_goal_lds_tokens/lds_64.npz"
LDS_PATHS["libero_10"]="/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_10_lds_tokens/lds_64.npz"

# 평가할 base suite 목록
BASE_SUITES=("libero_spatial" "libero_goal" "libero_10")

# 퍼터베이션 타입 (HF에서 받은 파일 기준)
PERTURBATIONS=("lan" "swap" "object" "task")

# ── 헬퍼: 서버 실행 ─────────────────────────────────────────────────
start_server() {
    local mode=$1        # baseline | kf | ema
    local base_suite=$2  # LDS 조회용 원본 suite
    local server_log=$3
    local alpha=${4:-""}

    fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 2
    rm -f "${server_log}"

    case ${mode} in
        kf)
            ${PYTHON} /home/choi/VLA-JEPA/deployment/model_server/server_policy.py \
                --ckpt_path ${CKPT} --port ${PORT} --use_bf16 --cuda 0 \
                --lds_path ${LDS_PATHS[$base_suite]} \
                --kf_q 0.1 --kf_r 5.0 > "${server_log}" 2>&1 &
            ;;
        ema)
            ${PYTHON} /home/choi/VLA-JEPA/deployment/model_server/server_policy.py \
                --ckpt_path ${CKPT} --port ${PORT} --use_bf16 --cuda 0 \
                --ema_alpha ${alpha} > "${server_log}" 2>&1 &
            ;;
        *)
            ${PYTHON} /home/choi/VLA-JEPA/deployment/model_server/server_policy.py \
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
    local perturbed_suite=$1  # e.g. libero_spatial_swap
    local base_suite=$2       # e.g. libero_spatial
    local mode=$3             # baseline | kf | ema
    local run_id=$4
    local alpha=${5:-""}
    local seed=${SEEDS[$((run_id - 1))]}

    local folder
    if [ "${mode}" = "ema" ]; then
        local alpha_str=$(echo ${alpha} | tr '.' 'p')
        folder="EMA_${alpha_str}_run${run_id}"
    else
        folder="${mode}_run${run_id}"
    fi

    local out_dir="results_pro/${perturbed_suite}/${folder}"
    local result_log="${out_dir}/eval.log"

    if [ -f "${result_log}" ] && grep -q "Total success rate" "${result_log}" 2>/dev/null; then
        local sr=$(grep "Total success rate" "${result_log}" | tail -1 | grep -oP '\d+\.\d+')
        echo "  [SKIP] ${perturbed_suite}/${folder} (${sr})"
        return 0
    fi

    mkdir -p "${out_dir}"
    echo ""
    echo "  ── [${perturbed_suite}] ${folder}  seed=${seed} ──"

    # KF 사용 시 LDS 파일 존재 확인
    if [ "${mode}" = "kf" ] && [ ! -f "${LDS_PATHS[$base_suite]}" ]; then
        echo "  [SKIP] LDS not found: ${LDS_PATHS[$base_suite]}"
        return 0
    fi

    local server_log="/tmp/vla_server_pro_${perturbed_suite}_${mode}_${run_id}.log"
    start_server ${mode} ${base_suite} ${server_log} ${alpha}
    echo "  Server up."

    ${PYTHON} /home/choi/VLA-JEPA/examples/LIBERO/eval_libero.py \
        --args.pretrained-path ${CKPT} \
        --args.host "127.0.0.1" \
        --args.port ${PORT} \
        --args.task-suite-name "${perturbed_suite}" \
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
cd /home/choi/VLA-JEPA

for base_suite in "${BASE_SUITES[@]}"; do
    for perturbation in "${PERTURBATIONS[@]}"; do
        perturbed_suite="${base_suite}_${perturbation}"

        echo ""
        echo "════════════════════════════════════════════════════════"
        echo "SUITE: ${perturbed_suite}"
        echo "════════════════════════════════════════════════════════"

        for i in 1 2 3; do run_eval ${perturbed_suite} ${base_suite} "baseline" $i; done
        for i in 1 2 3; do run_eval ${perturbed_suite} ${base_suite} "kf"       $i; done
        for alpha in "${EMA_ALPHAS[@]}"; do
            for i in 1 2 3; do run_eval ${perturbed_suite} ${base_suite} "ema" $i ${alpha}; done
        done
    done
done

# ════════════════════════════════════════════════════════════════════
# 최종 결과 집계
# ════════════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════════════════════"
echo "LIBERO-PRO 결과 요약"
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

base_suites    = ["libero_spatial", "libero_goal", "libero_10"]
perturbations  = ["lan", "swap", "object", "task"]
methods = [
    ("baseline", "Baseline",    lambda s,i: f"baseline_run{i}"),
    ("kf",       "KF (lds_64)", lambda s,i: f"kf_run{i}"),
    ("ema_0p5",  "EMA (α=0.5)", lambda s,i: f"EMA_0p5_run{i}"),
    ("ema_0p7",  "EMA (α=0.7)", lambda s,i: f"EMA_0p7_run{i}"),
]

hdr = f"{'Suite':<26} {'Method':<16} {'Run1':>7} {'Run2':>7} {'Run3':>7} {'Mean':>8} {'Std':>6} {'vs Base':>9} {'p':>7}"
print(f"\n{hdr}")
print("─" * len(hdr))

for base in base_suites:
    for pert in perturbations:
        suite = f"{base}_{pert}"
        b_rates = None
        for key, label, folder_fn in methods:
            rates = [get_sr(f"results_pro/{suite}/{folder_fn(suite, i)}/eval.log") for i in range(1,4)]
            rates = [r for r in rates if r is not None]
            if not rates:
                continue
            mean = np.mean(rates) * 100
            std  = np.std(rates)  * 100
            r_strs = [f"{r*100:.1f}%" for r in rates] + ["  N/A"] * (3 - len(rates))
            delta_str, p_str = "", ""
            if b_rates is not None and len(rates) == 3 and len(b_rates) == 3:
                _, p = stats.ttest_ind(rates, b_rates)
                delta_str = f"{mean - np.mean(b_rates)*100:+.2f}%p"
                p_str = f"{p:.3f}{'*' if p < 0.05 else ' '}"
            print(f"  {suite:<24} {label:<16} {r_strs[0]:>7} {r_strs[1]:>7} {r_strs[2]:>7} {mean:>7.2f}% {std:>5.2f}% {delta_str:>9} {p_str:>7}")
            if key == "baseline":
                b_rates = rates
        print()
PYEOF
