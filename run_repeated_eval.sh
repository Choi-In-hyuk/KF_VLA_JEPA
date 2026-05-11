#!/bin/bash
# 3회 반복 실험: Baseline vs KF (VLA-JEPA + Learned LDS)
# Usage: bash run_repeated_eval.sh
# 예상 소요 시간: ~5시간 (각 50분 × 6회)

set -eo pipefail

_NVIDIA_LIBS=/home/choi/miniconda3/envs/vjepa2/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $_NVIDIA_LIBS -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
export LIBERO_HOME=/home/choi/LGHA/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1

SIM_PYTHON=/home/choi/miniconda3/envs/vla_jepa/bin/python
CKPT=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
LDS_PATH=/media/choi/8AA890DCA890C859/vjepa2_baseline/checkpoints/libero_spatial_lds_tokens/lds_64.npz
TASK_SUITE=libero_spatial
NUM_TRIALS=50
N_RUNS=3
PORT=15085
SEEDS=(7 42 123)

if [ ! -f "${LDS_PATH}" ]; then
    echo "ERROR: LDS checkpoint not found: ${LDS_PATH}"
    echo "먼저 run_kf_pipeline.sh의 Step 1, 2를 완료하세요."
    exit 1
fi

run_eval() {
    local mode=$1      # "baseline" or "kf"
    local run_id=$2    # 1, 2, 3
    local seed=${SEEDS[$((run_id - 1))]}
    local out_dir="results/${TASK_SUITE}/${mode}_run${run_id}"
    local server_log="/tmp/vla_server_${mode}_${run_id}.log"

    mkdir -p "${out_dir}"
    echo ""
    echo "================================================================"
    echo "[${mode} run ${run_id}/3] seed=${seed}  Output: ${out_dir}"
    echo "================================================================"

    fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 2

    rm -f "${server_log}"
    if [ "${mode}" = "kf" ]; then
        ${SIM_PYTHON} ./deployment/model_server/server_policy.py \
            --ckpt_path ${CKPT} \
            --port ${PORT} \
            --use_bf16 \
            --cuda 0 \
            --lds_path ${LDS_PATH} \
            --kf_q 0.1 \
            --kf_r 5.0 > "${server_log}" 2>&1 &
    else
        ${SIM_PYTHON} ./deployment/model_server/server_policy.py \
            --ckpt_path ${CKPT} \
            --port ${PORT} \
            --use_bf16 \
            --cuda 0 > "${server_log}" 2>&1 &
    fi
    SERVER_PID=$!
    echo "Policy server PID: ${SERVER_PID}"

    elapsed=0
    until grep -q "server listening" "${server_log}" 2>/dev/null; do
        sleep 2; elapsed=$((elapsed + 2))
        if [ $elapsed -ge 120 ]; then
            echo "ERROR: Server failed to start. Log:"
            cat "${server_log}"
            kill $SERVER_PID 2>/dev/null
            exit 1
        fi
    done
    echo "Server is up."

    ${SIM_PYTHON} ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path ${CKPT} \
        --args.host "127.0.0.1" \
        --args.port ${PORT} \
        --args.task-suite-name "${TASK_SUITE}" \
        --args.num-trials-per-task ${NUM_TRIALS} \
        --args.video-out-path "${out_dir}" \
        --args.with_state "true" \
        --args.seed ${seed} \
        2>&1 | tee "${out_dir}/eval.log"

    kill ${SERVER_PID} 2>/dev/null
    sleep 2
}

# ── 실험 실행 ──────────────────────────────────────────────────────────
echo "Baseline 3회 실행..."
for i in 1 2 3; do
    run_eval "baseline" $i
done

echo "KF 3회 실행..."
for i in 1 2 3; do
    run_eval "kf" $i
done

# ── 결과 집계 ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "결과 집계"
echo "================================================================"

extract_sr() {
    local log=$1
    grep "Total success rate" "${log}" | tail -1 | grep -oP '\d+\.\d+'
}

echo ""
echo "Baseline:"
BASELINE_RATES=()
for i in 1 2 3; do
    log="results/${TASK_SUITE}/baseline_run${i}/eval.log"
    sr=$(extract_sr "${log}")
    echo "  Run ${i}: ${sr}"
    BASELINE_RATES+=($sr)
done

echo ""
echo "KF (lds_64):"
KF_RATES=()
for i in 1 2 3; do
    log="results/${TASK_SUITE}/kf_run${i}/eval.log"
    sr=$(extract_sr "${log}")
    echo "  Run ${i}: ${sr}"
    KF_RATES+=($sr)
done

${SIM_PYTHON} - <<PYEOF
import numpy as np

baseline = [${BASELINE_RATES[0]}, ${BASELINE_RATES[1]}, ${BASELINE_RATES[2]}]
kf       = [${KF_RATES[0]}, ${KF_RATES[1]}, ${KF_RATES[2]}]

b_mean, b_std = np.mean(baseline), np.std(baseline)
k_mean, k_std = np.mean(kf),       np.std(kf)
delta = k_mean - b_mean

from scipy import stats
t_stat, p_val = stats.ttest_ind(kf, baseline)

print()
print("=" * 48)
print(f"  Baseline : {b_mean*100:.2f}% ± {b_std*100:.2f}%")
print(f"  KF       : {k_mean*100:.2f}% ± {k_std*100:.2f}%")
print(f"  Delta    : {delta*100:+.2f}%p")
print(f"  t-test   : t={t_stat:.3f}, p={p_val:.3f}")
if p_val < 0.05:
    print("  → p < 0.05: KF 효과 통계적으로 유의함")
else:
    print("  → p >= 0.05: 통계적으로 유의하지 않음")
print("=" * 48)
PYEOF

echo ""
echo "개별 로그: results/${TASK_SUITE}/baseline_run*/eval.log"
echo "          results/${TASK_SUITE}/kf_run*/eval.log"
