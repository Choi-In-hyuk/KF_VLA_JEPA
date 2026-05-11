#!/bin/bash
# EMA alpha 비교 실험: 0.3, 0.5, 0.7 순차 실행
# Usage: bash run_ema_eval.sh

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
TASK_SUITE=libero_spatial
NUM_TRIALS=50
PORT=15086
ALPHAS=(0.3 0.5 0.7)
SEEDS=(7 42 123)

run_ema() {
    local alpha=$1
    local run_id=$2
    local seed=${SEEDS[$((run_id - 1))]}
    local alpha_str=$(echo ${alpha} | tr '.' 'p')
    local out_dir="results/${TASK_SUITE}/EMA_${alpha_str}_run${run_id}"
    local server_log="/tmp/vla_server_ema_${alpha_str}_${run_id}.log"

    mkdir -p "${out_dir}"
    echo ""
    echo "================================================================"
    echo "[EMA alpha=${alpha} run ${run_id}/3] seed=${seed}  Output: ${out_dir}"
    echo "================================================================"

    fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 2

    rm -f "${server_log}"
    ${SIM_PYTHON} ./deployment/model_server/server_policy.py \
        --ckpt_path ${CKPT} \
        --port ${PORT} \
        --use_bf16 \
        --cuda 0 \
        --ema_alpha ${alpha} > "${server_log}" 2>&1 &
    SERVER_PID=$!
    echo "Policy server PID: ${SERVER_PID}"

    elapsed=0
    until grep -q "server listening" "${server_log}" 2>/dev/null; do
        sleep 2; elapsed=$((elapsed + 2))
        if [ $elapsed -ge 120 ]; then
            echo "ERROR: Server failed to start. Log:"; cat "${server_log}"
            kill $SERVER_PID 2>/dev/null; exit 1
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

    kill ${SERVER_PID} 2>/dev/null; sleep 2
}

# 각 alpha마다 3회 반복
for alpha in "${ALPHAS[@]}"; do
    for i in 1 2 3; do
        run_ema ${alpha} ${i}
    done
done

# 결과 집계
echo ""
echo "================================================================"
echo "결과 집계"
echo "================================================================"
echo ""
printf "%-12s %8s %8s %8s %10s\n" "Alpha" "Run1" "Run2" "Run3" "Mean±Std"

for alpha in "${ALPHAS[@]}"; do
    alpha_str=$(echo ${alpha} | tr '.' 'p')
    rates=()
    for i in 1 2 3; do
        log="results/${TASK_SUITE}/EMA_${alpha_str}_run${i}/eval.log"
        sr=$(grep "Total success rate" "${log}" | tail -1 | grep -oP '\d+\.\d+')
        rates+=($sr)
    done
    ${SIM_PYTHON} - <<PYEOF
import numpy as np
rates = [${rates[0]}, ${rates[1]}, ${rates[2]}]
m, s = np.mean(rates)*100, np.std(rates)*100
print(f"  alpha={${alpha}:.1f}    {rates[0]*100:.1f}%   {rates[1]*100:.1f}%   {rates[2]*100:.1f}%   {m:.2f}%±{s:.2f}%")
PYEOF
done

echo ""
echo "참고) Baseline: 95.20%±0.49%  |  KF: 97.13%±0.34%"
