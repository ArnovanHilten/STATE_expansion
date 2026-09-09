#!/bin/bash
#SBATCH --job-name=state_qc_ablation
#SBATCH --account=cu_0055
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=state_qc_ablation_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/users/hilarn/55_cu_0055/dockers/latest/dcai_test+docker_test+state-expansion.sqsh

# =========================
# Configurable
# =========================
RUN_ID="31"
CHECKPOINT="best.ckpt"
N_SOURCES=9

BASE="/dcai/users/hilarn/55_cu_0055/code/enhance_state/results/${RUN_ID}"
REPLOGLE_DIR="${BASE}/qc_emb_lr1e-4/qc_emb_${RUN_ID}_lr1e-4"
TIAN_DIR="${BASE}/qc_emb_Tian_lr1e-5/qc_emb_Tian_${RUN_ID}_lr1e-5"

# =========================
# Environment setup
# =========================

unset LMOD_CMD

export WANDB_BASE_URL="https://wandb.gefion.dcai.dk"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "NODELIST=${SLURM_NODELIST}"
cd /dcai/users/hilarn/55_cu_0055/code/enhance_state

run_ablations() {
    local RUN_DIR="$1"
    local LABEL="$2"

    echo "=== ${LABEL}: full model predict ==="
    CUDA_VISIBLE_DEVICES=0 state tx predict \
        --output-dir "${RUN_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --save-attn-weights \
        --profile full
    echo "${LABEL}: full model done."

    echo "=== ${LABEL}: source ablations ==="
    gpu=0
    pids=()
    for s in $(seq 0 $((N_SOURCES - 1))); do
        echo "  Ablating source ${s} on GPU ${gpu}"
        CUDA_VISIBLE_DEVICES=${gpu} state tx predict \
            --output-dir "${RUN_DIR}" \
            --checkpoint "${CHECKPOINT}" \
            --ablate-source "${s}" \
            --profile full &

        pids+=($!)
        gpu=$(( (gpu + 1) % 4 ))

        if (( (s + 1) % 4 == 0 )); then
            wait "${pids[@]}"
            pids=()
        fi
    done
    wait
    echo "${LABEL}: all ablations done."
}

run_ablations "${REPLOGLE_DIR}" "Replogle"
run_ablations "${TIAN_DIR}"     "Tian"

echo "All ablation predicts complete."
