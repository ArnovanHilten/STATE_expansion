#!/bin/bash
#SBATCH --job-name=state_qc_ablation
#SBATCH --account=cu_0055
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=state_qc_ablation_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/users/hilarn/55_cu_0055/dockers/latest/dcai_test+docker_test+state-expansion.sqsh

# =========================
# Configurable
# =========================
RUN_DIR="/dcai/users/hilarn/55_cu_0055/code/enhance_state/results/30/qc_emb_30_lr1e-4/qc_emb_30_lr1e-4"
CHECKPOINT="last.ckpt"
N_SOURCES=22   # number of QC sources; must match the npz

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

# =========================
# Full model (no ablation) — baseline for comparison
# =========================
CUDA_VISIBLE_DEVICES=0 state tx predict \
    --output-dir "${RUN_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

wait
echo "Full model predict done."

# =========================
# Ablate each source sequentially, 4 at a time across 4 GPUs
# =========================
gpu=0
pids=()

for s in $(seq 0 $((N_SOURCES - 1))); do
    echo "Ablating source ${s} on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=${gpu} state tx predict \
        --output-dir "${RUN_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --ablate-source "${s}" \
        --profile full &

    pids+=($!)
    gpu=$(( (gpu + 1) % 4 ))

    # Wait every 4 jobs to avoid overloading GPUs
    if (( (s + 1) % 4 == 0 )); then
        wait "${pids[@]}"
        pids=()
    fi
done

wait
echo "All ablation predicts complete."
echo "Results in ${RUN_DIR}/ablate_source_*/eval_${CHECKPOINT}/"
