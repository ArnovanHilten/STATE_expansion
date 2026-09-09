#!/bin/bash
#SBATCH --job-name=state_predict
#SBATCH --nodes=1
#SBATCH --account=cu_0055
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=400GB
#SBATCH --output=state_predict_log_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/users/hilarn/55_cu_0055/dockers/latest/dcai_test+docker_test+state-expansion.sqsh

# =========================
# Configurable
# =========================
RUN_ID="31"
CHECKPOINT="best.ckpt"
BASE="/dcai/users/hilarn/55_cu_0055/code/enhance_state/results/${RUN_ID}"

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
# Replogle runs (batch 1 — GPUs 0-3)
# =========================
echo "=== Replogle predict ==="

CUDA_VISIBLE_DEVICES=0 state tx predict \
    --output-dir "${BASE}/qc_emb_lr1e-4/qc_emb_${RUN_ID}_lr1e-4" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

CUDA_VISIBLE_DEVICES=1 state tx predict \
    --output-dir "${BASE}/qc_emb_lr1e-5/qc_emb_${RUN_ID}_lr1e-5" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

CUDA_VISIBLE_DEVICES=2 state tx predict \
    --output-dir "${BASE}/baseline_lr1e-4/baseline_${RUN_ID}_lr1e-4" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

CUDA_VISIBLE_DEVICES=3 state tx predict \
    --output-dir "${BASE}/baseline_lr1e-5/baseline_${RUN_ID}_lr1e-5" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

wait
echo "Replogle predict done."

# =========================
# Tian runs (batch 2 — GPUs 0-3)
# =========================
echo "=== Tian predict ==="

CUDA_VISIBLE_DEVICES=0 state tx predict \
    --output-dir "${BASE}/qc_emb_Tian_lr1e-4/qc_emb_Tian_${RUN_ID}_lr1e-4" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

CUDA_VISIBLE_DEVICES=1 state tx predict \
    --output-dir "${BASE}/qc_emb_Tian_lr1e-5/qc_emb_Tian_${RUN_ID}_lr1e-5" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

CUDA_VISIBLE_DEVICES=2 state tx predict \
    --output-dir "${BASE}/baseline_Tian_lr1e-4/baseline_Tian_${RUN_ID}_lr1e-4" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

CUDA_VISIBLE_DEVICES=3 state tx predict \
    --output-dir "${BASE}/baseline_Tian_lr1e-5/baseline_Tian_${RUN_ID}_lr1e-5" \
    --checkpoint "${CHECKPOINT}" \
    --profile full &

wait
echo "Tian predict done."

echo "All predict jobs complete."
