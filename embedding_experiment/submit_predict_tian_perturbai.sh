#!/bin/bash
#SBATCH --job-name=feng_replogle_tian_predict
#SBATCH --nodes=1
#SBATCH --account=cu_0055
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --mem=400GB
#SBATCH --output=feng_replogle_tian_predict_log_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/users/hilarn/55_cu_0055/dockers/latest/dcai_test+docker_test+state-expansion.sqsh
#
# Test-set evaluation for all 4 runs from
# submit_embedding_training_tian_perturbai.sh's RUN_ID=41 sweep (see
# submit_predict.sh in this same directory for the proven pattern this is
# based on). state tx predict only loads a checkpoint and evaluates -- no
# training happens -- so this is independent of the (96h, 8-GPU) training
# job and can be resubmitted any time for fresh numbers.
#
# --shared-only restricts evaluation to perturbations present in both train
# and test -- important here since the with_perturbai vs. baseline runs
# have different train compositions, so what counts as "shared with test"
# differs between them; without this flag the two wouldn't be compared on
# the same held-out perturbation set. --skip-adatas skips writing full
# predicted AnnData outputs, just the metrics.

# =========================
# Configurable
# =========================
RUN_ID="41"
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
# Feng/Replogle/Tian2021 (+/- perturbai_wholebrain) x LR sweep -- GPUs 0-3
# =========================
echo "=== predict ==="

CUDA_VISIBLE_DEVICES=0 state tx predict \
    --output-dir "${BASE}/with_perturbai_lr1e-4/with_perturbai_${RUN_ID}_lr1e-4" \
    --checkpoint "${CHECKPOINT}" \
    --shared-only \
    --skip-adatas &

CUDA_VISIBLE_DEVICES=1 state tx predict \
    --output-dir "${BASE}/with_perturbai_lr1e-5/with_perturbai_${RUN_ID}_lr1e-5" \
    --checkpoint "${CHECKPOINT}" \
    --shared-only \
    --skip-adatas &

CUDA_VISIBLE_DEVICES=2 state tx predict \
    --output-dir "${BASE}/baseline_lr1e-4/baseline_${RUN_ID}_lr1e-4" \
    --checkpoint "${CHECKPOINT}" \
    --shared-only \
    --skip-adatas &

CUDA_VISIBLE_DEVICES=3 state tx predict \
    --output-dir "${BASE}/baseline_lr1e-5/baseline_${RUN_ID}_lr1e-5" \
    --checkpoint "${CHECKPOINT}" \
    --shared-only \
    --skip-adatas &

wait
echo "predict done."
