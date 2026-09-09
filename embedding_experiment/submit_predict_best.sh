#!/bin/bash
#SBATCH --job-name=state_predict_best
#SBATCH --account=cu_0055
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --mem=400GB
#SBATCH --output=state_predict_best_log_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/users/hilarn/55_cu_0055/dockers/latest/dcai_test+docker_test+state-expansion.sqsh

RUN_ID="31"
BASE="/dcai/users/hilarn/55_cu_0055/code/enhance_state/results/${RUN_ID}"

unset LMOD_CMD
export WANDB_BASE_URL="https://wandb.gefion.dcai.dk"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "NODELIST=${SLURM_NODELIST}"
cd /dcai/users/hilarn/55_cu_0055/code/enhance_state

CUDA_VISIBLE_DEVICES=0 state tx predict \
    --output-dir "${BASE}/qc_emb_lr1e-4/qc_emb_${RUN_ID}_lr1e-4" \
    --checkpoint best.ckpt \
    --save-attn-weights \
    --profile full &

CUDA_VISIBLE_DEVICES=1 state tx predict \
    --output-dir "${BASE}/qc_emb_Tian_lr1e-5/qc_emb_Tian_${RUN_ID}_lr1e-5" \
    --checkpoint best.ckpt \
    --save-attn-weights \
    --profile full &

wait
echo "Done."
