#!/bin/bash
#SBATCH --job-name=predict_qc_emb
#SBATCH --account=cu_0055
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=predict_qc_emb_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/users/hilarn/55_cu_0055/dockers/latest/dcai_test+docker_test+state-expansion.sqsh

# =========================
# Configurable run number
# =========================
RUN_ID="20"   # must match the training RUN_ID

# =========================
# Environment setup
# =========================

unset LMOD_CMD

export NCCL_SOCKET_IFNAME=ens6f0
export NCCL_IB_HCA=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_btl=^vader,tcp,openib,uct
export OMPI_MCA_pml=ucx

export WANDB_BASE_URL="https://wandb.gefion.dcai.dk"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "NODELIST=${SLURM_NODELIST}"
echo "GPUS_REQUESTED=${SLURM_JOB_GPUS}"
echo "RUN_ID=${RUN_ID}"

cd /dcai/users/hilarn/55_cu_0055/code/enhance_state

# =========================
# Run predict sequentially for all runs in this RUN_ID
# =========================

for run_dir in "results/${RUN_ID}"/*/*/; do
  run_dir=${run_dir%/}
  base=$(basename "$run_dir")

  echo "Running predict for ${base}"

  CUDA_VISIBLE_DEVICES=0 state tx predict \
    --output-dir "${run_dir}" \
    --checkpoint "best.ckpt"
done
