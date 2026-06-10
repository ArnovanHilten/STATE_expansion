#!/bin/bash
#SBATCH --job-name=retrain_SE_state
#SBATCH --nodes=1
#SBATCH --account=cu_0055
#SBATCH --gres=gpu:5                 # node has 8 GPUs
#SBATCH --time=96:00:00
#SBATCH --mem=1800GB
#SBATCH --output=retrain_state_R_all_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/projects01/cu_0055/notebooks/training_replogle_nadig/training_replogle_nadig.sqsh

# =========================
# Configurable run number
# =========================
RUN_ID="15"   # <--- change this per batch of runs


# =========================
# Environment setup
# =========================

# Avoid Lmod noise inside container
unset LMOD_CMD

export NCCL_SOCKET_IFNAME=ens6f0
export NCCL_IB_HCA=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_btl=^vader,tcp,openib,uct
export OMPI_MCA_pml=ucx

# W&B config: private server + cert bundle from host
export WANDB_BASE_URL="https://wandb.gefion.dcai.dk"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Stable GPU indexing
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "NODELIST=${SLURM_NODELIST}"
echo "GPUS_REQUESTED=${SLURM_JOB_GPUS}"
echo "RUN_ID=${RUN_ID}"

cd /dcai/projects/cu_0055/code/finetune_state

# =========================
# Launch runs on separate GPUs
# =========================

HIDDEN_DIM=128
CELL_SET_LEN=32
BATCHSIZE=64
LEARNINGRATE=1e-5


CUDA_VISIBLE_DEVICES=0 state tx train \
  data.kwargs.toml_config_path="toml/SE_R_Rk562.toml" \
  data.kwargs.num_workers=30 \
  data.kwargs.embed_key='X_state' \
  data.kwargs.output_space="gene" \
  data.kwargs.batch_col="gem_group" \
  data.kwargs.pert_col="gene" \
  data.kwargs.cell_type_key="cell_line" \
  data.kwargs.control_pert="non-targeting" \
  training.max_steps=80000 \
  training.ckpt_every_n_steps=2000 \
  training.batch_size=${BATCHSIZE} \
  training.lr=1e-4 \
  model.kwargs.cell_set_len=${CELL_SET_LEN} \
  model.kwargs.hidden_dim=${HIDDEN_DIM} \
  model.kwargs.batch_encoder=True \
  model=state \
  wandb.tags='["'"${RUN_ID}"'", "SE_R_Rk563", "SE"]' \
  wandb.entity="cu_0055" \
  wandb.project="finetune_state" \
  output_dir="results/${RUN_ID}/SE_R_Rk562_1" \
  name="SE_${RUN_ID}_R_Rk562_1" &

CUDA_VISIBLE_DEVICES=1 state tx train \
  data.kwargs.toml_config_path="toml/SE_R_Rk562.toml" \
  data.kwargs.num_workers=30 \
  data.kwargs.embed_key='X_state' \
  data.kwargs.output_space="gene" \
  data.kwargs.batch_col="gem_group" \
  data.kwargs.pert_col="gene" \
  data.kwargs.cell_type_key="cell_line" \
  data.kwargs.control_pert="non-targeting" \
  training.max_steps=80000 \
  training.ckpt_every_n_steps=2000 \
  training.batch_size=${BATCHSIZE} \
  training.lr=1e-4 \
  model.kwargs.cell_set_len=${CELL_SET_LEN} \
  model.kwargs.hidden_dim=${HIDDEN_DIM} \
  model.kwargs.batch_encoder=True \
  model=state \
  wandb.tags='["'"${RUN_ID}"'", "SE_R_Rk563", "SE"]' \
  wandb.entity="cu_0055" \
  wandb.project="finetune_state" \
  output_dir="results/${RUN_ID}/SE_R_Rk562_2" \
  name="SE_${RUN_ID}_R_Rk562_2" &

CUDA_VISIBLE_DEVICES=2 state tx train \
  data.kwargs.toml_config_path="toml/SE_R_Rk562.toml" \
  data.kwargs.num_workers=30 \
  data.kwargs.embed_key='X_state' \
  data.kwargs.output_space="gene" \
  data.kwargs.batch_col="gem_group" \
  data.kwargs.pert_col="gene" \
  data.kwargs.cell_type_key="cell_line" \
  data.kwargs.control_pert="non-targeting" \
  training.max_steps=80000 \
  training.ckpt_every_n_steps=2000 \
  training.batch_size=${BATCHSIZE} \
  training.lr=1e-5 \
  model.kwargs.cell_set_len=${CELL_SET_LEN} \
  model.kwargs.hidden_dim=${HIDDEN_DIM} \
  model.kwargs.batch_encoder=True \
  model=state \
  wandb.tags='["'"${RUN_ID}"'", "SE_R_Rk563", "SE"]' \
  wandb.entity="cu_0055" \
  wandb.project="finetune_state" \
  output_dir="results/${RUN_ID}/SE_R_Rk562_3" \
  name="SE_${RUN_ID}_R_Rk562_3" &


CUDA_VISIBLE_DEVICES=3 state tx train \
  data.kwargs.toml_config_path="toml/SE_R_Rk562.toml" \
  data.kwargs.num_workers=30 \
  data.kwargs.embed_key='X_state' \
  data.kwargs.output_space="gene" \
  data.kwargs.batch_col="gem_group" \
  data.kwargs.pert_col="gene" \
  data.kwargs.cell_type_key="cell_line" \
  data.kwargs.control_pert="non-targeting" \
  training.max_steps=80000 \
  training.ckpt_every_n_steps=2000 \
  training.batch_size=${BATCHSIZE} \
  training.lr=1e-3 \
  model.kwargs.cell_set_len=${CELL_SET_LEN} \
  model.kwargs.hidden_dim=${HIDDEN_DIM} \
  model.kwargs.batch_encoder=True \
  model=state \
  wandb.tags='["'"${RUN_ID}"'", "SE_R_Rk563", "SE"]' \
  wandb.entity="cu_0055" \
  wandb.project="finetune_state" \
  output_dir="results/${RUN_ID}/SE_R_Rk562_4" \
  name="SE_${RUN_ID}_R_Rk562_4" &
wait

