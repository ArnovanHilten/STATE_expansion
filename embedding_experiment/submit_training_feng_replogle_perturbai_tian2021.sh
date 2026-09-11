#!/bin/bash
#SBATCH --job-name=feng_replogle_tian
#SBATCH --nodes=1
#SBATCH --cpus-per-task=150
#SBATCH --account=cu_0055
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00
#SBATCH --mem=1800GB
#SBATCH --output=feng_replogle_tian_log_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/users/hilarn/55_cu_0055/dockers/latest/dcai_test+docker_test+state-expansion.sqsh
#
# 2x2 sweep, same shape as submit_training.sh/submit_training_neurons.sh in
# this directory: dataset composition (perturbai_wholebrain in vs. out) x
# learning rate (1e-4 vs 1e-5) = 4 runs, 2 GPUs each (all 8 requested GPUs
# used, unlike the 4x1-GPU example -- this corpus is bigger than a single
# dataset, so 1 GPU/run isn't enough to be worth doing 4-way at all).
#
# Both TOMLs share the same train/val/test design (see either file's own
# header comment): train on feng2025 + replogle + tian2021 (+ optionally
# perturbai_wholebrain), validate+test both carved from tian2019.neuron's
# real (unfiltered-on-perturbation-effect) gene panel via [fewshot] --
# they differ ONLY in whether perturbai_wholebrain is included.

# =========================
# Configurable run number
# =========================
RUN_ID="41"   # <--- change this per batch of runs

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

# W&B config: private server + cert bundle from host
export WANDB_BASE_URL="https://wandb.gefion.dcai.dk"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "NODELIST=${SLURM_NODELIST}"
echo "GPUS_REQUESTED=${SLURM_JOB_GPUS}"
echo "RUN_ID=${RUN_ID}"

cd /dcai/users/hilarn/55_cu_0055/code/enhance_state

# =========================
# Shared config
# =========================

HIDDEN_DIM=128
CELL_SET_LEN=32
BATCHSIZE=64

TOML_WITH_PERTURBAI="/dcai/projects01/cu_0055/data/datasplits/feng_replogle_perturai_tian2021.toml"
TOML_BASELINE="/dcai/projects01/cu_0055/data/datasplits/feng_replogle_tian2021.toml"

SHARED_ARGS="
  data.kwargs.num_workers=30
  data.kwargs.embed_key=X_state
  data.kwargs.output_space=gene
  data.kwargs.batch_col=gem_group
  data.kwargs.pert_col=gene
  data.kwargs.cell_type_key=cell_line
  data.kwargs.control_pert=non-targeting
  experiment.num_gpus_per_node=2
  training.max_steps=200000
  training.ckpt_every_n_steps=2000
  training.batch_size=${BATCHSIZE}
  model.kwargs.cell_set_len=${CELL_SET_LEN}
  model.kwargs.hidden_dim=${HIDDEN_DIM}
  model.kwargs.transformer_backbone_kwargs.num_attention_heads=8
  model.kwargs.transformer_backbone_kwargs.num_key_value_heads=8
  model.kwargs.transformer_backbone_kwargs.head_dim=16
  model.kwargs.batch_encoder=True
  model=state
  wandb.entity=cu_0055
  wandb.project=state_qc_emb
"

# =========================
# Runs WITH perturbai_wholebrain (GPUs 0-1, 2-3, LR sweep)
# =========================

CUDA_VISIBLE_DEVICES=0,1 state tx train \
  ${SHARED_ARGS} \
  data.kwargs.toml_config_path="${TOML_WITH_PERTURBAI}" \
  training.lr=1e-4 \
  wandb.tags='["'"${RUN_ID}"'", "with_perturbai", "lr1e-4"]' \
  output_dir="results/${RUN_ID}/with_perturbai_lr1e-4" \
  name="with_perturbai_${RUN_ID}_lr1e-4" &

CUDA_VISIBLE_DEVICES=2,3 state tx train \
  ${SHARED_ARGS} \
  data.kwargs.toml_config_path="${TOML_WITH_PERTURBAI}" \
  training.lr=1e-5 \
  wandb.tags='["'"${RUN_ID}"'", "with_perturbai", "lr1e-5"]' \
  output_dir="results/${RUN_ID}/with_perturbai_lr1e-5" \
  name="with_perturbai_${RUN_ID}_lr1e-5" &

# =========================
# Baseline runs WITHOUT perturbai_wholebrain (GPUs 4-5, 6-7, LR sweep)
# =========================

CUDA_VISIBLE_DEVICES=4,5 state tx train \
  ${SHARED_ARGS} \
  data.kwargs.toml_config_path="${TOML_BASELINE}" \
  training.lr=1e-4 \
  wandb.tags='["'"${RUN_ID}"'", "baseline", "lr1e-4"]' \
  output_dir="results/${RUN_ID}/baseline_lr1e-4" \
  name="baseline_${RUN_ID}_lr1e-4" &

CUDA_VISIBLE_DEVICES=6,7 state tx train \
  ${SHARED_ARGS} \
  data.kwargs.toml_config_path="${TOML_BASELINE}" \
  training.lr=1e-5 \
  wandb.tags='["'"${RUN_ID}"'", "baseline", "lr1e-5"]' \
  output_dir="results/${RUN_ID}/baseline_lr1e-5" \
  name="baseline_${RUN_ID}_lr1e-5" &

wait
