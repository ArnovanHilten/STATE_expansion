#!/bin/bash
#SBATCH --job-name=state_qc_emb_R_Rk562
#SBATCH --nodes=1
#SBATCH --account=cu_0055
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --mem=1800GB
#SBATCH --output=state_qc_emb_R_Rk562_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/projects01/cu_0055/notebooks/state_expansion/state_expansion.sqsh

# =========================
# Configurable run number
# =========================
RUN_ID="1"   # <--- change this per batch of runs


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

cd /dcai/projects/cu_0055/code/state_expansion/embedding_experiment

# =========================
# Shared config
# =========================

HIDDEN_DIM=128
CELL_SET_LEN=32
BATCHSIZE=64
QC_EMB_PATH="/dcai/users/hilarn/55_cu_0055/data/embeddings/STATE_embedddings/gene_embeddings_combined.npz"

SHARED_ARGS="
  data.kwargs.toml_config_path=toml/SE_R_Rk562.toml
  data.kwargs.num_workers=30
  data.kwargs.embed_key=X_state
  data.kwargs.output_space=gene
  data.kwargs.batch_col=gem_group
  data.kwargs.pert_col=gene
  data.kwargs.cell_type_key=cell_line
  data.kwargs.control_pert=non-targeting
  training.max_steps=80000
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
# Runs WITH QC cross-attention (GPUs 0-1, LR sweep)
# =========================

CUDA_VISIBLE_DEVICES=0 state tx train \
  ${SHARED_ARGS} \
  model.kwargs.use_qc_cross_attn=true \
  model.kwargs.qc_emb_path="${QC_EMB_PATH}" \
  model.kwargs.qc_mode=per_source \
  model.kwargs.cross_attn_freq=3 \
  training.lr=1e-4 \
  wandb.tags='["'"${RUN_ID}"'", "qc_emb", "lr1e-4"]' \
  output_dir="results/${RUN_ID}/qc_emb_lr1e-4" \
  name="qc_emb_${RUN_ID}_lr1e-4" &

CUDA_VISIBLE_DEVICES=1 state tx train \
  ${SHARED_ARGS} \
  model.kwargs.use_qc_cross_attn=true \
  model.kwargs.qc_emb_path="${QC_EMB_PATH}" \
  model.kwargs.qc_mode=per_source \
  model.kwargs.cross_attn_freq=3 \
  training.lr=1e-5 \
  wandb.tags='["'"${RUN_ID}"'", "qc_emb", "lr1e-5"]' \
  output_dir="results/${RUN_ID}/qc_emb_lr1e-5" \
  name="qc_emb_${RUN_ID}_lr1e-5" &

# =========================
# Baseline runs WITHOUT QC cross-attention (GPUs 2-3)
# =========================

CUDA_VISIBLE_DEVICES=2 state tx train \
  ${SHARED_ARGS} \
  model.kwargs.use_qc_cross_attn=false \
  training.lr=1e-4 \
  wandb.tags='["'"${RUN_ID}"'", "baseline", "lr1e-4"]' \
  output_dir="results/${RUN_ID}/baseline_lr1e-4" \
  name="baseline_${RUN_ID}_lr1e-4" &

CUDA_VISIBLE_DEVICES=3 state tx train \
  ${SHARED_ARGS} \
  model.kwargs.use_qc_cross_attn=false \
  training.lr=1e-5 \
  wandb.tags='["'"${RUN_ID}"'", "baseline", "lr1e-5"]' \
  output_dir="results/${RUN_ID}/baseline_lr1e-5" \
  name="baseline_${RUN_ID}_lr1e-5" &

wait
