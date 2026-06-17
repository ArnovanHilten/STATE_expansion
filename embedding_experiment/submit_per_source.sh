#!/bin/bash
#SBATCH --job-name=state_per_source
#SBATCH --nodes=1
#SBATCH --account=cu_0055
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00
#SBATCH --mem=1800GB
#SBATCH --output=state_per_source_%j.log
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs
#SBATCH --container-image=/dcai/users/hilarn/55_cu_0055/dockers/latest/dcai_test+docker_test+state-expansion.sqsh

# =========================
# Batch selection
# Submit this script 3 times with BATCH=0, 1, 2
#   BATCH=0 → sources  0-7  (GTEx … FunCoup)
#   BATCH=1 → sources  8-15 (HIPPIE … consensus)
#   BATCH=2 → sources 16-21 (Reactome … SynGO)
# =========================
BATCH=0   # <--- change to 0, 1, or 2

RUN_ID="21"   # <--- change this per batch of runs

# All 22 source names (must match filenames produced by make_per_source_npz.py)
SOURCES=(
  GTEx
  GWASAtlas_signed
  ESM-2
  DepMap
  CellPainting
  STRING
  BioGRID
  FunCoup
  HIPPIE
  IntAct
  OmniPath
  MINT
  CORUM
  DIP
  ComplexPortal
  ELM
  consensus
  Reactome
  GeneOntology
  MSigDB
  WikiPathways
  SynGO
)

START=$((BATCH * 8))
END=$((START + 8))
N=${#SOURCES[@]}
if [ $END -gt $N ]; then END=$N; fi

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
echo "RUN_ID=${RUN_ID}  BATCH=${BATCH}  sources ${START}-$((END-1))"

cd /dcai/users/hilarn/55_cu_0055/code/enhance_state

# =========================
# Shared config
# =========================

HIDDEN_DIM=128
CELL_SET_LEN=32
BATCHSIZE=64
EMB_DIR="/dcai/users/hilarn/55_cu_0055/data/embeddings/STATE_embedddings/per_source"

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
  training.lr=1e-4
  model.kwargs.cell_set_len=${CELL_SET_LEN}
  model.kwargs.hidden_dim=${HIDDEN_DIM}
  model.kwargs.transformer_backbone_kwargs.num_attention_heads=8
  model.kwargs.transformer_backbone_kwargs.num_key_value_heads=8
  model.kwargs.transformer_backbone_kwargs.head_dim=16
  model.kwargs.batch_encoder=True
  model.kwargs.use_qc_cross_attn=true
  model.kwargs.qc_mode=per_source
  model.kwargs.cross_attn_freq=3
  model=state
  wandb.entity=cu_0055
  wandb.project=state_per_source_emb
"

# =========================
# Launch one run per source on its own GPU
# =========================

GPU=0
for i in $(seq $START $((END - 1))); do
  SRC=${SOURCES[$i]}
  SAFE_SRC=${SRC//-/_}   # replace hyphens for use in run name

  (
    CUDA_VISIBLE_DEVICES=$GPU state tx train \
      ${SHARED_ARGS} \
      model.kwargs.qc_emb_path="${EMB_DIR}/${SRC}.npz" \
      wandb.tags='["'"${RUN_ID}"'", "per_source", "'"${SRC}"'"]' \
      output_dir="results/${RUN_ID}/src_${SAFE_SRC}" \
      name="src_${RUN_ID}_${SAFE_SRC}"

    echo "Training done for ${SRC}, running predict..."

    CUDA_VISIBLE_DEVICES=$GPU state tx predict \
      --output-dir "results/${RUN_ID}/src_${SAFE_SRC}/src_${RUN_ID}_${SAFE_SRC}" \
      --checkpoint "best.ckpt"
  ) &

  GPU=$((GPU + 1))
done

wait
