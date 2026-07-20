#!/bin/bash
#SBATCH --job-name=baselines_k562
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=cu_0055
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --mem=400GB
#SBATCH --output=baselines_k562_%j.log
#SBATCH --container-image=/dcai/projects01/cu_0055/notebooks/state_expansion/state_expansion.sqsh
#SBATCH --container-mounts=/dcai:/dcai,/etc/ssl/certs:/etc/ssl/certs

unset LMOD_CMD

export WANDB_BASE_URL="https://wandb.gefion.dcai.dk"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "NODELIST=${SLURM_NODELIST}"

cd /dcai/projects/cu_0055/code/state_expansion/embedding_experiment

# Common data args
COMMON_DATA="
  data.kwargs.toml_config_path=toml/SE_R_Rk562.toml
  data.kwargs.embed_key=X_hvg
  data.kwargs.output_space=gene
  data.kwargs.batch_col=gem_group
  data.kwargs.pert_col=gene
  data.kwargs.cell_type_key=cell_line
  data.kwargs.control_pert=non-targeting
  data.kwargs.num_workers=8
"

# cell_set_len=1 so the on_fit_start accumulation loop sees individual cell
# vectors rather than padded sentence tensors.
COMMON_MODEL="
  model.kwargs.cell_set_len=1
  model.kwargs.cell_sentence_len=1
  model.kwargs.hidden_dim=256
"

# max_steps=2: statistics are accumulated in on_fit_start before any gradient
# step; we just need a couple of steps to trigger last.ckpt to be written.
COMMON_TRAINING="
  training.max_steps=2
  training.ckpt_every_n_steps=1
  training.batch_size=1024
"

COMMON_WANDB="
  wandb.entity=cu_0055
  wandb.project=state_qc_emb
"

# ── perturbmean baseline (GPU 0) ──────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 state tx train \
  ${COMMON_DATA} \
  ${COMMON_MODEL} \
  ${COMMON_TRAINING} \
  ${COMMON_WANDB} \
  model=perturb_mean \
  output_dir="results/baselines" \
  name="perturbmean_k562" \
  wandb.tags='["baseline", "perturbmean", "k562"]' &

# ── contextmean baseline (GPU 1) ─────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 state tx train \
  ${COMMON_DATA} \
  ${COMMON_MODEL} \
  ${COMMON_TRAINING} \
  ${COMMON_WANDB} \
  model=context_mean \
  output_dir="results/baselines" \
  name="contextmean_k562" \
  wandb.tags='["baseline", "contextmean", "k562"]' &

wait
echo "Training complete. Running predictions..."

# ── predict: perturbmean ──────────────────────────────────────────────────────
state tx predict \
  --output-dir "results/baselines/perturbmean_k562" \
  --checkpoint last.ckpt \
  --profile full &

# ── predict: contextmean ─────────────────────────────────────────────────────
state tx predict \
  --output-dir "results/baselines/contextmean_k562" \
  --checkpoint last.ckpt \
  --profile full &

wait
echo "All done. Results in results/baselines/"
