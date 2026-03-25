#!/usr/bin/env bash
set -euo pipefail

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_IB_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR=${ROOT_DIR:-${SCRIPT_DIR}}
LATENT_LMDB=${LATENT_LMDB:-${ROOT_DIR}/data/train_vae_latents_lmdb.lmdb}
FID_STATS=${FID_STATS:-${ROOT_DIR}/data/adm_in256_stats.npz}
AE_CKPT=${AE_CKPT:-${ROOT_DIR}/checkpoints/sd_vae_ft_ema.pt}

MODEL_SIZE=${MODEL_SIZE:-XL}
MODEL_VARIANT=${MODEL_VARIANT:-}
MODEL_CKPT_OVERRIDE=${MODEL_CKPT_OVERRIDE:-}

ENABLE_KD=${ENABLE_KD:-0}
KD_WEIGHT=${KD_WEIGHT:-0.1}
KD_LOSS_TYPE=${KD_LOSS_TYPE:-mse}
TEACHER_USE_EMA=${TEACHER_USE_EMA:-true}

RESUME_ARGS="training.resume.whole_state=false training.resume.allow_missing_extra_state_on_start=true"
KD_ARGS=""

case "$MODEL_SIZE" in
  B)
    MODEL_DESC=temf-latentspace-B-2-cfg-training-finetune
    MODEL_DIM=768
    MODEL_BLOCKS=12
    MODEL_HEADS=12
    ;;
  M)
    MODEL_DESC=temf-latentspace-M-2-cfg-training-finetune
    MODEL_DIM=896
    MODEL_BLOCKS=16
    MODEL_HEADS=14
    ;;
  L)
    MODEL_DESC=temf-latentspace-L-2-cfg-training-finetune
    MODEL_DIM=1024
    MODEL_BLOCKS=24
    MODEL_HEADS=16
    ;;
  XL)
    if [ "$MODEL_VARIANT" = "plus" ]; then
      MODEL_DESC=temf-latentspace-XL-2-plus-cfg-training-finetune
    else
      MODEL_DESC=temf-latentspace-XL-2-cfg-training-finetune
    fi
    MODEL_DIM=1152
    MODEL_BLOCKS=28
    MODEL_HEADS=16
    ;;
  *)
    echo "Unsupported MODEL_SIZE: $MODEL_SIZE. Expected one of: B, M, L, XL" >&2
    exit 1
    ;;
esac

MODEL_CKPT="${MODEL_CKPT_OVERRIDE}"
if [ -n "$MODEL_CKPT" ]; then
  if [ ! -f "$MODEL_CKPT" ]; then
    echo "Missing checkpoint: $MODEL_CKPT" >&2
    echo "Set MODEL_CKPT_OVERRIDE=/path/to/temf_ckpt.pt" >&2
    exit 1
  fi
  RESUME_ARGS="training.resume.on_start_ckpt_path=${MODEL_CKPT} ${RESUME_ARGS}"
  DEFAULT_LR=5e-6
else
  DEFAULT_LR=1e-4
fi

if [ ! -e "$LATENT_LMDB" ]; then
  echo "Missing latent dataset path: $LATENT_LMDB" >&2
  exit 1
fi

if [ -d "$LATENT_LMDB" ]; then
  if [ ! -f "$LATENT_LMDB/data.mdb" ]; then
    echo "Invalid LMDB directory: $LATENT_LMDB (missing data.mdb)" >&2
    exit 1
  fi
fi

if [ ! -f "$FID_STATS" ]; then
  echo "Missing fid stats file: $FID_STATS" >&2
  exit 1
fi

if [ ! -f "$AE_CKPT" ]; then
  echo "Missing autoencoder checkpoint: $AE_CKPT" >&2
  exit 1
fi

if [ "$ENABLE_KD" = "1" ]; then
  TEACHER_CKPT=${TEACHER_CKPT:-}
  if [ -z "$TEACHER_CKPT" ]; then
    echo "ENABLE_KD=1 requires TEACHER_CKPT=/path/to/teacher.pt" >&2
    exit 1
  fi
  if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Missing teacher checkpoint: $TEACHER_CKPT" >&2
    exit 1
  fi
  KD_ARGS="loss.teacher.weights.prediction=${KD_WEIGHT} loss.teacher.prediction_loss_type=${KD_LOSS_TYPE} loss.teacher.ckpt.snapshot_path=${TEACHER_CKPT} loss.teacher.ckpt.use_ema=${TEACHER_USE_EMA}"
  MODEL_DESC=${MODEL_DESC}-kd
fi

LR=${LR:-$DEFAULT_LR}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-13221}
EXP_DIR=${EXP_DIR:-${ROOT_DIR}/experiments/${MODEL_DESC}}

torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" src/train.py \
    desc="${MODEL_DESC}" \
    hydra.run.dir="${EXP_DIR}" \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled \
    model=dit_temf_ldm \
    loss=temf \
    dataset=imagenet \
    sampling=recflow \
    env=local \
    pre_extract_latents=none \
    quiet_launch=true \
    num_nodes=1 \
    wandb.tags='["temf-latentspace"]' \
    training.metrics='[fid50k_full]' \
    training.metrics_rare='[]' \
    training.metrics_final='[fid50k_full]' \
    training.fid_statistics_file="${FID_STATS}" \
    '+training.metrics_extra_sampling_cfg_overwrites.NFE_2_recflow_sampling={num_steps:2}' \
    '+training.metrics_extra_sampling_cfg_overwrites.NFE_2_consistency_sampling={num_steps:2,enable_consistency_sampling:true}' \
    training.max_steps=1300000 \
    training.freqs.snapshot_latest=1000 \
    training.freqs.snapshot=10000 \
    training.freqs.loss_per_sigma=null \
    training.freqs.loss_val=null \
    training.freqs.metrics=100 \
    training.freqs.metrics_rare=50 \
    training.traj_len_for_vis_gen=1 \
    training.num_vis_samples=16 \
    training.dp.strategy=ddp \
    ${RESUME_ARGS} \
    ${KD_ARGS} \
    sampling.num_steps=1 \
    sampling.sigma_noise=1.0 \
    sampling.enable_trajectory_sampling=true \
    sampling.enable_consistency_sampling=false \
    dataset.name=imagenet_folder \
    dataset.resolution='[1,256,256]' \
    dataset.batch_gpu=32 \
    dataset.batch_size=256 \
    dataset.test_batch_gpu=1 \
    dataset.use_val_data_for_eval_stream=false \
    dataset.print_traceback=true \
    dataset.print_exceptions=true \
    dataset.data_type=image_latent \
    dataset.src="${LATENT_LMDB}" \
    dataset.src_val=null \
    model.use_precomputed_latents=true \
    model.num_blocks="${MODEL_BLOCKS}" \
    model.dim="${MODEL_DIM}" \
    model.num_heads="${MODEL_HEADS}" \
    model.label_dropout=0.1 \
    model.tokenizer.resolution='[1,2,2]' \
    model.optim.class_name=torch.optim.Adam \
    model.optim.betas='[0.9,0.95]' \
    model.optim.weight_decay=0.0 \
    model.optim.lr="${LR}" \
    model.dropout=0.0 \
    model.use_ema=true \
    model.ema_rampup_ratio=null \
    model.checkpointing=false \
    model.sigma_data=1.0 \
    model.use_fused_modulation=false \
    model.lr_scheduler.num_warmup_steps=0 \
    model.lr_scheduler.final_lr="${LR}" \
    model.autoencoder_ckpt.snapshot_path="${AE_CKPT}" \
    model.autoencoder_ckpt.convert_params_to_buffers=false \
    "$@"
