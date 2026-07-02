#!/usr/bin/env bash
set -euo pipefail

WAN_BACKBONE="ti2v"
WAN_MODEL_ID="Wan-AI/Wan2.2-TI2V-5B"
TRAIN_TYPE="full"
TRAIN_STAGE="${TRAIN_STAGE:-base}"
RESUME_MODEL="${RESUME_MODEL:-./checkpoints/world_teacher/bridgedatav2_pretrain}"
RESUME_STATE="${RESUME_STATE:-}"
RESUME_WEIGHTS_ONLY="${RESUME_WEIGHTS_ONLY:-1}"
ACTION_HEAD="${ACTION_HEAD:-dit}"
ACTION_MODE="${ACTION_MODE:-libero}"
TRAIN_METAS="${TRAIN_METAS:-./data/vlabench_primitive_ft_lerobot/meta/info.json}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29536}"
FSDP_FLAGS=""
SPARSE_T2V="${SPARSE_T2V:-1}"
DELTA_T_SCALE="${DELTA_T_SCALE:-30}"
USE_FUTURE_TOKEN="${USE_FUTURE_TOKEN:-0}"
FUTURE_PROMPT_SOURCE="${FUTURE_PROMPT_SOURCE:-none}"
SCHEDULED_SAMPLING="${SCHEDULED_SAMPLING:-0}"
FUTURE_INDEX="${FUTURE_INDEX:-30}"
CHUNK_SIZE="${CHUNK_SIZE:-30}"
: "${VLABENCH_FRONT_VIEW:=image}"
: "${VLABENCH_FUTURE_VIEW:=${VLABENCH_FRONT_VIEW}}"

validate_vlabench_view() {
  case "$1" in
    image|second_image) ;;
    *)
      echo "[world_teacher-vlabench-imageonly] unsupported VLABENCH view '$1' (expected: image or second_image)" >&2
      exit 1
      ;;
  esac
}

validate_vlabench_view "${VLABENCH_FRONT_VIEW}"
validate_vlabench_view "${VLABENCH_FUTURE_VIEW}"

if [[ "${VLABENCH_FRONT_VIEW}" == "image" ]]; then
  VLABENCH_SIDE_VIEW="second_image"
else
  VLABENCH_SIDE_VIEW="image"
fi
VLABENCH_CAMERA_ORDER="${VLABENCH_FRONT_VIEW},wrist_image,${VLABENCH_SIDE_VIEW}"
VIEW_TAG="${VLABENCH_FRONT_VIEW}"
if [[ "${VLABENCH_FUTURE_VIEW}" != "${VLABENCH_FRONT_VIEW}" ]]; then
  VIEW_TAG="${VIEW_TAG}_future-${VLABENCH_FUTURE_VIEW}"
fi
DEFAULT_DATASET_TAG="posttrain_vlabench_imageonly_from_bridge"
DEFAULT_OUTPUT_DIR="./checkpoints/world_teacher_vlabench_imageonly"
if [[ "${VIEW_TAG}" != "image" ]]; then
  DEFAULT_DATASET_TAG="${DEFAULT_DATASET_TAG}_${VIEW_TAG}"
  DEFAULT_OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}_${VIEW_TAG}"
fi
DATASET_TAG="${DATASET_TAG:-${DEFAULT_DATASET_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
elif [[ -n "${GPUS:-}" ]]; then
  GPUS="${GPUS}"
else
  GPUS=1
fi
BATCH_SIZE="${BATCH_SIZE:-24}"
: "${LEARNING_RATE:=5e-5}"
: "${LR_SCHEDULER_TYPE:=cosine}"
: "${WARMUP_RATIO:=0.05}"
: "${ACTION_LOSS_WEIGHT:=0.0}"
: "${ACTION_SUP_WEIGHT:=0.0}"
: "${IMAGE_LOSS_WEIGHT:=1.0}"
: "${RECON_LOSS_WEIGHT:=0.6}"
: "${ITERS:=120000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
DEEPSPEED_ZERO_STAGE="${DEEPSPEED_ZERO_STAGE:-2}"
DEEPSPEED_OFFLOAD="${DEEPSPEED_OFFLOAD:-0}"
USE_WANDB="${USE_WANDB:-1}"
FREEZE_WAN_NON_DIT="${FREEZE_WAN_NON_DIT:-1}"
PARAM_REPORT_MAX_NAMES="${PARAM_REPORT_MAX_NAMES:-30}"
if [[ "${GPUS}" -le 1 && "${USE_DEEPSPEED}" == "1" ]]; then USE_DEEPSPEED=0; fi
SPARSE_T2V_FLAGS=""
if [[ "${SPARSE_T2V}" == "1" ]]; then SPARSE_T2V_FLAGS="--sparse_t2v --delta_t_scale ${DELTA_T_SCALE}"; fi
SCHED_FLAGS="--lr_scheduler_type ${LR_SCHEDULER_TYPE} --warmup_ratio ${WARMUP_RATIO} --min_lr_ratio ${MIN_LR_RATIO}"
if [[ "${WARMUP_STEPS}" -gt 0 ]]; then SCHED_FLAGS="${SCHED_FLAGS} --warmup_steps ${WARMUP_STEPS}"; fi
DEEPSPEED_FLAGS=""
if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  DEEPSPEED_FLAGS="--deepspeed --deepspeed_zero_stage ${DEEPSPEED_ZERO_STAGE}"
  if [[ "${DEEPSPEED_OFFLOAD}" == "1" ]]; then DEEPSPEED_FLAGS="${DEEPSPEED_FLAGS} --deepspeed_offload"; fi
fi
WANDB_FLAGS=""
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_FLAGS="--wandb"
  WANDB_HOME="${WANDB_HOME:-${HOME:-}}"
  export HOME="${WANDB_HOME}"
  export NETRC="${NETRC:-${WANDB_HOME}/.netrc}"
fi
RETURN_FUTURE_IMAGE_FLAG="--return_future_image"
MODEL_FLAGS="--init_from_wan_only --wan_model_id ${WAN_MODEL_ID}"
if [[ -n "${RESUME_MODEL}" ]]; then MODEL_FLAGS="--models ${RESUME_MODEL}"; fi
RESUME_FLAGS=""
if [[ -n "${RESUME_STATE}" ]]; then RESUME_FLAGS="${RESUME_FLAGS} --resume_from ${RESUME_STATE}"; fi
if [[ "${RESUME_WEIGHTS_ONLY}" == "1" ]]; then RESUME_FLAGS="${RESUME_FLAGS} --resume_weights_only"; fi
if [[ "${FREEZE_WAN_NON_DIT}" == "1" ]]; then RESUME_FLAGS="${RESUME_FLAGS} --freeze_wan_non_dit"; fi
mkdir -p ./log_a800
LOG_NAME="${LOG_NAME:-./log_a800/train_world_teacher_posttrain_vlabench_imageonly_${VIEW_TAG}_ah${ACTION_HEAD}_${TRAIN_TYPE}_wb${WAN_BACKBONE}_bs${BATCH_SIZE}.log}"
nohup accelerate launch --main_process_port "${MAIN_PROCESS_PORT}" scripts/train_world_teacher.py \
  ${MODEL_FLAGS} \
  ${RESUME_FLAGS} \
  --wan_backbone "${WAN_BACKBONE}" \
  --train_metas_path "${TRAIN_METAS}" \
  --output_dir "${OUTPUT_DIR}" \
  --vlabench_camera_order "${VLABENCH_CAMERA_ORDER}" \
  --vlabench_future_view "${VLABENCH_FUTURE_VIEW}" \
  ${FSDP_FLAGS} \
  ${DEEPSPEED_FLAGS} \
  ${WANDB_FLAGS} \
  --wandb_project "WorldTeacher" \
  --wandb_run_name "world_teacher-${DATASET_TAG}-${WAN_BACKBONE}-${TRAIN_TYPE}-g${GPUS}-lr${LEARNING_RATE}-bs${BATCH_SIZE}" \
  ${SPARSE_T2V_FLAGS} \
  ${SCHED_FLAGS} \
  --action_mode "${ACTION_MODE}" \
  --action_head "${ACTION_HEAD}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --iters "${ITERS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --future_index "${FUTURE_INDEX}" \
  --num_actions "${CHUNK_SIZE}" \
  --use_future_token "${USE_FUTURE_TOKEN}" \
  --future_prompt_source "${FUTURE_PROMPT_SOURCE}" \
  --wan_height 256 \
  --wan_width 256 \
  --action_loss_weight "${ACTION_LOSS_WEIGHT}" \
  --action_supervised_weight "${ACTION_SUP_WEIGHT}" \
  --image_loss_weight "${IMAGE_LOSS_WEIGHT}" \
  --recon_loss_weight "${RECON_LOSS_WEIGHT}" \
  --param_report_max_names "${PARAM_REPORT_MAX_NAMES}" \
  ${RETURN_FUTURE_IMAGE_FLAG} \
  --image_only_training 2>&1 | tee "${LOG_NAME}" &
