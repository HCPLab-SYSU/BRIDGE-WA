#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${WORLD_TEACHER_PATH:=./checkpoints/world_teacher_vlabench}"
: "${TRAIN_METAS_PATH:=./data/vlabench_primitive_ft_lerobot/meta/info.json}"
: "${OUTPUT_DIR:=./cache/bridge_wa/vlabench_primitive_ft_lerobot}"
: "${BATCH_SIZE:=128}"
: "${TEACHER_FUTURE_STEPS:=10}"
: "${NUM_ACTIONS:=30}"
: "${ACTION_MODE:=libero}"
: "${VLABENCH_FRONT_VIEW:=image}"
: "${VLABENCH_FUTURE_VIEW:=${VLABENCH_FRONT_VIEW}}"
: "${KEEP_STATIC_FRAMES:=1}"
: "${NUM_WORKERS:=32}"
: "${FLOW_WORKERS:=64}"
: "${CACHE_GRANULARITY:=sample}"
: "${LIMIT:=0}"

validate_vlabench_view() {
  case "$1" in
    image|second_image) ;;
    *)
      echo "[bridge_wa-cache-vlabench] unsupported VLABENCH view '$1' (expected: image or second_image)" >&2
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

mkdir -p "$(dirname "${OUTPUT_DIR}")" ./log_a800
LOG_FILE="./log_a800/precompute_bridge_wa_cache_vlabench.log"
CMD=(
  python -u scripts/precompute_bridge_wa_cache_all.py
  --world_teacher_path "${WORLD_TEACHER_PATH}"
  --train_metas_path "${TRAIN_METAS_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --teacher_future_steps "${TEACHER_FUTURE_STEPS}"
  --num_actions "${NUM_ACTIONS}"
  --action_mode "${ACTION_MODE}"
  --vlabench_camera_order "${VLABENCH_CAMERA_ORDER}"
  --vlabench_future_view "${VLABENCH_FUTURE_VIEW}"
  --num_workers "${NUM_WORKERS}"
  --flow_workers "${FLOW_WORKERS}"
  --cache_granularity "${CACHE_GRANULARITY}"
)
if [[ "${KEEP_STATIC_FRAMES}" == "1" ]]; then
  CMD+=(--keep_static_frames)
fi
printf '[bridge_wa-cache-vlabench] cmd=' | tee -a "${LOG_FILE}"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_FILE}"
printf '\n' | tee -a "${LOG_FILE}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_SHARDS="${#GPU_IDS[@]}"
if [[ "${NUM_SHARDS}" -le 1 ]]; then
  if [[ "${LIMIT}" != "0" ]]; then
    CMD+=(--limit "${LIMIT}")
  fi
  nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
  echo "[bridge_wa-cache-vlabench] pid=$!" | tee -a "${LOG_FILE}"
  echo "[bridge_wa-cache-vlabench] log=${LOG_FILE}" | tee -a "${LOG_FILE}"
else
  echo "[bridge_wa-cache-vlabench] launching ${NUM_SHARDS} shards" | tee -a "${LOG_FILE}"
  for SHARD_ID in "${!GPU_IDS[@]}"; do
    SHARD_LOG="${LOG_FILE%.log}.rank${SHARD_ID}.log"
    SHARD_CMD=("${CMD[@]}" --num_shards "${NUM_SHARDS}" --shard_id "${SHARD_ID}")
    if [[ "${LIMIT}" != "0" ]]; then
      BASE_LIMIT=$(( LIMIT / NUM_SHARDS ))
      EXTRA=$(( LIMIT % NUM_SHARDS ))
      SHARD_LIMIT=${BASE_LIMIT}
      if [[ "${SHARD_ID}" -lt "${EXTRA}" ]]; then
        SHARD_LIMIT=$(( SHARD_LIMIT + 1 ))
      fi
      if [[ "${SHARD_LIMIT}" -gt 0 ]]; then
        SHARD_CMD+=(--limit "${SHARD_LIMIT}")
      fi
    fi
    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS[$SHARD_ID]}" PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${SHARD_CMD[@]}" >> "${SHARD_LOG}" 2>&1 &
    echo "[bridge_wa-cache-vlabench] rank=${SHARD_ID} gpu=${GPU_IDS[$SHARD_ID]} pid=$! log=${SHARD_LOG}" | tee -a "${LOG_FILE}"
  done
fi
