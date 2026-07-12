#!/usr/bin/env bash
set -euo pipefail

export USE_TF="${USE_TF:-0}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

: "${RUN_IN_BACKGROUND:=1}"
: "${LOG_DIR:=./log}"
: "${LOG_FILE:=}"
: "${_BRIDGE_WA_VLABENCH_EVAL_CHILD:=0}"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${MODEL_PATH:=}"
: "${PROCESSOR_PATH:=}"
: "${HOST:=0.0.0.0}"
: "${ADVERTISE_HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${DEVICE:=cuda}"
: "${DTYPE:=float32}"
: "${STEPS:=10}"
: "${DOMAIN_ID:=8}"

: "${SERVER_OUTPUT_DIR:=./logs/bridge_wa_vlabench_server}"
: "${CLIENT_OUTPUT_DIR:=./logs/bridge_wa_vlabench_eval}"
: "${VLABENCH_EVAL_DIR:=evaluation/vlabench}"
: "${VLABENCH_CONDA_ENV:=vlabench}"
: "${SERVER_CONDA_ENV:=bridge-wa}"
: "${CONDA_SH:=}"
: "${PYTHON_BIN:=python}"
: "${SERVER_PYTHON_BIN:=}"
: "${CLIENT_PYTHON_BIN:=python}"

: "${RUN_CLIENT:=1}"
: "${PREFLIGHT_ONLY:=0}"
: "${STRICT_BRIDGE_WA_OFFICIAL:=1}"
: "${EVAL_TRACKS:=track_1_in_distribution track_2_cross_category track_3_common_sense track_4_semantic_instruction track_6_unseen_texture}"
: "${TRACK_SOURCE_JSON:=}"
: "${METRICS:=success_rate intention_score progress_score}"
: "${N_EPISODE:=10}"
: "${RESUME:=0}"
: "${SAVE_VIDEO:=${VISUALIZATION:-0}}"
: "${WAIT_TIMEOUT_SECONDS:=300}"
: "${STOP_SERVER_AFTER_EVAL:=0}"

if [[ "${_BRIDGE_WA_VLABENCH_EVAL_CHILD}" != "1" ]]; then
  case "${RUN_IN_BACKGROUND}" in
    1|true|TRUE|yes|YES|on|ON)
      mkdir -p "${LOG_DIR}"
      if [[ -z "${LOG_FILE}" ]]; then
        TS="$(date +%Y%m%d_%H%M%S)"
        LOG_FILE="${LOG_DIR}/bridge_wa_vlabench_eval_${TS}.log"
      fi
      SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
      echo "[bridge_wa-vlabench-eval] launching in background with nohup"
      echo "[bridge_wa-vlabench-eval] log=${LOG_FILE}"
      nohup env _BRIDGE_WA_VLABENCH_EVAL_CHILD=1 LOG_DIR="${LOG_DIR}" LOG_FILE="${LOG_FILE}" \
        bash "${SCRIPT_PATH}" "$@" >> "${LOG_FILE}" 2>&1 &
      echo "[bridge_wa-vlabench-eval] pid=$!"
      exit 0
      ;;
    0|false|FALSE|no|NO|off|OFF|"")
      ;;
    *)
      echo "[bridge_wa-vlabench-eval] unsupported RUN_IN_BACKGROUND=${RUN_IN_BACKGROUND} (expected: 0|1|true|false)"
      exit 1
      ;;
  esac
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

_CONDA_SOURCED=0
source_conda() {
  if [[ "${_CONDA_SOURCED}" == "1" ]]; then
    return
  fi
  if [[ -n "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
  else
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
  _CONDA_SOURCED=1
}

if [[ -z "${MODEL_PATH}" ]]; then
  for CANDIDATE_MODEL_PATH in \
    "./checkpoints/bridge_wa_vlabench/ckpt-100000" \
    "./models/train/BridgeWA_vlabench5_ablation_main4_20260517_wandb_offline/fco/ckpt-25000"
  do
    if [[ -d "${CANDIDATE_MODEL_PATH}" ]]; then
      MODEL_PATH="${CANDIDATE_MODEL_PATH}"
      break
    fi
  done
fi
if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="./checkpoints/bridge_wa_vlabench/ckpt-100000"
fi

if [[ -z "${SERVER_PYTHON_BIN}" ]]; then
  if [[ -n "${SERVER_CONDA_ENV}" ]]; then
    source_conda
    SERVER_PYTHON_BIN="$(conda run -n "${SERVER_CONDA_ENV}" python -c 'import sys; print(sys.executable)')"
  else
    SERVER_PYTHON_BIN="${PYTHON_BIN}"
  fi
fi

CLIENT_RUNNER=()
if [[ -n "${VLABENCH_CONDA_ENV}" ]]; then
  source_conda
  CLIENT_RUNNER=(conda run --no-capture-output -n "${VLABENCH_CONDA_ENV}" "${CLIENT_PYTHON_BIN}")
else
  CLIENT_RUNNER=("${CLIENT_PYTHON_BIN}")
fi

if [[ ! -d "${VLABENCH_EVAL_DIR}/VLABench/VLABench" ]]; then
  echo "[bridge_wa-vlabench-eval] missing ${VLABENCH_EVAL_DIR}/VLABench/VLABench"
  echo "[bridge_wa-vlabench-eval] install VLABench under ${VLABENCH_EVAL_DIR} or set VLABENCH_EVAL_DIR."
  exit 1
fi

echo "[bridge_wa-vlabench-eval] validating VLABench client environment"
(
  cd "${VLABENCH_EVAL_DIR}"
  "${CLIENT_RUNNER[@]}" check_vlabench_env.py
)

echo "[bridge_wa-vlabench-eval] checking server port ${HOST}:${PORT}"
if ! HOST="${HOST}" PORT="${PORT}" "${SERVER_PYTHON_BIN}" - <<'PY'
import os
import socket
import sys

host = os.environ["HOST"]
port = int(os.environ["PORT"])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    # Match Uvicorn's bind behavior: a recently stopped listener can leave
    # connections in TIME_WAIT without making the port genuinely unavailable.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
except OSError as exc:
    print(
        f"[bridge_wa-vlabench-eval] port {host}:{port} is unavailable: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)
finally:
    sock.close()
PY
then
  echo "[bridge_wa-vlabench-eval] stop the existing process or choose another PORT."
  exit 1
fi

case "${PREFLIGHT_ONLY}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "[bridge_wa-vlabench-eval] preflight complete; PREFLIGHT_ONLY=1, not starting server."
    exit 0
    ;;
  0|false|FALSE|no|NO|off|OFF|"")
    ;;
  *)
    echo "[bridge_wa-vlabench-eval] unsupported PREFLIGHT_ONLY=${PREFLIGHT_ONLY} (expected: 0|1|true|false)"
    exit 1
    ;;
esac

OFFICIAL_EVAL_TRACKS="track_1_in_distribution track_2_cross_category track_3_common_sense track_4_semantic_instruction track_6_unseen_texture"
OFFICIAL_METRICS="success_rate intention_score progress_score"
OFFICIAL_N_EPISODE="10"
if [[ "${STRICT_BRIDGE_WA_OFFICIAL}" == "1" ]]; then
  if [[ "${EVAL_TRACKS}" != "${OFFICIAL_EVAL_TRACKS}" ]]; then
    echo "[bridge_wa-vlabench-eval] strict official protocol requires EVAL_TRACKS='${OFFICIAL_EVAL_TRACKS}', got '${EVAL_TRACKS}'"
    exit 1
  fi
  if [[ "${METRICS}" != "${OFFICIAL_METRICS}" ]]; then
    echo "[bridge_wa-vlabench-eval] strict official protocol requires METRICS='${OFFICIAL_METRICS}', got '${METRICS}'"
    exit 1
  fi
  if [[ "${N_EPISODE}" != "${OFFICIAL_N_EPISODE}" ]]; then
    echo "[bridge_wa-vlabench-eval] strict official protocol requires N_EPISODE=${OFFICIAL_N_EPISODE}, got ${N_EPISODE}"
    exit 1
  fi
fi

mkdir -p "${SERVER_OUTPUT_DIR}" "${CLIENT_OUTPUT_DIR}" "${LOG_DIR}"
SERVER_OUTPUT_DIR_ABS="$(cd "${SERVER_OUTPUT_DIR}" && pwd)"
CLIENT_OUTPUT_DIR_ABS="$(cd "${CLIENT_OUTPUT_DIR}" && pwd)"
if [[ -n "${TRACK_SOURCE_JSON}" && "${TRACK_SOURCE_JSON}" != /* ]]; then
  TRACK_SOURCE_JSON="${REPO_ROOT}/${TRACK_SOURCE_JSON}"
fi
CONNECTION_INFO="${SERVER_OUTPUT_DIR_ABS}/info.json"
SERVER_LOG_FILE="${SERVER_OUTPUT_DIR_ABS}/server.log"
CLIENT_LOG_FILE="${CLIENT_OUTPUT_DIR_ABS}/client.log"
rm -f "${CONNECTION_INFO}"

SERVER_CMD=(
  "${SERVER_PYTHON_BIN}" -u scripts/deploy_bridge_wa.py
  --model_path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --advertise_host "${ADVERTISE_HOST}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --default_steps "${STEPS}"
  --default_domain_id "${DOMAIN_ID}"
  --connection_info "${CONNECTION_INFO}"
)
if [[ -n "${PROCESSOR_PATH}" ]]; then
  SERVER_CMD+=(--processor_path "${PROCESSOR_PATH}")
fi

printf '[bridge_wa-vlabench-eval] server_cmd=' | tee -a "${SERVER_LOG_FILE}"
printf ' %q' "${SERVER_CMD[@]}" | tee -a "${SERVER_LOG_FILE}"
printf '\n' | tee -a "${SERVER_LOG_FILE}"
nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${SERVER_CMD[@]}" >> "${SERVER_LOG_FILE}" 2>&1 &
SERVER_PID="$!"

cleanup() {
  STATUS="$?"
  if [[ "${STOP_SERVER_AFTER_EVAL}" != "1" && "${STATUS}" == "0" ]]; then
    return
  fi
  if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[bridge_wa-vlabench-eval] stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        return
      fi
      sleep 1
    done
    echo "[bridge_wa-vlabench-eval] force stopping server pid=${SERVER_PID}"
    kill -9 "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[bridge_wa-vlabench-eval] server_pid=${SERVER_PID}"
echo "[bridge_wa-vlabench-eval] server_log=${SERVER_LOG_FILE}"
echo "[bridge_wa-vlabench-eval] connection_info=${CONNECTION_INFO}"

echo "[bridge_wa-vlabench-eval] waiting for connection info"
until [[ -f "${CONNECTION_INFO}" ]]; do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[bridge_wa-vlabench-eval] server exited before writing ${CONNECTION_INFO}; see ${SERVER_LOG_FILE}"
    exit 1
  fi
  sleep 1
done

read -r SERVER_HOST SERVER_PORT < <(
  CONNECTION_INFO="${CONNECTION_INFO}" "${SERVER_PYTHON_BIN}" - <<'PY'
import json
import os

with open(os.environ["CONNECTION_INFO"], "r", encoding="utf-8") as f:
    info = json.load(f)

print(info.get("host", "127.0.0.1"), int(info["port"]))
PY
)

HEALTH_URL="http://${SERVER_HOST}:${SERVER_PORT}/healthz"
echo "[bridge_wa-vlabench-eval] waiting for server health: ${HEALTH_URL}"
READY=0
for _ in $(seq 1 "${WAIT_TIMEOUT_SECONDS}"); do
  if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[bridge_wa-vlabench-eval] server exited while waiting; see ${SERVER_LOG_FILE}"
    exit 1
  fi
  sleep 1
done
if (( READY == 0 )); then
  echo "[bridge_wa-vlabench-eval] server health check timed out: ${HEALTH_URL}"
  exit 1
fi
echo "[bridge_wa-vlabench-eval] server is ready: ${SERVER_HOST}:${SERVER_PORT}"

case "${RUN_CLIENT}" in
  0|false|FALSE|no|NO|off|OFF|"")
    if [[ "${STOP_SERVER_AFTER_EVAL}" == "1" ]]; then
      echo "[bridge_wa-vlabench-eval] RUN_CLIENT=0, server health check complete; stopping server."
    else
      echo "[bridge_wa-vlabench-eval] RUN_CLIENT=0, server left running."
    fi
    exit 0
    ;;
  1|true|TRUE|yes|YES|on|ON)
    ;;
  *)
    echo "[bridge_wa-vlabench-eval] unsupported RUN_CLIENT=${RUN_CLIENT} (expected: 0|1|true|false)"
    exit 1
    ;;
esac

read -r -a EVAL_TRACK_ARRAY <<< "${EVAL_TRACKS}"
read -r -a METRIC_ARRAY <<< "${METRICS}"

CLIENT_CMD=(
  "${CLIENT_RUNNER[@]}" vlabench_client.py
  --eval-track "${EVAL_TRACK_ARRAY[@]}"
  --n-episode "${N_EPISODE}"
  --metrics "${METRIC_ARRAY[@]}"
  --host "${SERVER_HOST}"
  --port "${SERVER_PORT}"
  --eval_log_dir "${CLIENT_OUTPUT_DIR_ABS}"
)
if [[ -n "${TRACK_SOURCE_JSON}" ]]; then
  CLIENT_CMD+=(--track-source-json "${TRACK_SOURCE_JSON}")
fi
case "${RESUME}" in
  0|false|FALSE|no|NO|off|OFF|"")
    ;;
  1|true|TRUE|yes|YES|on|ON)
    CLIENT_CMD+=(--resume)
    ;;
  *)
    echo "[bridge_wa-vlabench-eval] unsupported RESUME=${RESUME} (expected: 0|1|true|false)"
    exit 1
    ;;
esac
case "${SAVE_VIDEO}" in
  0|false|FALSE|no|NO|off|OFF|"")
    CLIENT_CMD+=(--no-visulization)
    ;;
  1|true|TRUE|yes|YES|on|ON)
    ;;
  *)
    echo "[bridge_wa-vlabench-eval] unsupported SAVE_VIDEO=${SAVE_VIDEO} (expected: 0|1|true|false)"
    exit 1
    ;;
esac

echo "[bridge_wa-vlabench-eval] client_log=${CLIENT_LOG_FILE}"
printf '[bridge_wa-vlabench-eval] client_cmd=' | tee -a "${CLIENT_LOG_FILE}"
printf ' %q' "${CLIENT_CMD[@]}" | tee -a "${CLIENT_LOG_FILE}"
printf '\n' | tee -a "${CLIENT_LOG_FILE}"

(
  cd "${VLABENCH_EVAL_DIR}"
  "${CLIENT_CMD[@]}"
) 2>&1 | tee -a "${CLIENT_LOG_FILE}"

echo "[bridge_wa-vlabench-eval] done. results=${CLIENT_OUTPUT_DIR_ABS}"
