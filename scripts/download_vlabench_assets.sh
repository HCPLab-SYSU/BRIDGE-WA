#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VLABENCH_ROOT="${PROJECT_ROOT}/evaluation/vlabench/VLABench/VLABench"
: "${VLABENCH_CONDA_ENV:=vlabench}"
: "${CLIENT_PYTHON_BIN:=python}"

PYTHON_RUNNER=()
if [[ -n "${VLABENCH_CONDA_ENV}" ]]; then
    PYTHON_RUNNER=(conda run --no-capture-output -n "${VLABENCH_CONDA_ENV}" "${CLIENT_PYTHON_BIN}")
else
    PYTHON_RUNNER=("${CLIENT_PYTHON_BIN}")
fi

cd "${PROJECT_ROOT}/evaluation/vlabench/VLABench"

if [[ ! -f "${VLABENCH_ROOT}/assets/obj/meshes/table/table.xml" ]]; then
    "${PYTHON_RUNNER[@]}" scripts/download_assets.py --choice asset
else
    echo "[vlabench-assets] object assets already present"
fi

if [[ ! -f "${VLABENCH_ROOT}/assets/scenes/default/empty.xml" ]]; then
    "${PYTHON_RUNNER[@]}" scripts/download_assets.py --choice scene
else
    echo "[vlabench-assets] scene assets already present"
fi

"${PYTHON_RUNNER[@]}" "${PROJECT_ROOT}/scripts/download_vlabench_core_assets.py" \
    --asset-root "${VLABENCH_ROOT}/assets"
