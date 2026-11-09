#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SPLIT="${SPLIT:-test}"
INFERENCE_DIR="${INFERENCE_DIR:-${PROJECT_ROOT}/artifacts/inference/run_01_${SPLIT}}"
PREDICTIONS="${PREDICTIONS:-${INFERENCE_DIR}/${SPLIT}_predictions.jsonl}"
OUTPUT="${OUTPUT:-${INFERENCE_DIR}/${SPLIT}_metrics_recomputed.json}"
LONG_THRESHOLD="${LONG_THRESHOLD:-128}"
LENGTH_BUCKETS="${LENGTH_BUCKETS:-128 256 512}"
DISTANCE_BUCKETS="${DISTANCE_BUCKETS:-32 64 128 256}"

if [[ ! -f "${PREDICTIONS}" ]]; then
  echo "Prediction file does not exist: ${PREDICTIONS}" >&2
  echo "Run scripts/infer_long_context.sh first or set PREDICTIONS." >&2
  exit 2
fi

read -r -a length_bucket_values <<< "${LENGTH_BUCKETS}"
read -r -a distance_bucket_values <<< "${DISTANCE_BUCKETS}"

args=(
  "${PYTHON_BIN}" evaluate_long_context.py
  --predictions "${PREDICTIONS}"
  --output "${OUTPUT}"
  --long_threshold "${LONG_THRESHOLD}"
  --length_buckets "${length_bucket_values[@]}"
  --distance_buckets "${distance_bucket_values[@]}"
)
args+=("$@")

printf 'Running:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"
