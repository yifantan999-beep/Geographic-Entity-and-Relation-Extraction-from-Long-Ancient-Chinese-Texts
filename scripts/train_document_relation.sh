#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_stage1/bert-base-chinese/run_01}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${LOCAL_CHECKPOINT}/document_relation}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation.json}"
DEVICE="${DEVICE:-}"
BFLOAT16="${BFLOAT16:-0}"
OFFLINE="${OFFLINE:-0}"

if [[ ! -d "${LOCAL_CHECKPOINT}" ]]; then
  echo "Local checkpoint directory does not exist: ${LOCAL_CHECKPOINT}" >&2
  echo "Run scripts/train_long_context.sh first or set LOCAL_CHECKPOINT." >&2
  exit 2
fi
if [[ ! -f "${TRAINING_CONFIG}" ]]; then
  echo "Training config does not exist: ${TRAINING_CONFIG}" >&2
  exit 2
fi

args=(
  "${PYTHON_BIN}" train_document_relation.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --config "${CONFIG}"
  --training_config "${TRAINING_CONFIG}"
  --output_dir "${DOCUMENT_RE_DIR}"
)

[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ "${BFLOAT16}" == "1" ]] && args+=(--bfloat16)
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)
args+=("$@")

printf 'Running:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"
