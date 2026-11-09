#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_stage1/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation}"
USE_DOCUMENT_RE="${USE_DOCUMENT_RE:-0}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/run_01_${SPLIT}}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DEVICE="${DEVICE:-}"
LONG_THRESHOLD="${LONG_THRESHOLD:-128}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-}"
LOCAL_GLOBAL_ALPHA="${LOCAL_GLOBAL_ALPHA:-0.65}"
REVIEW_MARGIN="${REVIEW_MARGIN:-0.10}"
MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-64}"
MAX_DOCUMENTS="${MAX_DOCUMENTS:-}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"
OFFLINE="${OFFLINE:-0}"

if [[ ! -d "${LOCAL_CHECKPOINT}" ]]; then
  echo "Local checkpoint directory does not exist: ${LOCAL_CHECKPOINT}" >&2
  exit 2
fi
if [[ "${USE_DOCUMENT_RE}" == "1" && ! -d "${DOCUMENT_RE_CHECKPOINT}" ]]; then
  echo "Document RE checkpoint directory does not exist: ${DOCUMENT_RE_CHECKPOINT}" >&2
  echo "Run scripts/train_document_relation.sh first, set DOCUMENT_RE_CHECKPOINT, or set USE_DOCUMENT_RE=0." >&2
  exit 2
fi

args=(
  "${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --config "${CONFIG}"
  --split "${SPLIT}"
  --batch_size "${BATCH_SIZE}"
  --long_threshold "${LONG_THRESHOLD}"
  --local_global_alpha "${LOCAL_GLOBAL_ALPHA}"
  --review_margin "${REVIEW_MARGIN}"
  --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
  --output_dir "${OUTPUT_DIR}"
)

[[ "${USE_DOCUMENT_RE}" == "1" ]] && args+=(--document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}")
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ -n "${RELATION_THRESHOLD}" ]] && args+=(--relation_threshold "${RELATION_THRESHOLD}")
[[ -n "${MAX_DOCUMENTS}" ]] && args+=(--max_documents "${MAX_DOCUMENTS}")
[[ "${REBUILD_CACHE}" == "1" ]] && args+=(--rebuild_cache)
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)
args+=("$@")

printf 'Running:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"
