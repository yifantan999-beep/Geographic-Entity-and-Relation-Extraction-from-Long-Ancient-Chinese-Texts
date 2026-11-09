#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_ind_v8_fixed}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_pair_link_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_v8_pair_link_final_test}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-0.35}"

[[ -f "${DOCUMENT_RE_CHECKPOINT}/document_relation.safetensors" ]] || { echo "Missing document RE checkpoint: ${DOCUMENT_RE_CHECKPOINT}" >&2; exit 2; }

args=(
  python3 infer_long_context.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}"
  --config "${CONFIG}"
  --split test
  --device "${DEVICE}"
  --batch_size 1
  --ner_span_threshold 0.80
  --ner_type_threshold 0.50
  --ner_overlap_strategy nms
  --relation_threshold "${RELATION_THRESHOLD}"
  --max_document_entities 128
  --output_dir "${OUTPUT_DIR}"
)
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)
printf 'Running:'; printf ' %q' "${args[@]}"; printf '\n'
"${args[@]}"
echo "Pair-link final test metrics: ${OUTPUT_DIR}/test_metrics.json"
