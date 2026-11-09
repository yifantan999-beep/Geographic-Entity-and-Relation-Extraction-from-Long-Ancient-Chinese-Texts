#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_target}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${CHECKPOINT}/document_relation_distance_experts_v8}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_fixed_test}"

if [[ -e "${OUTPUT_DIR}/test_metrics.json" ]]; then
  echo "Refusing to overwrite existing result: ${OUTPUT_DIR}/test_metrics.json" >&2
  echo "Set OUTPUT_DIR to a new directory for another run." >&2
  exit 2
fi

for required in \
  "${CHECKPOINT}/config.json" \
  "${DOCUMENT_RE_CHECKPOINT}/document_relation.safetensors"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done

args=(
  "${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}"
  --config "${CONFIG}"
  --split test
  --batch_size 1
  --ner_span_threshold 0.80
  --ner_type_threshold 0.50
  --ner_overlap_strategy nms
  --relation_threshold 0.24
  --review_margin 0.0
  --max_document_entities 128
  --output_dir "${OUTPUT_DIR}"
)
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ "${OFFLINE:-1}" == "1" ]] && args+=(--offline)

printf 'Running:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"

echo "Primary experiment metrics: ${OUTPUT_DIR}/test_metrics.json"
echo "Primary experiment predictions: ${OUTPUT_DIR}/test_predictions.jsonl"
