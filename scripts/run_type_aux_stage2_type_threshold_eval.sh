#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_predicted_long_learned_fusion}"
THRESHOLDS_FILE="${THRESHOLDS_FILE:-${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_learned_fusion_type_thresholds.json}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_learned_fusion_type_threshold_eval}"
DEVICE="${DEVICE:-}"
OFFLINE="${OFFLINE:-1}"

args=(
  "${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}"
  --relation_thresholds_file "${THRESHOLDS_FILE}"
  --config "${CONFIG}"
  --split eval
  --ner_span_threshold 0.5
  --ner_type_threshold 0.3
  --ner_overlap_strategy nms
  --relation_threshold 0.20
  --output_dir "${OUTPUT_DIR}"
)
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)
printf 'Running:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"

echo "Per-relation threshold eval metrics: ${OUTPUT_DIR}/eval_metrics.json"
