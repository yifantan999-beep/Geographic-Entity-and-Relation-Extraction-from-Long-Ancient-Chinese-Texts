#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the same frozen TypeAux + NMS + learned-fusion model on each
# relation-distance subset.  Calibration stays fixed from the original eval
# split so that only the dataset selection changes between runs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_predicted_long_learned_fusion}"
NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.5}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.3}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-0.2}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
REVIEW_MARGIN="${REVIEW_MARGIN:-0.0}"
SPLIT="${SPLIT:-test}"
REBUILD_CACHE="${REBUILD_CACHE:-1}"
DATASET_CONFIG_PREFIX="${DATASET_CONFIG_PREFIX:-wenyanwen/wenyanwen_long_context_512_type_aux_distance_gt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-long_context_distance_gt}"
if (( $# > 0 )); then
  THRESHOLDS=("$@")
else
  THRESHOLDS=(24 28 32 36)
fi

if [[ "${SPLIT}" != "eval" && "${SPLIT}" != "test" ]]; then
  echo "SPLIT must be eval or test (got ${SPLIT})" >&2
  exit 2
fi

for path in \
  "${LOCAL_CHECKPOINT}/config.json" \
  "${DOCUMENT_RE_CHECKPOINT}/document_relation_config.json"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 2
  fi
done

for threshold in "${THRESHOLDS[@]}"; do
  config="${DATASET_CONFIG_PREFIX}${threshold}"
  output_dir="${PROJECT_ROOT}/artifacts/inference/${OUTPUT_PREFIX}${threshold}_best_${SPLIT}"
  metrics_name="${SPLIT}_metrics.json"
  if [[ -f "${output_dir}/${metrics_name}" ]]; then
    echo "Refusing to overwrite existing result: ${output_dir}/${metrics_name}" >&2
    exit 2
  fi

  args=(
    "${PYTHON_BIN}" infer_long_context.py
    --checkpoint "${LOCAL_CHECKPOINT}"
    --document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}"
    --config "${config}"
    --split "${SPLIT}"
    --batch_size "${BATCH_SIZE}"
    --device "${DEVICE}"
    --ner_span_threshold "${NER_SPAN_THRESHOLD}"
    --ner_type_threshold "${NER_TYPE_THRESHOLD}"
    --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
    --relation_threshold "${RELATION_THRESHOLD}"
    --review_margin "${REVIEW_MARGIN}"
    --output_dir "${output_dir}"
  )
  if [[ "${REBUILD_CACHE}" == "1" ]]; then
    args+=(--rebuild_cache)
  fi
  if [[ "${OFFLINE}" == "1" ]]; then
    args+=(--offline)
  fi
  printf 'Running distance subset >%s:' "${threshold}"
  printf ' %q' "${args[@]}"
  printf '\n'
  "${args[@]}"
done

echo "Distance-threshold sweep completed."
