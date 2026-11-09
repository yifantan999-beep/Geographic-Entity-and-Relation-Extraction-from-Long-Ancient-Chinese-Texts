#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_recall}"
OLD_STAGE1="${OLD_STAGE1:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_target}"
OLD_DOCUMENT_RE="${OLD_DOCUMENT_RE:-${OLD_STAGE1}/document_relation_distance_experts_v8}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_recall}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${STAGE1_CHECKPOINT}/document_relation_distance_experts_v8_recall}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_distance_experts_v8_recall.json}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/diagnostic_easy_v8_recall_scores}"
STAGE1_TEST_DIR="${STAGE1_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_recall_stage1_test}"
STAGE2_EVAL_DIR="${STAGE2_EVAL_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_recall_stage2_eval}"
FINAL_TEST_DIR="${FINAL_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_recall_final_test}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE1_PATIENCE="${STAGE1_PATIENCE:-6}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-15}"
STAGE2_PATIENCE="${STAGE2_PATIENCE:-5}"
RECALL_HEAD_WEIGHT="${RECALL_HEAD_WEIGHT:-0.10}"
RECALL_MARGIN_LOSS_WEIGHT="${RECALL_MARGIN_LOSS_WEIGHT:-0.15}"
RECALL_SHORT_DISTANCE_WEIGHT="${RECALL_SHORT_DISTANCE_WEIGHT:-1.25}"
EVAL_REVIEW_MARGIN="${EVAL_REVIEW_MARGIN:-0.0}"

if [[ "${CONFIG}" != *"diagnostic_easy_v8_recall" ]]; then
  echo "This script requires the v8_recall dataset config; got: ${CONFIG}" >&2
  echo "Unset CONFIG or set it to wenyanwen/..._diagnostic_easy_v8_recall." >&2
  exit 2
fi

for required in "${OLD_STAGE1}/config.json" "${OLD_DOCUMENT_RE}/document_relation.safetensors" "${TRAINING_CONFIG}"; do
  [[ -f "${required}" ]] || { echo "Required file does not exist: ${required}" >&2; exit 2; }
done

if [[ ! -f "${STAGE1_CHECKPOINT}/config.json" ]]; then
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" "${PYTHON_BIN}" train_long_context.py \
    --transformer local-bert/google-bert/bert-base-chinese \
    --config "${CONFIG}" --model_path "${STAGE1_CHECKPOINT}" \
    --load_ckpt "${OLD_STAGE1}" --epochs "${STAGE1_EPOCHS}" --patience "${STAGE1_PATIENCE}" --batch_size 1 \
    --gradient_accumulation 8 --bfloat16 --seed 42 --rebuild_cache \
    --skip_final_test --offline
fi

if [[ ! -f "${STAGE1_TEST_DIR}/test_metrics.json" ]]; then
  "${PYTHON_BIN}" infer_long_context.py \
    --checkpoint "${STAGE1_CHECKPOINT}" --config "${CONFIG}" --split test \
    --device "${DEVICE}" --batch_size 1 --ner_span_threshold 0.80 \
    --ner_type_threshold 0.50 --ner_overlap_strategy nms \
    --max_document_entities 128 --output_dir "${STAGE1_TEST_DIR}" --offline
fi

if [[ ! -f "${DOCUMENT_RE_DIR}/document_relation.safetensors" ]]; then
  CONFIG="${CONFIG}" LOCAL_CHECKPOINT="${STAGE1_CHECKPOINT}" \
  BASE_DOCUMENT_RE="${OLD_DOCUMENT_RE}" PREDICTIONS_DIR="${PREDICTIONS_DIR}" \
  DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR}" EVAL_OUTPUT_DIR="${STAGE2_EVAL_DIR}" \
  TRAINING_CONFIG="${TRAINING_CONFIG}" NER_CALIBRATION="" \
  NER_SPAN_THRESHOLD=0.80 NER_TYPE_THRESHOLD=0.50 NER_OVERLAP_STRATEGY=nms \
  MAX_DOCUMENT_ENTITIES=128 BATCH_SIZE=1 DEVICE="${DEVICE}" OFFLINE=1 BFLOAT16=1 \
  FIXED_EVAL_THRESHOLD=0.24 RECALL_HEAD=1 RECALL_HEAD_SIZE=128 \
  RECALL_HEAD_WEIGHT="${RECALL_HEAD_WEIGHT}" RECALL_MARGIN_LOSS_WEIGHT="${RECALL_MARGIN_LOSS_WEIGHT}" \
  RECALL_THRESHOLD=0.24 RECALL_MARGIN=0.35 \
  RECALL_SHORT_DISTANCE_BOUNDARY=64 RECALL_SHORT_DISTANCE_WEIGHT="${RECALL_SHORT_DISTANCE_WEIGHT}" \
  EVAL_RELATION_THRESHOLD=0.24 EVAL_REVIEW_MARGIN="${EVAL_REVIEW_MARGIN}" \
  bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion.sh" \
    --load_document_checkpoint "${OLD_DOCUMENT_RE}" \
    --epochs "${STAGE2_EPOCHS}" --patience "${STAGE2_PATIENCE}"
fi

if [[ ! -f "${FINAL_TEST_DIR}/test_metrics.json" ]]; then
  "${PYTHON_BIN}" infer_long_context.py \
    --checkpoint "${STAGE1_CHECKPOINT}" \
    --document_re_checkpoint "${DOCUMENT_RE_DIR}" \
    --config "${CONFIG}" --split test --device "${DEVICE}" --batch_size 1 \
    --ner_span_threshold 0.80 --ner_type_threshold 0.50 \
    --ner_overlap_strategy nms --relation_threshold 0.24 --review_margin 0.0 \
    --max_document_entities 128 --output_dir "${FINAL_TEST_DIR}" --offline
fi

echo "Stage 1 metrics: ${STAGE1_TEST_DIR}/test_metrics.json"
echo "Stage 2 summary: ${DOCUMENT_RE_DIR}/training_summary.json"
echo "Final metrics: ${FINAL_TEST_DIR}/test_metrics.json"
