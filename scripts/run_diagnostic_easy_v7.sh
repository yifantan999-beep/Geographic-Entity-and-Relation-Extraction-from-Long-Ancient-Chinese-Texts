#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
TRANSFORMER="${TRANSFORMER:-local-bert/google-bert/bert-base-chinese}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v7}"
OLD_CHECKPOINT="${OLD_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_span_v6_train_matched}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v7}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${STAGE1_CHECKPOINT}/document_relation_distance_experts_v3}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/diagnostic_easy_v7_scores}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_distance_experts_v3.json}"
STAGE1_TEST_DIR="${STAGE1_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v7_stage1_test}"
STAGE2_EVAL_DIR="${STAGE2_EVAL_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v7_stage2_eval}"
FINAL_TEST_DIR="${FINAL_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v7_distance_experts_test}"
NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.5}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.3}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-0.4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"

for required in "${OLD_CHECKPOINT}/config.json" "${TRAINING_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${STAGE1_CHECKPOINT}/config.json" ]]; then
  echo "Refusing to overwrite Stage 1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
  echo "Set STAGE1_CHECKPOINT to a new directory for another run." >&2
  exit 2
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"${PYTHON_BIN}" train_long_context.py \
  --transformer "${TRANSFORMER}" \
  --config "${CONFIG}" \
  --model_path "${STAGE1_CHECKPOINT}" \
  --load_ckpt "${OLD_CHECKPOINT}" \
  --epochs "${STAGE1_EPOCHS}" \
  --batch_size 1 \
  --gradient_accumulation 8 \
  --bfloat16 \
  --seed 42 \
  --rebuild_cache \
  --skip_final_test \
  --offline

"${PYTHON_BIN}" infer_long_context.py \
  --checkpoint "${STAGE1_CHECKPOINT}" \
  --config "${CONFIG}" \
  --split test \
  --device "${DEVICE}" \
  --batch_size 1 \
  --ner_span_threshold "${NER_SPAN_THRESHOLD}" \
  --ner_type_threshold "${NER_TYPE_THRESHOLD}" \
  --ner_overlap_strategy nms \
  --max_document_entities 128 \
  --output_dir "${STAGE1_TEST_DIR}" \
  --rebuild_cache \
  --offline

CONFIG="${CONFIG}" \
LOCAL_CHECKPOINT="${STAGE1_CHECKPOINT}" \
PREDICTIONS_DIR="${PREDICTIONS_DIR}" \
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR}" \
EVAL_OUTPUT_DIR="${STAGE2_EVAL_DIR}" \
TRAINING_CONFIG="${TRAINING_CONFIG}" \
NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD}" \
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD}" \
NER_OVERLAP_STRATEGY=nms \
MAX_DOCUMENT_ENTITIES=128 \
BATCH_SIZE=1 \
DEVICE="${DEVICE}" \
OFFLINE=1 \
BFLOAT16=1 \
bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion.sh"

if [[ -e "${FINAL_TEST_DIR}/test_metrics.json" ]]; then
  echo "Refusing to overwrite diagnostic test output: ${FINAL_TEST_DIR}" >&2
  exit 2
fi

"${PYTHON_BIN}" infer_long_context.py \
  --checkpoint "${STAGE1_CHECKPOINT}" \
  --document_re_checkpoint "${DOCUMENT_RE_DIR}" \
  --config "${CONFIG}" \
  --split test \
  --device "${DEVICE}" \
  --batch_size 1 \
  --ner_span_threshold "${NER_SPAN_THRESHOLD}" \
  --ner_type_threshold "${NER_TYPE_THRESHOLD}" \
  --ner_overlap_strategy nms \
  --relation_threshold "${RELATION_THRESHOLD}" \
  --review_margin 0.1 \
  --max_document_entities 128 \
  --output_dir "${FINAL_TEST_DIR}" \
  --offline

echo "Stage 1 diagnostic metrics: ${STAGE1_TEST_DIR}/test_metrics.json"
echo "Stage 2 training summary: ${DOCUMENT_RE_DIR}/training_summary.json"
echo "Final diagnostic metrics: ${FINAL_TEST_DIR}/test_metrics.json"
