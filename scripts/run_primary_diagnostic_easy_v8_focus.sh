#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
CONFIG="wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_fixed_boundary_focus"
OLD_STAGE1="${OLD_STAGE1:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_target}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_fixed_boundary_focus}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${STAGE1_CHECKPOINT}/document_relation_fixed_balanced}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/diagnostic_easy_v8_fixed_boundary_focus_scores}"
STAGE1_EVAL_DIR="${STAGE1_EVAL_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_fixed_boundary_focus_stage1_eval}"
STAGE1_TEST_DIR="${STAGE1_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_fixed_boundary_focus_stage1_test}"
STAGE2_EVAL_DIR="${STAGE2_EVAL_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_fixed_boundary_focus_stage2_eval}"
FINAL_TEST_DIR="${FINAL_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_fixed_boundary_focus_final_test}"
TRAINING_CONFIG="${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_distance_experts_v8_fixed_balanced.json"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE1_PATIENCE="${STAGE1_PATIENCE:-6}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-15}"
STAGE2_PATIENCE="${STAGE2_PATIENCE:-5}"

for required in "${OLD_STAGE1}/config.json" "${TRAINING_CONFIG}"; do
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

for split_dir in "${STAGE1_EVAL_DIR}" "${STAGE1_TEST_DIR}"; do
  [[ ! -e "${split_dir}/eval_metrics.json" && ! -e "${split_dir}/test_metrics.json" ]] || continue
  split="eval"; [[ "${split_dir}" == *stage1_test ]] && split="test"
  "${PYTHON_BIN}" infer_long_context.py \
    --checkpoint "${STAGE1_CHECKPOINT}" --config "${CONFIG}" --split "${split}" \
    --device "${DEVICE}" --batch_size 1 --ner_span_threshold 0.80 \
    --ner_type_threshold 0.50 --ner_overlap_strategy nms --max_document_entities 128 \
    --output_dir "${split_dir}" --offline
done

if [[ ! -f "${DOCUMENT_RE_DIR}/document_relation.safetensors" ]]; then
  CONFIG="${CONFIG}" LOCAL_CHECKPOINT="${STAGE1_CHECKPOINT}" \
  BASE_DOCUMENT_RE="${OLD_STAGE1}/document_relation_distance_experts_v8" \
  PREDICTIONS_DIR="${PREDICTIONS_DIR}" DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR}" \
  EVAL_OUTPUT_DIR="${STAGE2_EVAL_DIR}" TRAINING_CONFIG="${TRAINING_CONFIG}" \
  NER_CALIBRATION="" NER_SPAN_THRESHOLD=0.80 NER_TYPE_THRESHOLD=0.50 \
  NER_OVERLAP_STRATEGY=nms MAX_DOCUMENT_ENTITIES=128 BATCH_SIZE=1 DEVICE="${DEVICE}" \
  OFFLINE=1 BFLOAT16=1 FIXED_EVAL_THRESHOLD=0.24 EVAL_RELATION_THRESHOLD=0.24 \
  EVAL_REVIEW_MARGIN=0.0 bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion.sh" \
    --load_document_checkpoint "${OLD_STAGE1}/document_relation_distance_experts_v8" \
    --epochs "${STAGE2_EPOCHS}" --patience "${STAGE2_PATIENCE}"
fi

if [[ ! -f "${FINAL_TEST_DIR}/test_metrics.json" ]]; then
  "${PYTHON_BIN}" infer_long_context.py \
    --checkpoint "${STAGE1_CHECKPOINT}" --document_re_checkpoint "${DOCUMENT_RE_DIR}" \
    --config "${CONFIG}" --split test --device "${DEVICE}" --batch_size 1 \
    --ner_span_threshold 0.80 --ner_type_threshold 0.50 --ner_overlap_strategy nms \
    --relation_threshold 0.24 --review_margin 0.0 --max_document_entities 128 \
    --output_dir "${FINAL_TEST_DIR}" --offline
fi

echo "Stage-1 eval: ${STAGE1_EVAL_DIR}/eval_metrics.json"
echo "Stage-1 test: ${STAGE1_TEST_DIR}/test_metrics.json"
echo "Stage-2 summary: ${DOCUMENT_RE_DIR}/training_summary.json"
echo "Final test: ${FINAL_TEST_DIR}/test_metrics.json"
