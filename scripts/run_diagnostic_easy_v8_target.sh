#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
TRANSFORMER="${TRANSFORMER:-local-bert/google-bert/bert-base-chinese}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8}"
OLD_CHECKPOINT="${OLD_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_span_v6_train_matched}"
BASE_DOCUMENT_RE="${BASE_DOCUMENT_RE:-${OLD_CHECKPOINT}/document_relation_long_distance_v2}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_target}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${STAGE1_CHECKPOINT}/document_relation_distance_experts_v8}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_distance_experts_v8.json}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/diagnostic_easy_v8_target_scores}"
NER_EVIDENCE="${NER_EVIDENCE:-${PROJECT_ROOT}/artifacts/calibration/diagnostic_easy_v8_target_ner_evidence.json}"
NER_CALIBRATION="${NER_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/diagnostic_easy_v8_target_ner_calibration.json}"
STAGE1_TEST_DIR="${STAGE1_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_target_stage1_test}"
STAGE2_EVAL_DIR="${STAGE2_EVAL_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_target_stage2_eval}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_target_re_candidates}"
RELATION_THRESHOLDS_FILE="${RELATION_THRESHOLDS_FILE:-${PROJECT_ROOT}/artifacts/calibration/diagnostic_easy_v8_target_relation_thresholds.json}"
FINAL_TEST_DIR="${FINAL_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_target_final_test}"

"${PYTHON_BIN}" scripts/build_diagnostic_easy_target_dataset.py

for required in \
  "${OLD_CHECKPOINT}/config.json" \
  "${BASE_DOCUMENT_RE}/document_relation.safetensors" \
  "${TRAINING_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required file does not exist: ${required}" >&2
    exit 2
  fi
done
for output in \
  "${STAGE1_CHECKPOINT}/config.json" \
  "${DOCUMENT_RE_DIR}/document_relation.safetensors" \
  "${FINAL_TEST_DIR}/test_metrics.json"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing result: ${output}" >&2
    echo "Use new output paths for another run." >&2
    exit 2
  fi
done

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"${PYTHON_BIN}" train_long_context.py \
  --transformer "${TRANSFORMER}" \
  --config "${CONFIG}" \
  --model_path "${STAGE1_CHECKPOINT}" \
  --load_ckpt "${OLD_CHECKPOINT}" \
  --epochs 20 \
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
  --split eval \
  --device "${DEVICE}" \
  --batch_size 1 \
  --max_document_entities 128 \
  --dump_ner_evidence "${NER_EVIDENCE}" \
  --rebuild_cache \
  --offline

"${PYTHON_BIN}" calibrate_long_context_ner.py \
  --evidence "${NER_EVIDENCE}" \
  --output "${NER_CALIBRATION}" \
  --span_thresholds 0.20 0.35 0.50 0.65 0.80 \
  --type_thresholds 0.10 0.20 0.30 0.40 0.50 \
  --overlap_strategies nms \
  --quiet

mapfile -t ner_parameters < <(
  "${PYTHON_BIN}" - "${NER_CALIBRATION}" <<'PY'
import json
import sys
from pathlib import Path

best = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["best"]
print(best["span_threshold"])
print(best["type_threshold"])
print(best["overlap_strategy"])
PY
)
NER_SPAN_THRESHOLD="${ner_parameters[0]}"
NER_TYPE_THRESHOLD="${ner_parameters[1]}"
NER_OVERLAP_STRATEGY="${ner_parameters[2]}"

"${PYTHON_BIN}" infer_long_context.py \
  --checkpoint "${STAGE1_CHECKPOINT}" \
  --config "${CONFIG}" \
  --split test \
  --device "${DEVICE}" \
  --batch_size 1 \
  --ner_span_threshold "${NER_SPAN_THRESHOLD}" \
  --ner_type_threshold "${NER_TYPE_THRESHOLD}" \
  --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}" \
  --max_document_entities 128 \
  --output_dir "${STAGE1_TEST_DIR}" \
  --offline

CONFIG="${CONFIG}" \
LOCAL_CHECKPOINT="${STAGE1_CHECKPOINT}" \
PREDICTIONS_DIR="${PREDICTIONS_DIR}" \
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR}" \
EVAL_OUTPUT_DIR="${STAGE2_EVAL_DIR}" \
TRAINING_CONFIG="${TRAINING_CONFIG}" \
NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD}" \
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD}" \
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY}" \
MAX_DOCUMENT_ENTITIES=128 \
BATCH_SIZE=1 \
DEVICE="${DEVICE}" \
OFFLINE=1 \
BFLOAT16=1 \
bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion.sh" \
  --load_document_checkpoint "${BASE_DOCUMENT_RE}"

GLOBAL_RE_THRESHOLD="$(${PYTHON_BIN} - "${DOCUMENT_RE_DIR}/document_relation_config.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["threshold"])
PY
)"

"${PYTHON_BIN}" infer_long_context.py \
  --checkpoint "${STAGE1_CHECKPOINT}" \
  --document_re_checkpoint "${DOCUMENT_RE_DIR}" \
  --config "${CONFIG}" \
  --split eval \
  --device "${DEVICE}" \
  --batch_size 1 \
  --ner_span_threshold "${NER_SPAN_THRESHOLD}" \
  --ner_type_threshold "${NER_TYPE_THRESHOLD}" \
  --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}" \
  --relation_threshold 0.05 \
  --review_margin 0.0 \
  --max_document_entities 128 \
  --output_dir "${CANDIDATE_DIR}" \
  --offline

"${PYTHON_BIN}" calibrate_relation_type_thresholds.py \
  --predictions "${CANDIDATE_DIR}/eval_predictions.jsonl" \
  --output "${RELATION_THRESHOLDS_FILE}" \
  --source-split eval \
  --global-threshold "${GLOBAL_RE_THRESHOLD}" \
  --min-support 12 \
  --threshold-grid 0.05 0.08 0.10 0.12 0.16 0.20 0.24 0.30 0.36 0.40 0.48 0.60

"${PYTHON_BIN}" infer_long_context.py \
  --checkpoint "${STAGE1_CHECKPOINT}" \
  --document_re_checkpoint "${DOCUMENT_RE_DIR}" \
  --config "${CONFIG}" \
  --split test \
  --device "${DEVICE}" \
  --batch_size 1 \
  --ner_span_threshold "${NER_SPAN_THRESHOLD}" \
  --ner_type_threshold "${NER_TYPE_THRESHOLD}" \
  --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}" \
  --relation_threshold "${GLOBAL_RE_THRESHOLD}" \
  --relation_thresholds_file "${RELATION_THRESHOLDS_FILE}" \
  --review_margin 0.0 \
  --max_document_entities 128 \
  --output_dir "${FINAL_TEST_DIR}" \
  --offline

"${PYTHON_BIN}" - \
  "${PROJECT_ROOT}/artifacts/inference/cross_window_diagnostic_easy_v6_test/test_metrics.json" \
  "${FINAL_TEST_DIR}/test_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

baseline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
current = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
keys = (
    "ner_strict",
    "ner_boundary",
    "re_strict_end_to_end",
    "re_boundary_end_to_end",
    "re_on_matched_entities",
)
for key in keys:
    before = baseline["overall"][key]["f1"]
    after = current["overall"][key]["f1"]
    print(f"{key}: {before:.6f} -> {after:.6f} ({after - before:+.6f})")
PY

echo "NER calibration: ${NER_CALIBRATION}"
echo "Stage 1 target metrics: ${STAGE1_TEST_DIR}/test_metrics.json"
echo "Stage 2 summary: ${DOCUMENT_RE_DIR}/training_summary.json"
echo "Relation thresholds: ${RELATION_THRESHOLDS_FILE}"
echo "Final target metrics: ${FINAL_TEST_DIR}/test_metrics.json"
