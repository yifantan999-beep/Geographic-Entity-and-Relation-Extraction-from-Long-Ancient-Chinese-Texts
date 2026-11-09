#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_ind_v8_fixed}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01}"
NER_CALIBRATION="${NER_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/ind_v8_fixed_ner_calibration.json}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/ind_v8_fixed_scores}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${CHECKPOINT}/document_relation_distance_experts_v8}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_ind_v8_fixed.json}"
EVAL_DIR="${EVAL_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_v8_fixed_stage2_eval}"
GRID_DIR="${GRID_DIR:-${PROJECT_ROOT}/artifacts/calibration/ind_v8_fixed_re_grid}"
RE_CALIBRATION="${RE_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/ind_v8_fixed_re_calibration.json}"
for path in "${CHECKPOINT}/config.json" "${NER_CALIBRATION}" "${TRAINING_CONFIG}"; do
  [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 2; }
done
mapfile -t ner < <("${PYTHON_BIN}" -c 'import json,sys; b=json.load(open(sys.argv[1],encoding="utf-8"))["best"]; print(b["span_threshold"]); print(b["type_threshold"]); print(b["overlap_strategy"])' "${NER_CALIBRATION}")
if [[ ! -f "${DOCUMENT_RE_DIR}/document_relation.safetensors" ]]; then
  CONFIG="${CONFIG}" LOCAL_CHECKPOINT="${CHECKPOINT}" PREDICTIONS_DIR="${PREDICTIONS_DIR}" DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR}" EVAL_OUTPUT_DIR="${EVAL_DIR}" TRAINING_CONFIG="${TRAINING_CONFIG}" NER_CALIBRATION="" NER_SPAN_THRESHOLD="${ner[0]}" NER_TYPE_THRESHOLD="${ner[1]}" NER_OVERLAP_STRATEGY="${ner[2]}" MAX_DOCUMENT_ENTITIES=128 BATCH_SIZE=1 DEVICE="${DEVICE}" OFFLINE=1 BFLOAT16=1 EVAL_RELATION_THRESHOLD=0.0 PAIR_LINK_HEAD="${PAIR_LINK_HEAD:-}" PAIR_LINK_LOSS_WEIGHT="${PAIR_LINK_LOSS_WEIGHT:-}" PAIR_LINK_BIAS="${PAIR_LINK_BIAS:-}" bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion.sh" --epochs 15 --patience 5 --documents_per_batch 4 --window_batch_size 4 --max_unlabeled_pairs 1024 --fusion_mode learned --distance_conditioned_fusion --fusion_long_distance_boundary 128 --fusion_distance_hidden_size 24 --fusion_long_distance_bias 0.35 --long_distance_head --long_distance_boundary 128 --long_distance_head_weight 0.5 --long_distance_ranking_loss_weight 0.15 --long_distance_ranking_margin 0.20 --long_distance_ranking_max_negatives 96 --distance_experts --distance_expert_boundaries 32 64 128 --distance_expert_temperature 0.75 --distance_expert_weight 0.25 --distance_expert_loss_weight 0.10 --distance_expert_active_buckets 2 --pair_type_compatibility --pair_type_compatibility_loss_weight 0.08 --pair_type_compatibility_bias_weight 0.20 --pair_type_focus_weight 2.0 --use_direction_embedding --use_window_position_encoding --positive_focal_loss_weight 0.35 --positive_focal_gamma 1.5 --long_distance_sampling_weight 3.0 --focus_relation_names 包含 隶属 --focus_relation_positive_weight 1.25
else
  echo "Stage-2 checkpoint already exists; reusing ${DOCUMENT_RE_DIR}"
fi
if [[ ! -f "${RE_CALIBRATION}" ]]; then
  RELATION_THRESHOLDS="0.05 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.30 0.35 0.40 0.50 0.60 0.70 0.80 0.90 0.95" CONFIG="${CONFIG}" LOCAL_CHECKPOINT="${CHECKPOINT}" DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_DIR}" GRID_DIR="${GRID_DIR}" OUTPUT="${RE_CALIBRATION}" DEVICE="${DEVICE}" OFFLINE=1 NER_CALIBRATION="" NER_SPAN_THRESHOLD="${ner[0]}" NER_TYPE_THRESHOLD="${ner[1]}" NER_OVERLAP_STRATEGY="${ner[2]}" MAX_DOCUMENT_ENTITIES=128 BATCH_SIZE=1 bash "${SCRIPT_DIR}/calibrate_type_aux_stage2_learned_fusion.sh"
else
  echo "RE calibration already exists; reusing ${RE_CALIBRATION}"
fi
"${PYTHON_BIN}" -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8")); print(json.dumps({"source_split":m.get("source_split"),"best":m.get("best"),"best_at_search_boundary":m.get("best_at_search_boundary")},ensure_ascii=False,indent=2))' "${RE_CALIBRATION}"
echo "Stage-2 checkpoint: ${DOCUMENT_RE_DIR}"
echo "Stage-2 eval metrics: ${EVAL_DIR}/eval_metrics.json"
echo "RE calibration: ${RE_CALIBRATION}"
