#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_ind_v8_fixed}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/ind_v8_fixed_pair_link_scores}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${LOCAL_CHECKPOINT}/document_relation_pair_link_v1}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_v8_pair_link_stage2_eval}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_ind_v8_pair_link.json}"

[[ -f "${LOCAL_CHECKPOINT}/config.json" ]] || { echo "Missing checkpoint: ${LOCAL_CHECKPOINT}" >&2; exit 2; }
[[ -f "${TRAINING_CONFIG}" ]] || { echo "Missing training config: ${TRAINING_CONFIG}" >&2; exit 2; }

NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.80}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.50}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"

CONFIG="${CONFIG}" LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT}" \
PREDICTIONS_DIR="${PREDICTIONS_DIR}" DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR}" \
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" TRAINING_CONFIG="${TRAINING_CONFIG}" \
NER_CALIBRATION="" NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD}" \
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD}" NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY}" \
MAX_DOCUMENT_ENTITIES=128 BATCH_SIZE=1 DEVICE="${DEVICE}" OFFLINE="${OFFLINE}" \
BFLOAT16=1 EVAL_RELATION_THRESHOLD=0.0 PAIR_LINK_HEAD=1 \
PAIR_LINK_LOSS_WEIGHT=0.35 PAIR_LINK_BIAS=-0.20 \
bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion.sh" \
  --epochs 15 --patience 5 --documents_per_batch 4 --window_batch_size 4 \
  --max_unlabeled_pairs 1024 --fusion_mode learned \
  --distance_conditioned_fusion --fusion_long_distance_boundary 128 \
  --fusion_distance_hidden_size 24 --fusion_long_distance_bias 0.35 \
  --long_distance_head --long_distance_boundary 128 --long_distance_head_weight 0.5 \
  --long_distance_ranking_loss_weight 0.15 --long_distance_ranking_margin 0.20 \
  --long_distance_ranking_max_negatives 96 --distance_experts \
  --distance_expert_boundaries 32 64 128 --distance_expert_temperature 0.75 \
  --distance_expert_weight 0.25 --distance_expert_loss_weight 0.10 \
  --distance_expert_active_buckets 2 --pair_type_compatibility \
  --pair_type_compatibility_loss_weight 0.08 \
  --pair_type_compatibility_bias_weight 0.20 --pair_type_focus_weight 2.0 \
  --use_direction_embedding --use_window_position_encoding \
  --positive_focal_loss_weight 0.20 --positive_focal_gamma 1.5 \
  --long_distance_sampling_weight 1.0 --focus_relation_names 包含 隶属 \
  --focus_relation_positive_weight 1.25

echo "Pair-link Stage-2 checkpoint: ${DOCUMENT_RE_DIR}"
echo "Pair-link eval output: ${EVAL_OUTPUT_DIR}"
