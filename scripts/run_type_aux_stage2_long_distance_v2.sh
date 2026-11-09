#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# v5 Stage-1 uses the low-rank span-biaffine residual.  v2 Stage-2 adds
# directional pair encoding, window-position evidence, positive focal risk,
# and long-distance-aware candidate sampling.
export CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_v5}"
export LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_span_v5}"
export PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/ind_distance_gt28_rebalanced_cross_window_ner_span_v5_scores}"
export DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${LOCAL_CHECKPOINT}/document_relation_long_distance_v2}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_distance_gt28_rebalanced_cross_window_stage2_long_distance_v2_eval}"
export TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_long_distance_v2.json}"
export NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.40}"
export NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.80}"
export NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
export MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-128}"
export DISTANCE_CONDITIONED_FUSION="${DISTANCE_CONDITIONED_FUSION:-1}"
export FUSION_LONG_DISTANCE_BOUNDARY="${FUSION_LONG_DISTANCE_BOUNDARY:-128}"
export FUSION_LONG_DISTANCE_BIAS="${FUSION_LONG_DISTANCE_BIAS:-0.35}"
export LONG_DISTANCE_HEAD="${LONG_DISTANCE_HEAD:-1}"
export LONG_DISTANCE_BOUNDARY="${LONG_DISTANCE_BOUNDARY:-128}"
export LONG_DISTANCE_HEAD_WEIGHT="${LONG_DISTANCE_HEAD_WEIGHT:-0.5}"
export LONG_DISTANCE_RANKING_LOSS_WEIGHT="${LONG_DISTANCE_RANKING_LOSS_WEIGHT:-0.20}"
export LONG_DISTANCE_RANKING_MARGIN="${LONG_DISTANCE_RANKING_MARGIN:-0.20}"
export LONG_DISTANCE_RANKING_MAX_NEGATIVES="${LONG_DISTANCE_RANKING_MAX_NEGATIVES:-96}"
export PAIR_TYPE_COMPATIBILITY="${PAIR_TYPE_COMPATIBILITY:-1}"
export PAIR_TYPE_COMPATIBILITY_LOSS_WEIGHT="${PAIR_TYPE_COMPATIBILITY_LOSS_WEIGHT:-0.10}"
export PAIR_TYPE_COMPATIBILITY_BIAS_WEIGHT="${PAIR_TYPE_COMPATIBILITY_BIAS_WEIGHT:-0.20}"
export PAIR_TYPE_FOCUS_WEIGHT="${PAIR_TYPE_FOCUS_WEIGHT:-2.0}"
export POSITIVE_FOCAL_LOSS_WEIGHT="${POSITIVE_FOCAL_LOSS_WEIGHT:-0.35}"
export POSITIVE_FOCAL_GAMMA="${POSITIVE_FOCAL_GAMMA:-1.5}"
export LONG_DISTANCE_SAMPLING_WEIGHT="${LONG_DISTANCE_SAMPLING_WEIGHT:-3.0}"
export USE_DIRECTION_EMBEDDING="${USE_DIRECTION_EMBEDDING:-1}"
export USE_WINDOW_POSITION_ENCODING="${USE_WINDOW_POSITION_ENCODING:-1}"

exec bash "${SCRIPT_DIR}/run_type_aux_stage2_long_distance.sh" "$@"
