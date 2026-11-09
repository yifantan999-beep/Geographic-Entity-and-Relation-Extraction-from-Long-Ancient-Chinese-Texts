#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_v5}"
export LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_span_v5}"
export DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_long_distance_v2}"
export NER_CALIBRATION="${NER_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/ind_distance_gt28_rebalanced_cross_window_v5_ner_calibration.json}"
export GRID_DIR="${GRID_DIR:-${PROJECT_ROOT}/artifacts/calibration/ind_distance_gt28_rebalanced_cross_window_v5_re_grid}"
export OUTPUT="${OUTPUT:-${PROJECT_ROOT}/artifacts/calibration/ind_distance_gt28_rebalanced_cross_window_v5_re_calibration.json}"
export NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.40}"
export NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.80}"
export NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
export MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-128}"
export RELATION_THRESHOLDS="${RELATION_THRESHOLDS:-0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.70 0.80 0.90}"

exec bash "${SCRIPT_DIR}/calibrate_type_aux_stage2_long_distance.sh" "$@"
