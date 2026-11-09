#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window}"
export LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_01}"
export DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_long_distance_v1}"
export NER_CALIBRATION="${NER_CALIBRATION:-}"
export GRID_DIR="${GRID_DIR:-${PROJECT_ROOT}/artifacts/calibration/ind_distance_gt28_rebalanced_cross_window_stage2_long_distance_re_grid}"
export OUTPUT="${OUTPUT:-${PROJECT_ROOT}/artifacts/calibration/ind_distance_gt28_rebalanced_cross_window_stage2_long_distance_re_calibration.json}"
export NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.8}"
export NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.35}"
export NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
export MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-128}"
export RELATION_THRESHOLDS="${RELATION_THRESHOLDS:-0.10 0.14 0.18 0.20 0.24 0.28 0.32 0.36 0.40 0.44 0.48 0.52 0.56 0.60 0.64 0.68 0.72 0.78 0.84 0.90}"

exec bash "${SCRIPT_DIR}/calibrate_type_aux_stage2_learned_fusion.sh" "$@"
