#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window}"
export LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_01}"
export DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_long_distance_v1}"
export NER_CALIBRATION="${NER_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/ind_distance_gt28_rebalanced_cross_window_stage1_ner_calibration.json}"
export RE_CALIBRATION="${RE_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/ind_distance_gt28_rebalanced_cross_window_stage2_long_distance_re_calibration.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_distance_gt28_rebalanced_cross_window_stage2_long_distance_final_test}"
export MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-128}"
unset RELATION_THRESHOLDS_FILE

exec bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion_final_test.sh" "$@"
