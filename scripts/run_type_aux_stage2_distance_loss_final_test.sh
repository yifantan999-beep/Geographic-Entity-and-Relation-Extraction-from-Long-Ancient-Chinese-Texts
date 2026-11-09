#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${DOCUMENT_RE_CHECKPOINT:=${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01/document_relation_predicted_long_distance_loss}"
: "${RE_CALIBRATION:=${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_distance_loss_re_calibration.json}"
: "${OUTPUT_DIR:=${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_stage2_distance_loss_final_test}"
export DOCUMENT_RE_CHECKPOINT RE_CALIBRATION OUTPUT_DIR

exec bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion_final_test.sh" "$@"
