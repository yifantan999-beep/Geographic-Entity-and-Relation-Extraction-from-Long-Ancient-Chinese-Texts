#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${DOCUMENT_RE_CHECKPOINT:=${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01/document_relation_predicted_long_distance_loss}"
: "${GRID_DIR:=${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_distance_loss_re_grid}"
: "${OUTPUT:=${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_distance_loss_re_calibration.json}"
# The distance auxiliary loss shifts score calibration upward. Include the
# default eval operating point (0.40) and a broad range before rejecting the
# experiment; the learned-fusion narrow grid is intentionally not reused here.
: "${RELATION_THRESHOLDS:=0.10 0.14 0.18 0.20 0.22 0.26 0.30 0.34 0.38 0.40 0.44 0.50 0.60 0.70}"
export DOCUMENT_RE_CHECKPOINT GRID_DIR OUTPUT
export RELATION_THRESHOLDS

exec bash "${SCRIPT_DIR}/calibrate_type_aux_stage2_learned_fusion.sh" "$@"
