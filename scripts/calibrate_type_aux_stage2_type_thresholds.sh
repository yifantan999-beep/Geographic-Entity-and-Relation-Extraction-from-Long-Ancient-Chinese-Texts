#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PREDICTIONS="${PREDICTIONS:-${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_learned_fusion_type_threshold_candidates/eval_predictions.jsonl}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_learned_fusion_type_thresholds.json}"
MIN_SUPPORT="${MIN_SUPPORT:-8}"
GLOBAL_THRESHOLD="${GLOBAL_THRESHOLD:-0.20}"
THRESHOLD_GRID="${THRESHOLD_GRID:-0.10 0.14 0.18 0.20 0.22 0.26 0.30 0.34 0.38 0.40 0.44 0.50 0.60 0.70}"

read -r -a threshold_values <<< "${THRESHOLD_GRID}"

"${PYTHON_BIN}" calibrate_relation_type_thresholds.py \
  --predictions "${PREDICTIONS}" \
  --output "${OUTPUT}" \
  --source-split eval \
  --global-threshold "${GLOBAL_THRESHOLD}" \
  --min-support "${MIN_SUPPORT}" \
  --threshold-grid "${threshold_values[@]}"
