#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${TRAINING_CONFIG:=${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_type_balanced.json}"
: "${DOCUMENT_RE_DIR:=${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01/document_relation_predicted_long_type_balanced}"
: "${PREDICTIONS_DIR:=${PROJECT_ROOT}/artifacts/predicted_entities/long_context_type_aux_type_balanced_scores}"
: "${EVAL_OUTPUT_DIR:=${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_stage2_type_balanced_eval_default}"
export TRAINING_CONFIG DOCUMENT_RE_DIR PREDICTIONS_DIR EVAL_OUTPUT_DIR

exec bash "${SCRIPT_DIR}/run_type_aux_stage2_learned_fusion.sh" "$@"
