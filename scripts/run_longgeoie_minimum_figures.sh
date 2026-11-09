#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"

DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_fixed_boundary_focus}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_fixed_boundary_focus}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-$STAGE1_CHECKPOINT/document_relation_re_ablation_v8_fixed/full}"
ABLATION_RE_CHECKPOINT="${ABLATION_RE_CHECKPOINT:-$STAGE1_CHECKPOINT/document_relation_re_ablation_v8_fixed/no_long_distance}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-artifacts/inference/longgeoie_figure_trace_test}"
ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-artifacts/inference/longgeoie_figure_no_long_distance_test}"
DATA_DIR="${DATA_DIR:-artifacts/analysis/longgeoie_minimum_figures}"
FIGURE_DIR="${FIGURE_DIR:-docs/overleaf/figures}"

python3 infer_long_context.py \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --document_re_checkpoint "$DOCUMENT_RE_CHECKPOINT" \
  --config "$CONFIG" \
  --split test \
  --device "$DEVICE" \
  --batch_size 2 \
  --ner_span_threshold 0.80 \
  --ner_type_threshold 0.50 \
  --ner_overlap_strategy nms \
  --relation_threshold 0.24 \
  --max_document_entities 128 \
  --export_fusion_trace \
  --offline \
  --output_dir "$TRACE_OUTPUT_DIR"

python3 infer_long_context.py \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --document_re_checkpoint "$ABLATION_RE_CHECKPOINT" \
  --config "$CONFIG" \
  --split test \
  --device "$DEVICE" \
  --batch_size 2 \
  --ner_span_threshold 0.80 \
  --ner_type_threshold 0.50 \
  --ner_overlap_strategy nms \
  --relation_threshold 0.24 \
  --max_document_entities 128 \
  --offline \
  --output_dir "$ABLATION_OUTPUT_DIR"

python3 scripts/build_longgeoie_minimum_figures.py \
  --predictions "LongGeoIE=$TRACE_OUTPUT_DIR/test_predictions.jsonl" \
  --predictions "w/o Long-distance modules=$ABLATION_OUTPUT_DIR/test_predictions.jsonl" \
  --output-dir "$DATA_DIR" \
  --figure-dir "$FIGURE_DIR"

echo "Figure data: $DATA_DIR"
echo "Figure files: $FIGURE_DIR"
