#!/usr/bin/env bash
set -euo pipefail

# One-command runner for Experiment 4 (Oracle/error propagation) and
# Experiment 5 (length/distance/cross-window degradation).
# Usage:
#   bash scripts/run_experiment4_5.sh \
#     artifacts/inference/.../test_predictions.jsonl \
#     artifacts/analysis/v8_fixed_test_exp45

PREDICTIONS="${1:?Pass predictions JSONL as the first argument}"
OUTPUT_DIR="${2:?Pass output directory as the second argument}"

python3 scripts/analyze_oracle_error_propagation.py \
  --predictions "$PREDICTIONS" \
  --output-dir "$OUTPUT_DIR/oracle"

python3 scripts/analyze_length_distance.py \
  --predictions "$PREDICTIONS" \
  --output-dir "$OUTPUT_DIR/length_distance"

echo "Experiment 4 outputs: $OUTPUT_DIR/oracle"
echo "Experiment 5 outputs: $OUTPUT_DIR/length_distance"
