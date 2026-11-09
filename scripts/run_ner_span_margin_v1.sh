#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_fixed_span_margin_v1}"
OLD_CHECKPOINT="${OLD_CHECKPOINT:-$ROOT/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_target}"
CHECKPOINT="${CHECKPOINT:-$ROOT/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_span_margin_v1}"
EVAL_DIR="${EVAL_DIR:-$ROOT/artifacts/inference/diagnostic_easy_v8_span_margin_v1_eval}"
TEST_DIR="${TEST_DIR:-$ROOT/artifacts/inference/diagnostic_easy_v8_span_margin_v1_test}"
DEVICE="${DEVICE:-cuda:0}"

if [[ ! -f "$CHECKPOINT/config.json" ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train_long_context.py \
    --transformer local-bert/google-bert/bert-base-chinese \
    --config "$CONFIG" --model_path "$CHECKPOINT" --load_ckpt "$OLD_CHECKPOINT" \
    --epochs "${EPOCHS:-20}" --patience "${PATIENCE:-6}" --batch_size 1 \
    --gradient_accumulation 8 --bfloat16 --seed 42 --rebuild_cache --skip_final_test --offline
fi

python3 infer_long_context.py --checkpoint "$CHECKPOINT" --config "$CONFIG" --split eval \
  --device "$DEVICE" --batch_size 1 --ner_span_threshold 0.80 --ner_type_threshold 0.50 \
  --ner_overlap_strategy nms --max_document_entities 128 --output_dir "$EVAL_DIR" --offline

python3 infer_long_context.py --checkpoint "$CHECKPOINT" --config "$CONFIG" --split test \
  --device "$DEVICE" --batch_size 1 --ner_span_threshold 0.80 --ner_type_threshold 0.50 \
  --ner_overlap_strategy nms --max_document_entities 128 --output_dir "$TEST_DIR" --offline

echo "NER eval: $EVAL_DIR/eval_metrics.json"
echo "NER test: $TEST_DIR/test_metrics.json"
