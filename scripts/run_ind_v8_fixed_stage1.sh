#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_ind_v8_fixed}"
TRANSFORMER="${TRANSFORMER:-local-bert/google-bert/bert-base-chinese}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01}"
EVIDENCE="${EVIDENCE:-${PROJECT_ROOT}/artifacts/calibration/ind_v8_fixed_ner_evidence.json}"
CALIBRATION="${CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/ind_v8_fixed_ner_calibration.json}"
EVAL_DIR="${EVAL_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_v8_fixed_stage1_eval}"

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
  [[ ! -e "${CHECKPOINT}" ]] || { echo "Checkpoint exists but is incomplete: ${CHECKPOINT}" >&2; exit 2; }
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "${PYTHON_BIN}" train_long_context.py --transformer "${TRANSFORMER}" --config "${CONFIG}" \
    --model_path "${CHECKPOINT}" --epochs "${EPOCHS:-20}" --patience "${PATIENCE:-6}" \
    --batch_size 1 --gradient_accumulation 8 --bfloat16 --seed 42 --rebuild_cache \
    --skip_final_test --offline
else
  echo "Stage-1 checkpoint already exists; reusing ${CHECKPOINT}"
fi

if [[ ! -f "${EVIDENCE}" ]]; then
  args=("${PYTHON_BIN}" infer_long_context.py --checkpoint "${CHECKPOINT}" --config "${CONFIG}" --split eval --batch_size 2 --max_document_entities 128 --dump_ner_evidence "${EVIDENCE}" --offline)
  [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
  "${args[@]}"
fi

if [[ ! -f "${CALIBRATION}" ]]; then
  "${PYTHON_BIN}" calibrate_long_context_ner.py --evidence "${EVIDENCE}" --output "${CALIBRATION}" \
    --span_thresholds 0.40 0.50 0.60 0.70 0.75 0.80 0.85 0.90 0.95 \
    --type_thresholds 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 \
    --overlap_strategies nms weighted_interval none
fi

"${PYTHON_BIN}" - "${CALIBRATION}" <<'PY'
import json
import sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
if m.get("source_split") not in {"eval", "dev"} or not m.get("best"):
    raise SystemExit("NER calibration must come from eval/dev")
b = m["best"]
print(json.dumps({"source_split": m.get("source_split"), "best": b,
                  "best_at_search_boundary": m.get("best_at_search_boundary")},
                 ensure_ascii=False, indent=2))
PY

if [[ ! -f "${EVAL_DIR}/eval_metrics.json" ]]; then
  mapfile -t ner < <("${PYTHON_BIN}" - "${CALIBRATION}" <<'PY'
import json
import sys
b = json.load(open(sys.argv[1], encoding="utf-8"))["best"]
print(b["span_threshold"])
print(b["type_threshold"])
print(b["overlap_strategy"])
PY
  )
  args=("${PYTHON_BIN}" infer_long_context.py --checkpoint "${CHECKPOINT}" --config "${CONFIG}" --split eval --batch_size 2 --max_document_entities 128 --ner_span_threshold "${ner[0]}" --ner_type_threshold "${ner[1]}" --ner_overlap_strategy "${ner[2]}" --output_dir "${EVAL_DIR}" --offline)
  [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
  "${args[@]}"
fi

echo "Stage-1 checkpoint: ${CHECKPOINT}"
echo "NER calibration: ${CALIBRATION}"
echo "Stage-1 eval metrics: ${EVAL_DIR}/eval_metrics.json"
