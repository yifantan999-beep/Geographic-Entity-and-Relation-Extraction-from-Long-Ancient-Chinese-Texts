#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_predicted_long_learned_fusion}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_learned_fusion_type_threshold_candidates}"
NER_CALIBRATION="${NER_CALIBRATION:-}"
NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.5}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.3}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
CANDIDATE_THRESHOLD="${CANDIDATE_THRESHOLD:-0.10}"
DEVICE="${DEVICE:-}"
OFFLINE="${OFFLINE:-1}"

if [[ -n "${NER_CALIBRATION}" ]]; then
  mapfile -t ner_calibrated < <(
    "${PYTHON_BIN}" - "${NER_CALIBRATION}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("source_split") not in {"eval", "dev"} or not payload.get("best"):
    raise SystemExit("NER calibration must contain an eval/dev best result")
best = payload["best"]
grid = payload.get("grid", [])
for field in ("span_threshold", "type_threshold"):
    values = {row[field] for row in grid if field in row}
    if values and best[field] in {min(values), max(values)}:
        raise SystemExit(f"Best NER {field} is at the calibration grid boundary")
print(best["span_threshold"])
print(best["type_threshold"])
print(best["overlap_strategy"])
PY
  )
  if [[ "${#ner_calibrated[@]}" -ne 3 ]]; then
    echo "Expected three calibrated NER parameters, found ${#ner_calibrated[@]}" >&2
    exit 2
  fi
  NER_SPAN_THRESHOLD="${ner_calibrated[0]}"
  NER_TYPE_THRESHOLD="${ner_calibrated[1]}"
  NER_OVERLAP_STRATEGY="${ner_calibrated[2]}"
fi

args=(
  "${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}"
  --config "${CONFIG}"
  --split eval
  --ner_span_threshold "${NER_SPAN_THRESHOLD}"
  --ner_type_threshold "${NER_TYPE_THRESHOLD}"
  --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
  --relation_threshold "${CANDIDATE_THRESHOLD}"
  --review_margin 0.0
  --output_dir "${OUTPUT_DIR}"
)
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)
printf 'Running:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"

echo "Eval relation candidates written to: ${OUTPUT_DIR}/eval_predictions.jsonl"
