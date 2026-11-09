#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate IND test with pre-declared fixed decoding parameters.
# No calibration file is read and no threshold is searched on test.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_ind_v8_fixed}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${CHECKPOINT}/document_relation_distance_experts_v8}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_v8_fixed_fixed_test}"

# Fixed before test; override only for a pre-registered comparison.
NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.80}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.50}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-0.24}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-128}"

for path in "${CHECKPOINT}/config.json" "${DOCUMENT_RE_DIR}/document_relation.safetensors"; do
  [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 2; }
done
[[ ! -f "${OUTPUT_DIR}/test_metrics.json" ]] || {
  echo "Refusing to overwrite ${OUTPUT_DIR}/test_metrics.json; set OUTPUT_DIR to a new directory." >&2
  exit 2
}

args=("${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_DIR}"
  --config "${CONFIG}"
  --split test
  --batch_size "${BATCH_SIZE}"
  --ner_span_threshold "${NER_SPAN_THRESHOLD}"
  --ner_type_threshold "${NER_TYPE_THRESHOLD}"
  --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
  --relation_threshold "${RELATION_THRESHOLD}"
  --review_margin 0.0
  --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
  --output_dir "${OUTPUT_DIR}")
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)

printf 'Running fixed IND test:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/test_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
run = metrics.get("run", {})
overall = metrics["overall"]
print(json.dumps({
    "split": run.get("split"),
    "config": run.get("config"),
    "checkpoint": run.get("checkpoint"),
    "ner_span_threshold": run.get("ner_span_threshold"),
    "ner_type_threshold": run.get("ner_type_threshold"),
    "ner_overlap_strategy": run.get("ner_overlap_strategy"),
    "relation_threshold": run.get("relation_threshold"),
    "ner_strict": overall["ner_strict"],
    "ner_boundary": overall["ner_boundary"],
    "re_strict_end_to_end": overall["re_strict_end_to_end"],
    "re_boundary_end_to_end": overall["re_boundary_end_to_end"],
    "re_on_matched_entities": overall["re_on_matched_entities"],
    "macro_average": metrics.get("macro_average"),
    "relation_distance_buckets": metrics.get("relation_distance_buckets"),
    "window_relation_coverage": metrics.get("window_relation_coverage"),
}, ensure_ascii=False, indent=2))
print(f"Metrics written to: {path.resolve()}")
PY
