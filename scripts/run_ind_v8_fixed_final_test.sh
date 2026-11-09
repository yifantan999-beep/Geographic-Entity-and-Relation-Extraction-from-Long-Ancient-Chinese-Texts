#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_ind_v8_fixed}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${CHECKPOINT}/document_relation_distance_experts_v8}"
NER_CALIBRATION="${NER_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/ind_v8_fixed_ner_calibration.json}"
RE_CALIBRATION="${RE_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/ind_v8_fixed_re_calibration.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/ind_v8_fixed_final_test}"
for path in "${CHECKPOINT}/config.json" "${DOCUMENT_RE_DIR}/document_relation_config.json" "${NER_CALIBRATION}" "${RE_CALIBRATION}"; do
  [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 2; }
done
[[ ! -f "${OUTPUT_DIR}/test_metrics.json" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}/test_metrics.json; set OUTPUT_DIR to a new directory." >&2; exit 2; }
mapfile -t values < <("${PYTHON_BIN}" - "${NER_CALIBRATION}" "${RE_CALIBRATION}" <<'PY'
import json
import sys
ner = json.load(open(sys.argv[1], encoding="utf-8"))
rel = json.load(open(sys.argv[2], encoding="utf-8"))
if ner.get("best_at_search_boundary"):
    raise SystemExit("NER calibration best is at a search boundary; expand eval grid first")
if rel.get("best_at_search_boundary"):
    raise SystemExit("RE calibration best is at a search boundary; expand eval grid first")
n, r = ner["best"], rel["best"]
print(n["span_threshold"])
print(n["type_threshold"])
print(n["overlap_strategy"])
print(r["relation_threshold"])
PY
)
[[ "${#values[@]}" -eq 4 ]] || { echo "Expected four calibrated values, got ${#values[@]}" >&2; exit 2; }
args=("${PYTHON_BIN}" infer_long_context.py --checkpoint "${CHECKPOINT}" --document_re_checkpoint "${DOCUMENT_RE_DIR}" --config "${CONFIG}" --split test --batch_size 2 --max_document_entities 128 --ner_span_threshold "${values[0]}" --ner_type_threshold "${values[1]}" --ner_overlap_strategy "${values[2]}" --relation_threshold "${values[3]}" --review_margin 0.0 --output_dir "${OUTPUT_DIR}" --offline)
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
"${args[@]}"
"${PYTHON_BIN}" -c 'import json,sys; m=json.load(open(sys.argv[1],encoding="utf-8")); print(json.dumps({k:m["overall"][k] for k in ("ner_strict","ner_boundary","re_strict_end_to_end","re_boundary_end_to_end","re_on_matched_entities")},ensure_ascii=False,indent=2))' "${OUTPUT_DIR}/test_metrics.json"
echo "IND v8 fixed final test complete: ${OUTPUT_DIR}/test_metrics.json"
