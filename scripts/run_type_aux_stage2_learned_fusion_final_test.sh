#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_predicted_long_learned_fusion}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux}"
NER_CALIBRATION="${NER_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/long_context_stage1_type_aux_ner_calibration.json}"
RE_CALIBRATION="${RE_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_learned_fusion_re_calibration.json}"
RELATION_THRESHOLDS_FILE="${RELATION_THRESHOLDS_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_stage2_learned_fusion_final_test}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-64}"
DEVICE="${DEVICE:-}"
OFFLINE="${OFFLINE:-1}"

for path in \
  "${LOCAL_CHECKPOINT}/config.json" \
  "${DOCUMENT_RE_CHECKPOINT}/document_relation_config.json" \
  "${NER_CALIBRATION}" \
  "${RE_CALIBRATION}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 2
  fi
done
if [[ -n "${RELATION_THRESHOLDS_FILE}" && ! -f "${RELATION_THRESHOLDS_FILE}" ]]; then
  echo "Per-relation thresholds file does not exist: ${RELATION_THRESHOLDS_FILE}" >&2
  exit 2
fi
if [[ -f "${OUTPUT_DIR}/test_metrics.json" ]]; then
  echo "Refusing to overwrite existing final test: ${OUTPUT_DIR}/test_metrics.json" >&2
  exit 2
fi

mapfile -t calibrated < <(
  "${PYTHON_BIN}" - "${NER_CALIBRATION}" "${RE_CALIBRATION}" "${DOCUMENT_RE_CHECKPOINT}/document_relation_config.json" <<'PY'
import json
import sys
from pathlib import Path

ner = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
relation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
config = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
for name, payload in (("NER", ner), ("RE", relation)):
    if payload.get("source_split") not in {"eval", "dev"}:
        raise SystemExit(f"Refusing {name} calibration from {payload.get('source_split')!r}")
    if not payload.get("best"):
        raise SystemExit(f"{name} calibration has no best result")
if relation.get("fusion_mode") != "learned" or config.get("fusion_mode") != "learned":
    raise SystemExit("Calibration/checkpoint do not describe learned fusion")
if relation.get("best_at_search_boundary", True):
    raise SystemExit("Best RE threshold is at an unchecked grid boundary; recalibrate on eval")
ner_best = ner["best"]
grid = ner.get("grid", [])
for field in ("span_threshold", "type_threshold"):
    values = {row[field] for row in grid if field in row}
    if values and ner_best[field] in {min(values), max(values)}:
        raise SystemExit(f"Best NER {field} is at an unchecked grid boundary")
expected_ner = {
    "span_threshold": float(ner_best["span_threshold"]),
    "type_threshold": float(ner_best["type_threshold"]),
    "overlap_strategy": ner_best["overlap_strategy"],
}
if relation.get("ner_decoding") != expected_ner:
    raise SystemExit(
        f"RE calibration used different NER decoding: "
        f"{relation.get('ner_decoding')!r} != {expected_ner!r}"
    )
print(ner_best["span_threshold"])
print(ner_best["type_threshold"])
print(ner_best["overlap_strategy"])
print(relation["best"]["relation_threshold"])
PY
)
if [[ "${#calibrated[@]}" -ne 4 ]]; then
  echo "Expected four calibrated parameters, found ${#calibrated[@]}" >&2
  exit 2
fi

if [[ -n "${RELATION_THRESHOLDS_FILE}" ]]; then
  "${PYTHON_BIN}" - "${RELATION_THRESHOLDS_FILE}" "${RE_CALIBRATION}" "${NER_CALIBRATION}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("source_split") not in {"eval", "dev"}:
    raise SystemExit("Per-relation thresholds must come from eval/dev")
if payload.get("best_at_search_boundary", True):
    raise SystemExit("Per-relation threshold calibration is on a search boundary")
if not isinstance(payload.get("thresholds"), dict):
    raise SystemExit("Per-relation calibration has no thresholds object")
relation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if float(payload.get("global_threshold", -1.0)) != float(relation["best"]["relation_threshold"]):
    raise SystemExit("Per-relation calibration global threshold does not match RE calibration")
candidate_run = payload.get("candidate_run")
if not isinstance(candidate_run, dict):
    raise SystemExit("Per-relation calibration has no candidate run provenance")
ner = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["best"]
if (
    float(candidate_run.get("ner_span_threshold", -1.0)) != float(ner["span_threshold"])
    or float(candidate_run.get("ner_type_threshold", -1.0)) != float(ner["type_threshold"])
    or candidate_run.get("ner_overlap_strategy") != ner["overlap_strategy"]
):
    raise SystemExit("Per-relation candidates used different NER decoding")
PY
fi

args=(
  "${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}"
  --config "${CONFIG}"
  --split test
  --batch_size "${BATCH_SIZE}"
  --ner_span_threshold "${calibrated[0]}"
  --ner_type_threshold "${calibrated[1]}"
  --ner_overlap_strategy "${calibrated[2]}"
  --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
  --relation_threshold "${calibrated[3]}"
  --output_dir "${OUTPUT_DIR}"
)
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)
[[ -n "${RELATION_THRESHOLDS_FILE}" ]] && args+=(--relation_thresholds_file "${RELATION_THRESHOLDS_FILE}")
printf 'Running learned-fusion final test:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/test_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
overall = metrics["overall"]
long_256 = metrics["length_buckets"]["256-511"]
print(json.dumps({
    "test_metrics": str(metrics_path.resolve()),
    "fusion_mode": metrics["run"]["document_re_fusion_mode"],
    "ner_strict_f1": overall["ner_strict"]["f1"],
    "re_strict_f1": overall["re_strict_end_to_end"]["f1"],
    "re_boundary_f1": overall["re_boundary_end_to_end"]["f1"],
    "re_on_matched_entities_f1": overall["re_on_matched_entities"]["f1"],
    "re_strict_256_511_f1": long_256["re_strict_end_to_end"]["f1"],
    "fusion_diagnostics": metrics.get("fusion_diagnostics"),
}, ensure_ascii=False, indent=2))
PY

echo "Learned-fusion final test is complete; do not retune it on test."
