#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_predicted_long}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux}"
NER_CALIBRATION="${NER_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/long_context_stage1_type_aux_ner_calibration.json}"
RE_CALIBRATION="${RE_CALIBRATION:-${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_re_fine_calibration.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_stage2_final_test}"
BATCH_SIZE="${BATCH_SIZE:-2}"
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

if [[ -f "${OUTPUT_DIR}/test_metrics.json" ]]; then
  echo "Final test metrics already exist; refusing to overwrite: ${OUTPUT_DIR}/test_metrics.json" >&2
  echo "Use the existing result. Do not rerun test for parameter selection." >&2
  exit 2
fi

mapfile -t calibrated < <(
  "${PYTHON_BIN}" - "${NER_CALIBRATION}" "${RE_CALIBRATION}" <<'PY'
import json
import sys
from pathlib import Path

ner = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
relation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for name, payload in (("NER", ner), ("RE", relation)):
    if payload.get("source_split") not in {"eval", "dev"}:
        raise SystemExit(
            f"Refusing {name} calibration from split={payload.get('source_split')!r}; "
            "final test parameters must come from eval/dev"
        )
    if not payload.get("best"):
        raise SystemExit(f"{name} calibration has no best result")

ner_best = ner["best"]
re_best = relation["best"]
print(ner_best["span_threshold"])
print(ner_best["type_threshold"])
print(ner_best["overlap_strategy"])
print(re_best["relation_threshold"])
print(re_best["local_global_alpha"])
PY
)

if [[ "${#calibrated[@]}" -ne 5 ]]; then
  echo "Expected five calibrated parameters, found ${#calibrated[@]}" >&2
  exit 2
fi

NER_SPAN_THRESHOLD="${calibrated[0]}"
NER_TYPE_THRESHOLD="${calibrated[1]}"
NER_OVERLAP_STRATEGY="${calibrated[2]}"
RELATION_THRESHOLD="${calibrated[3]}"
LOCAL_GLOBAL_ALPHA="${calibrated[4]}"

args=(
  "${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}"
  --config "${CONFIG}"
  --split test
  --batch_size "${BATCH_SIZE}"
  --ner_span_threshold "${NER_SPAN_THRESHOLD}"
  --ner_type_threshold "${NER_TYPE_THRESHOLD}"
  --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
  --relation_threshold "${RELATION_THRESHOLD}"
  --local_global_alpha "${LOCAL_GLOBAL_ALPHA}"
  --output_dir "${OUTPUT_DIR}"
)
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)

echo "Frozen NER: span=${NER_SPAN_THRESHOLD} type=${NER_TYPE_THRESHOLD} strategy=${NER_OVERLAP_STRATEGY}"
echo "Frozen RE: threshold=${RELATION_THRESHOLD} alpha=${LOCAL_GLOBAL_ALPHA}"
printf 'Running final test:'
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
summary = {
    "test_metrics": str(metrics_path.resolve()),
    "ner_strict_f1": overall["ner_strict"]["f1"],
    "ner_boundary_f1": overall["ner_boundary"]["f1"],
    "re_strict_f1": overall["re_strict_end_to_end"]["f1"],
    "re_boundary_f1": overall["re_boundary_end_to_end"]["f1"],
    "re_on_matched_entities_f1": overall["re_on_matched_entities"]["f1"],
    "re_strict_256_511_f1": long_256["re_strict_end_to_end"]["f1"],
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Final test is complete. Do not use this result to change thresholds or fusion alpha."
