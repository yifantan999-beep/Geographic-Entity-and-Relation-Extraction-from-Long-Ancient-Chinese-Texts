#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_predicted_long_learned_fusion}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux}"
GRID_DIR="${GRID_DIR:-${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_learned_fusion_re_grid}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_learned_fusion_re_calibration.json}"
NER_CALIBRATION="${NER_CALIBRATION:-}"

NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.5}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.3}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
# The learned checkpoint selected 0.20 internally.  Keep the calibration grid
# dense around that eval-only operating region; a best point at a search edge
# must not be promoted to final test.
RELATION_THRESHOLDS="${RELATION_THRESHOLDS:-0.10 0.14 0.18 0.20 0.22 0.26 0.30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-64}"
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

"${PYTHON_BIN}" - "${DOCUMENT_RE_CHECKPOINT}/document_relation_config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Learned-fusion checkpoint config does not exist: {path}")
config = json.loads(path.read_text(encoding="utf-8"))
if config.get("fusion_mode") != "learned":
    raise SystemExit(f"Refusing non-learned checkpoint: fusion_mode={config.get('fusion_mode')!r}")
PY

read -r -a threshold_values <<< "${RELATION_THRESHOLDS}"
mkdir -p "${GRID_DIR}"
for threshold in "${threshold_values[@]}"; do
  output_dir="${GRID_DIR}/threshold_${threshold}"
  args=(
    "${PYTHON_BIN}" infer_long_context.py
    --checkpoint "${LOCAL_CHECKPOINT}"
    --document_re_checkpoint "${DOCUMENT_RE_CHECKPOINT}"
    --config "${CONFIG}"
    --split eval
    --batch_size "${BATCH_SIZE}"
    --ner_span_threshold "${NER_SPAN_THRESHOLD}"
    --ner_type_threshold "${NER_TYPE_THRESHOLD}"
    --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
    --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
    --relation_threshold "${threshold}"
    --output_dir "${output_dir}"
  )
  [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
  [[ "${OFFLINE}" == "1" ]] && args+=(--offline)
  echo "threshold=${threshold}"
  "${args[@]}"
done

"${PYTHON_BIN}" - "${GRID_DIR}" "${OUTPUT}" "${NER_SPAN_THRESHOLD}" "${NER_TYPE_THRESHOLD}" "${NER_OVERLAP_STRATEGY}" <<'PY'
import json
import sys
from pathlib import Path

grid_dir, output = map(Path, sys.argv[1:3])
expected_ner = {
    "span_threshold": float(sys.argv[3]),
    "type_threshold": float(sys.argv[4]),
    "overlap_strategy": sys.argv[5],
}
rows = []
for metrics_path in sorted(grid_dir.glob("*/eval_metrics.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run = metrics.get("run", {})
    if run.get("split") != "eval" or run.get("document_re_fusion_mode") != "learned":
        raise SystemExit(f"Refusing incompatible calibration metrics: {metrics_path}")
    actual_ner = {
        "span_threshold": float(run.get("ner_span_threshold")),
        "type_threshold": float(run.get("ner_type_threshold")),
        "overlap_strategy": run.get("ner_overlap_strategy"),
    }
    if actual_ner != expected_ner:
        raise SystemExit(
            f"NER decoding mismatch in {metrics_path}: {actual_ner} != {expected_ner}"
        )
    overall = metrics["overall"]
    rows.append({
        "relation_threshold": float(run["relation_threshold"]),
        "re_strict": overall["re_strict_end_to_end"],
        "re_boundary": overall["re_boundary_end_to_end"],
        "re_on_matched_entities": overall["re_on_matched_entities"],
        "re_strict_256_511": metrics["length_buckets"]["256-511"]["re_strict_end_to_end"],
        "metrics": str(metrics_path.resolve()),
    })
if len(rows) != len(list(grid_dir.glob("*/eval_metrics.json"))) or not rows:
    raise SystemExit("No complete learned-fusion eval rows were found")
rows.sort(key=lambda row: (
    row["re_strict"]["f1"],
    row["re_strict_256_511"]["f1"],
    row["re_boundary"]["f1"],
    row["re_on_matched_entities"]["f1"],
    row["re_strict"]["precision"],
), reverse=True)
result = {
    "source_split": "eval",
        "fusion_mode": "learned",
        "ner_decoding": expected_ner,
    "best_at_search_boundary": rows[0]["relation_threshold"] in {
        min(row["relation_threshold"] for row in rows),
        max(row["relation_threshold"] for row in rows),
    },
    "selection_metric": (
        "overall.re_strict_end_to_end.f1, then 256-511 RE strict F1, "
        "then RE boundary F1, matched-entity F1, and precision"
    ),
    "best": rows[0],
    "grid": rows,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"source_split": "eval", "best": rows[0]}, ensure_ascii=False, indent=2))
if result["best_at_search_boundary"]:
    print("WARNING: best threshold is on the calibration boundary; expand eval grid before test")
print(f"Calibration written to: {output.resolve()}")
PY

echo "Do not run learned-fusion test until this eval result has been reviewed."
