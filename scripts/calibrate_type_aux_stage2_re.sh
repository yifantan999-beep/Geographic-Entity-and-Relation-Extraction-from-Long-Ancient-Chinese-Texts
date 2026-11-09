#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
DOCUMENT_RE_CHECKPOINT="${DOCUMENT_RE_CHECKPOINT:-${LOCAL_CHECKPOINT}/document_relation_predicted_long}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux}"
GRID_DIR="${GRID_DIR:-${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_re_grid}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/artifacts/calibration/long_context_type_aux_stage2_re_calibration.json}"

NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.5}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.3}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
RELATION_THRESHOLDS="${RELATION_THRESHOLDS:-0.56 0.64 0.72}"
LOCAL_GLOBAL_ALPHAS="${LOCAL_GLOBAL_ALPHAS:-0.35 0.50 0.65 0.80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DEVICE="${DEVICE:-}"
OFFLINE="${OFFLINE:-1}"

if [[ ! -f "${LOCAL_CHECKPOINT}/config.json" ]]; then
  echo "Type-aux checkpoint is missing config.json: ${LOCAL_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${DOCUMENT_RE_CHECKPOINT}/document_relation_config.json" ]]; then
  echo "Document RE checkpoint is missing its config: ${DOCUMENT_RE_CHECKPOINT}" >&2
  exit 2
fi

read -r -a threshold_values <<< "${RELATION_THRESHOLDS}"
read -r -a alpha_values <<< "${LOCAL_GLOBAL_ALPHAS}"
mkdir -p "${GRID_DIR}"

total=$(( ${#threshold_values[@]} * ${#alpha_values[@]} ))
index=0
for alpha in "${alpha_values[@]}"; do
  for threshold in "${threshold_values[@]}"; do
    index=$((index + 1))
    run_name="alpha_${alpha}_threshold_${threshold}"
    output_dir="${GRID_DIR}/${run_name}"
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
      --relation_threshold "${threshold}"
      --local_global_alpha "${alpha}"
      --output_dir "${output_dir}"
    )
    [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
    [[ "${OFFLINE}" == "1" ]] && args+=(--offline)

    echo "[${index}/${total}] alpha=${alpha} threshold=${threshold}"
    printf 'Running:'
    printf ' %q' "${args[@]}"
    printf '\n'
    "${args[@]}"
  done
done

"${PYTHON_BIN}" - "${GRID_DIR}" "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

grid_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
for metrics_path in sorted(grid_dir.glob("*/eval_metrics.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("run", {}).get("split") != "eval":
        raise SystemExit(f"Refusing non-eval metrics: {metrics_path}")
    overall = metrics["overall"]
    long_256 = metrics["length_buckets"]["256-511"]
    rows.append({
        "relation_threshold": float(metrics["run"]["relation_threshold"]),
        "local_global_alpha": float(metrics["run"]["local_global_alpha"]),
        "re_strict": overall["re_strict_end_to_end"],
        "re_boundary": overall["re_boundary_end_to_end"],
        "re_on_matched_entities": overall["re_on_matched_entities"],
        "re_strict_256_511": long_256["re_strict_end_to_end"],
        "metrics": str(metrics_path.resolve()),
    })

expected = len(list(grid_dir.glob("*/eval_metrics.json")))
if not rows or expected != len(rows):
    raise SystemExit("No complete eval relation-calibration rows were found")

rows.sort(
    key=lambda row: (
        row["re_strict"]["f1"],
        row["re_strict_256_511"]["f1"],
        row["re_boundary"]["f1"],
        row["re_on_matched_entities"]["f1"],
        row["re_strict"]["precision"],
    ),
    reverse=True,
)
result = {
    "source_split": "eval",
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
print(f"Calibration written to: {output.resolve()}")
PY

echo "Do not run test until the best eval parameters are fixed."
