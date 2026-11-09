#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/long_context_type_aux_fusion_scores}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${LOCAL_CHECKPOINT}/document_relation_predicted_long_learned_fusion}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_stage2_learned_fusion_eval_default}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_learned_fusion.json}"
NER_CALIBRATION="${NER_CALIBRATION:-}"

NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.5}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.3}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-64}"
DEVICE="${DEVICE:-}"
OFFLINE="${OFFLINE:-1}"
BFLOAT16="${BFLOAT16:-0}"
DISTANCE_POSITIVE_LOSS_WEIGHT="${DISTANCE_POSITIVE_LOSS_WEIGHT:-}"
DISTANCE_LOSS_BOUNDARIES="${DISTANCE_LOSS_BOUNDARIES:-}"
DISTANCE_LOSS_WEIGHTS="${DISTANCE_LOSS_WEIGHTS:-}"
DISTANCE_CONDITIONED_FUSION="${DISTANCE_CONDITIONED_FUSION:-0}"
FUSION_LONG_DISTANCE_BOUNDARY="${FUSION_LONG_DISTANCE_BOUNDARY:-}"
FUSION_DISTANCE_HIDDEN_SIZE="${FUSION_DISTANCE_HIDDEN_SIZE:-}"
FUSION_LONG_DISTANCE_BIAS="${FUSION_LONG_DISTANCE_BIAS:-}"
LONG_DISTANCE_HEAD="${LONG_DISTANCE_HEAD:-0}"
LONG_DISTANCE_BOUNDARY="${LONG_DISTANCE_BOUNDARY:-128}"
LONG_DISTANCE_HEAD_WEIGHT="${LONG_DISTANCE_HEAD_WEIGHT:-}"
LONG_DISTANCE_RANKING_LOSS_WEIGHT="${LONG_DISTANCE_RANKING_LOSS_WEIGHT:-}"
LONG_DISTANCE_RANKING_MARGIN="${LONG_DISTANCE_RANKING_MARGIN:-}"
LONG_DISTANCE_RANKING_MAX_NEGATIVES="${LONG_DISTANCE_RANKING_MAX_NEGATIVES:-}"
PAIR_TYPE_COMPATIBILITY="${PAIR_TYPE_COMPATIBILITY:-0}"
PAIR_TYPE_COMPATIBILITY_LOSS_WEIGHT="${PAIR_TYPE_COMPATIBILITY_LOSS_WEIGHT:-}"
PAIR_TYPE_COMPATIBILITY_BIAS_WEIGHT="${PAIR_TYPE_COMPATIBILITY_BIAS_WEIGHT:-}"
PAIR_TYPE_FOCUS_PAIRS="${PAIR_TYPE_FOCUS_PAIRS:-}"
PAIR_TYPE_FOCUS_WEIGHT="${PAIR_TYPE_FOCUS_WEIGHT:-}"
POSITIVE_FOCAL_LOSS_WEIGHT="${POSITIVE_FOCAL_LOSS_WEIGHT:-}"
POSITIVE_FOCAL_GAMMA="${POSITIVE_FOCAL_GAMMA:-}"
LONG_DISTANCE_SAMPLING_WEIGHT="${LONG_DISTANCE_SAMPLING_WEIGHT:-}"
USE_DIRECTION_EMBEDDING="${USE_DIRECTION_EMBEDDING:-0}"
USE_WINDOW_POSITION_ENCODING="${USE_WINDOW_POSITION_ENCODING:-0}"
FIXED_EVAL_THRESHOLD="${FIXED_EVAL_THRESHOLD:-}"
EVAL_RELATION_THRESHOLD="${EVAL_RELATION_THRESHOLD:-${FIXED_EVAL_THRESHOLD:-}}"
EVAL_REVIEW_MARGIN="${EVAL_REVIEW_MARGIN:-0.0}"
RECALL_HEAD="${RECALL_HEAD:-0}"
RECALL_HEAD_SIZE="${RECALL_HEAD_SIZE:-}"
RECALL_HEAD_WEIGHT="${RECALL_HEAD_WEIGHT:-}"
RECALL_MARGIN_LOSS_WEIGHT="${RECALL_MARGIN_LOSS_WEIGHT:-}"
RECALL_THRESHOLD="${RECALL_THRESHOLD:-}"
RECALL_MARGIN="${RECALL_MARGIN:-}"
RECALL_SHORT_DISTANCE_BOUNDARY="${RECALL_SHORT_DISTANCE_BOUNDARY:-}"
RECALL_SHORT_DISTANCE_WEIGHT="${RECALL_SHORT_DISTANCE_WEIGHT:-}"
PAIR_LINK_GATE_WEIGHT="${PAIR_LINK_GATE_WEIGHT:-}"
PAIR_LINK_POSITIVE_WEIGHT="${PAIR_LINK_POSITIVE_WEIGHT:-}"

if [[ -n "${NER_CALIBRATION}" ]]; then
  if [[ ! -f "${NER_CALIBRATION}" ]]; then
    echo "NER calibration does not exist: ${NER_CALIBRATION}" >&2
    exit 2
  fi
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

for path in "${LOCAL_CHECKPOINT}/config.json" "${TRAINING_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${PREDICTIONS_DIR}"

run_stage1_split() {
  local split="$1"
  local args=(
    "${PYTHON_BIN}" infer_long_context.py
    --checkpoint "${LOCAL_CHECKPOINT}"
    --config "${CONFIG}"
    --split "${split}"
    --batch_size "${BATCH_SIZE}"
    --ner_span_threshold "${NER_SPAN_THRESHOLD}"
    --ner_type_threshold "${NER_TYPE_THRESHOLD}"
    --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
    --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
    --export_local_relation_scores
    --output_dir "${PREDICTIONS_DIR}"
  )
  [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
  [[ "${OFFLINE}" == "1" ]] && args+=(--offline)
  printf 'Running:'
  printf ' %q' "${args[@]}"
  printf '\n'
  "${args[@]}"
}

run_stage1_split train
run_stage1_split eval

  "${PYTHON_BIN}" - "${PREDICTIONS_DIR}" "${NER_SPAN_THRESHOLD}" "${NER_TYPE_THRESHOLD}" "${NER_OVERLAP_STRATEGY}" "${CONFIG}" "${LOCAL_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
expected_ner = {
    "span_threshold": float(sys.argv[2]),
    "type_threshold": float(sys.argv[3]),
    "overlap_strategy": sys.argv[4],
}
expected_config = sys.argv[5]
expected_checkpoint = Path(sys.argv[6]).resolve()
split_runs = {}
for split in ("train", "eval"):
    path = directory / f"{split}_predictions.jsonl"
    metrics_path = directory / f"{split}_metrics.json"
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    expected = len(records)
    if not records or {int(row["document_id"]) for row in records} != set(range(expected)):
        raise SystemExit(f"{path}: document ids must be contiguous 0..{expected - 1}")
    score_count = 0
    for row in records:
        scores = row.get("local_relation_scores")
        entities = row.get("prediction", {}).get("entities", [])
        if not isinstance(scores, list):
            raise SystemExit(f"{path}: document {row['document_id']} has no exported local scores")
        for score in scores:
            if not (isinstance(score, list) and len(score) == 4):
                raise SystemExit(f"{path}: local score does not use the compact four-field schema")
            if not (0 <= int(score[0]) < len(entities) and 0 <= int(score[1]) < len(entities)):
                raise SystemExit(f"{path}: local score has an invalid entity index")
        score_count += len(scores)
    if score_count == 0:
        raise SystemExit(f"{path}: all exported local score lists are empty")
    if not metrics_path.exists():
        raise SystemExit(f"Missing inference provenance: {metrics_path}")
    run = json.loads(metrics_path.read_text(encoding="utf-8")).get("run", {})
    actual_ner = {
        "span_threshold": float(run.get("ner_span_threshold")),
        "type_threshold": float(run.get("ner_type_threshold")),
        "overlap_strategy": run.get("ner_overlap_strategy"),
    }
    if actual_ner != expected_ner:
        raise SystemExit(f"{metrics_path}: NER decoding mismatch: {actual_ner} != {expected_ner}")
    if run.get("split") != split or run.get("config") not in {expected_config, f"cfg/{expected_config}"}:
        raise SystemExit(f"{metrics_path}: split/config provenance mismatch")
    if Path(run.get("checkpoint", "")).resolve() != expected_checkpoint:
        raise SystemExit(f"{metrics_path}: checkpoint provenance mismatch")
    split_runs[split] = run
    print(f"Validated {split}: {len(records)} documents, {score_count} local scores")
(directory / "prediction_manifest.json").write_text(
    json.dumps({
        "ner_decoding": expected_ner,
        "config": expected_config,
        "checkpoint": str(expected_checkpoint),
        "splits": split_runs,
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

train_args=(
  "${PYTHON_BIN}" train_document_relation.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --config "${CONFIG}"
  --training_config "${TRAINING_CONFIG}"
  --entity_source predicted
  --predictions_dir "${PREDICTIONS_DIR}"
  --output_dir "${DOCUMENT_RE_DIR}"
)
[[ -n "${DEVICE}" ]] && train_args+=(--device "${DEVICE}")
[[ "${BFLOAT16}" == "1" ]] && train_args+=(--bfloat16)
[[ "${OFFLINE}" == "1" ]] && train_args+=(--offline)
if [[ -n "${DISTANCE_POSITIVE_LOSS_WEIGHT}" ]]; then
  train_args+=(--distance_positive_loss_weight "${DISTANCE_POSITIVE_LOSS_WEIGHT}")
fi
if [[ -n "${DISTANCE_LOSS_BOUNDARIES}" ]]; then
  read -r -a distance_boundaries <<< "${DISTANCE_LOSS_BOUNDARIES}"
  train_args+=(--distance_loss_boundaries "${distance_boundaries[@]}")
fi
if [[ -n "${DISTANCE_LOSS_WEIGHTS}" ]]; then
  read -r -a distance_weights <<< "${DISTANCE_LOSS_WEIGHTS}"
  train_args+=(--distance_loss_weights "${distance_weights[@]}")
fi
[[ "${DISTANCE_CONDITIONED_FUSION}" == "1" ]] && train_args+=(--distance_conditioned_fusion)
[[ -n "${FUSION_LONG_DISTANCE_BOUNDARY}" ]] && train_args+=(--fusion_long_distance_boundary "${FUSION_LONG_DISTANCE_BOUNDARY}")
[[ -n "${FUSION_DISTANCE_HIDDEN_SIZE}" ]] && train_args+=(--fusion_distance_hidden_size "${FUSION_DISTANCE_HIDDEN_SIZE}")
[[ -n "${FUSION_LONG_DISTANCE_BIAS}" ]] && train_args+=(--fusion_long_distance_bias "${FUSION_LONG_DISTANCE_BIAS}")
[[ "${LONG_DISTANCE_HEAD}" == "1" ]] && train_args+=(--long_distance_head)
[[ -n "${LONG_DISTANCE_BOUNDARY}" ]] && train_args+=(--long_distance_boundary "${LONG_DISTANCE_BOUNDARY}")
[[ -n "${LONG_DISTANCE_HEAD_WEIGHT}" ]] && train_args+=(--long_distance_head_weight "${LONG_DISTANCE_HEAD_WEIGHT}")
[[ -n "${LONG_DISTANCE_RANKING_LOSS_WEIGHT}" ]] && train_args+=(--long_distance_ranking_loss_weight "${LONG_DISTANCE_RANKING_LOSS_WEIGHT}")
[[ -n "${LONG_DISTANCE_RANKING_MARGIN}" ]] && train_args+=(--long_distance_ranking_margin "${LONG_DISTANCE_RANKING_MARGIN}")
[[ -n "${LONG_DISTANCE_RANKING_MAX_NEGATIVES}" ]] && train_args+=(--long_distance_ranking_max_negatives "${LONG_DISTANCE_RANKING_MAX_NEGATIVES}")
[[ "${PAIR_TYPE_COMPATIBILITY}" == "1" ]] && train_args+=(--pair_type_compatibility)
[[ -n "${PAIR_TYPE_COMPATIBILITY_LOSS_WEIGHT}" ]] && train_args+=(--pair_type_compatibility_loss_weight "${PAIR_TYPE_COMPATIBILITY_LOSS_WEIGHT}")
[[ -n "${PAIR_TYPE_COMPATIBILITY_BIAS_WEIGHT}" ]] && train_args+=(--pair_type_compatibility_bias_weight "${PAIR_TYPE_COMPATIBILITY_BIAS_WEIGHT}")
if [[ -n "${PAIR_TYPE_FOCUS_PAIRS}" ]]; then
  read -r -a focus_pairs <<< "${PAIR_TYPE_FOCUS_PAIRS}"
  train_args+=(--pair_type_focus_pairs "${focus_pairs[@]}")
fi
[[ -n "${PAIR_TYPE_FOCUS_WEIGHT}" ]] && train_args+=(--pair_type_focus_weight "${PAIR_TYPE_FOCUS_WEIGHT}")
[[ -n "${POSITIVE_FOCAL_LOSS_WEIGHT}" ]] && train_args+=(--positive_focal_loss_weight "${POSITIVE_FOCAL_LOSS_WEIGHT}")
[[ -n "${POSITIVE_FOCAL_GAMMA}" ]] && train_args+=(--positive_focal_gamma "${POSITIVE_FOCAL_GAMMA}")
[[ -n "${LONG_DISTANCE_SAMPLING_WEIGHT}" ]] && train_args+=(--long_distance_sampling_weight "${LONG_DISTANCE_SAMPLING_WEIGHT}")
[[ "${USE_DIRECTION_EMBEDDING}" == "1" ]] && train_args+=(--use_direction_embedding)
[[ "${USE_WINDOW_POSITION_ENCODING}" == "1" ]] && train_args+=(--use_window_position_encoding)
[[ -n "${FIXED_EVAL_THRESHOLD}" ]] && train_args+=(--fixed_eval_threshold "${FIXED_EVAL_THRESHOLD}")
[[ "${RECALL_HEAD}" == "1" ]] && train_args+=(--recall_head)
[[ -n "${RECALL_HEAD_SIZE}" ]] && train_args+=(--recall_head_size "${RECALL_HEAD_SIZE}")
[[ -n "${RECALL_HEAD_WEIGHT}" ]] && train_args+=(--recall_head_weight "${RECALL_HEAD_WEIGHT}")
[[ -n "${RECALL_MARGIN_LOSS_WEIGHT}" ]] && train_args+=(--recall_margin_loss_weight "${RECALL_MARGIN_LOSS_WEIGHT}")
[[ -n "${RECALL_THRESHOLD}" ]] && train_args+=(--recall_threshold "${RECALL_THRESHOLD}")
[[ -n "${RECALL_MARGIN}" ]] && train_args+=(--recall_margin "${RECALL_MARGIN}")
[[ -n "${RECALL_SHORT_DISTANCE_BOUNDARY}" ]] && train_args+=(--recall_short_distance_boundary "${RECALL_SHORT_DISTANCE_BOUNDARY}")
[[ -n "${RECALL_SHORT_DISTANCE_WEIGHT}" ]] && train_args+=(--recall_short_distance_weight "${RECALL_SHORT_DISTANCE_WEIGHT}")
[[ "${PAIR_LINK_HEAD}" == "1" ]] && train_args+=(--pair_link_head)
[[ -n "${PAIR_LINK_LOSS_WEIGHT}" ]] && train_args+=(--pair_link_loss_weight "${PAIR_LINK_LOSS_WEIGHT}")
[[ -n "${PAIR_LINK_BIAS}" ]] && train_args+=(--pair_link_bias "${PAIR_LINK_BIAS}")
[[ -n "${PAIR_LINK_GATE_WEIGHT}" ]] && train_args+=(--pair_link_gate_weight "${PAIR_LINK_GATE_WEIGHT}")
[[ -n "${PAIR_LINK_POSITIVE_WEIGHT}" ]] && train_args+=(--pair_link_positive_weight "${PAIR_LINK_POSITIVE_WEIGHT}")
train_args+=("$@")
printf 'Running:'
printf ' %q' "${train_args[@]}"
printf '\n'
"${train_args[@]}"

if [[ -n "${DISTANCE_POSITIVE_LOSS_WEIGHT}" ]]; then
  "${PYTHON_BIN}" - "${DOCUMENT_RE_DIR}/training_summary.json" "${DISTANCE_POSITIVE_LOSS_WEIGHT}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected = float(sys.argv[2])
if not summary_path.exists():
    raise SystemExit(f"Missing training summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
actual = float(summary.get("distance_positive_loss_weight", 0.0))
if abs(actual - expected) > 1e-9:
    raise SystemExit(
        f"Distance-loss weight mismatch: requested {expected}, summary records {actual}"
    )
print(f"Validated distance_positive_loss_weight={actual}")
PY
fi

eval_args=(
  "${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${LOCAL_CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_DIR}"
  --config "${CONFIG}"
  --split eval
  --batch_size "${BATCH_SIZE}"
  --ner_span_threshold "${NER_SPAN_THRESHOLD}"
  --ner_type_threshold "${NER_TYPE_THRESHOLD}"
  --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
  --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
  --review_margin "${EVAL_REVIEW_MARGIN}"
  --output_dir "${EVAL_OUTPUT_DIR}"
)
[[ -n "${EVAL_RELATION_THRESHOLD}" ]] && eval_args+=(--relation_threshold "${EVAL_RELATION_THRESHOLD}")
[[ -n "${DEVICE}" ]] && eval_args+=(--device "${DEVICE}")
[[ "${OFFLINE}" == "1" ]] && eval_args+=(--offline)
printf 'Running:'
printf ' %q' "${eval_args[@]}"
printf '\n'
"${eval_args[@]}"

echo "Learned-fusion eval metrics: ${EVAL_OUTPUT_DIR}/eval_metrics.json"
echo "Calibrate its relation threshold on eval before any test run."
