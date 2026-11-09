#!/usr/bin/env bash
set -Eeuo pipefail

# Isolated Stage-2 RE ablation. Stage-1 predictions are generated once and
# shared by every variant. Test uses fixed thresholds and is never calibrated.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_fixed_boundary_focus}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_fixed_boundary_focus}"
BASE_DOCUMENT_RE="${BASE_DOCUMENT_RE:-${LOCAL_CHECKPOINT}/document_relation_fixed_balanced}"
BASE_TRAINING_CONFIG="${BASE_TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_distance_experts_v8_fixed_balanced.json}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/diagnostic_easy_re_ablation_v8_fixed_scores}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${LOCAL_CHECKPOINT}/document_relation_re_ablation_v8_fixed}"
CONFIG_ROOT="${CONFIG_ROOT:-${PROJECT_ROOT}/cfg/wenyanwen/re_ablation_v8_fixed}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-}"
NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.80}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.50}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-0.24}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-128}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-5}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_TEST="${FORCE_TEST:-0}"

# OUTPUT_ROOT is a convenient run-isolation alias.  It intentionally wins over
# a stale RESULT_ROOT exported by a previous experiment; RESULT_ROOT remains
# available when no OUTPUT_ROOT is supplied.
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  RESULT_ROOT="${OUTPUT_ROOT}"
  ANALYSIS_ROOT="${OUTPUT_ROOT%/}_analysis"
elif [[ -z "${RESULT_ROOT:-}" ]]; then
  RESULT_ROOT="${PROJECT_ROOT}/artifacts/inference/re_ablation_v8_fixed_test"
fi
if [[ -z "${ANALYSIS_ROOT:-}" ]]; then
  ANALYSIS_ROOT="${PROJECT_ROOT}/artifacts/analysis/re_ablation_v8_fixed"
fi

echo "RE ablation result root: ${RESULT_ROOT}"
echo "RE ablation analysis root: ${ANALYSIS_ROOT}"

metrics_match_checkpoint() {
  local metrics_path="$1"
  local expected_checkpoint="$2"
  [[ -f "${metrics_path}" ]] || return 1
  "${PYTHON_BIN}" - "${metrics_path}" "${expected_checkpoint}" "${RELATION_THRESHOLD}" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics_path, checkpoint, expected_threshold = sys.argv[1:]
try:
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    run = payload.get("run", {})
    actual_checkpoint = run.get("document_re_checkpoint")
    actual_threshold = run.get("relation_threshold")
    if not actual_checkpoint or Path(actual_checkpoint).resolve() != Path(checkpoint).resolve():
        raise SystemExit(1)
    if actual_threshold is None or not math.isclose(float(actual_threshold), float(expected_threshold), rel_tol=0.0, abs_tol=1e-8):
        raise SystemExit(1)
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

for path in "${LOCAL_CHECKPOINT}/config.json" "${BASE_DOCUMENT_RE}/document_relation.safetensors" "${BASE_TRAINING_CONFIG}"; do
  [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 2; }
done
mkdir -p "${PREDICTIONS_DIR}" "${CHECKPOINT_ROOT}" "${RESULT_ROOT}" "${CONFIG_ROOT}" "${ANALYSIS_ROOT}"

# Build shared train/eval predicted entities and exported local RE scores.
for split in train eval; do
  if [[ ! -f "${PREDICTIONS_DIR}/${split}_predictions.jsonl" ]]; then
    args=("${PYTHON_BIN}" infer_long_context.py
      --checkpoint "${LOCAL_CHECKPOINT}" --config "${CONFIG}" --split "${split}"
      --batch_size "${BATCH_SIZE}" --ner_span_threshold "${NER_SPAN_THRESHOLD}"
      --ner_type_threshold "${NER_TYPE_THRESHOLD}" --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
      --max_document_entities "${MAX_DOCUMENT_ENTITIES}" --export_local_relation_scores
      --rebuild_cache --output_dir "${PREDICTIONS_DIR}")
    [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
    [[ "${OFFLINE}" == "1" ]] && args+=(--offline)
    printf 'Generating shared %s predictions:' "${split}"; printf ' %q' "${args[@]}"; printf '\n'
    "${args[@]}"
  else
    echo "Reusing shared predictions: ${PREDICTIONS_DIR}/${split}_predictions.jsonl"
  fi
done

"${PYTHON_BIN}" - "${BASE_TRAINING_CONFIG}" "${CONFIG_ROOT}" <<'PY'
import copy, json, sys
from pathlib import Path
base_path, out_root = Path(sys.argv[1]), Path(sys.argv[2])
base = json.loads(base_path.read_text(encoding="utf-8"))
variants = {
    "full": {},
    "no_learned_fusion": {"fusion_mode": "fixed"},
    "no_distance_gate": {"distance_conditioned_fusion": False},
    "no_long_distance_head": {"long_distance_head": False, "long_distance_ranking_loss_weight": 0.0},
    "no_distance_experts": {"distance_experts": False, "distance_expert_loss_weight": 0.0},
    "no_pair_type": {"pair_type_compatibility": False, "pair_type_compatibility_loss_weight": 0.0},
    "no_direction": {"use_direction_embedding": False},
    "no_window_position": {"use_window_position_encoding": False},
    "no_long_distance": {
        "distance_conditioned_fusion": False,
        "long_distance_head": False,
        "long_distance_ranking_loss_weight": 0.0,
        "distance_experts": False,
        "distance_expert_loss_weight": 0.0,
        "long_distance_sampling_weight": 0.0,
    },
}
for tag, overrides in variants.items():
    payload = copy.deepcopy(base)
    payload.update(overrides)
    payload["fusion_mode"] = payload.get("fusion_mode", "learned")
    if tag == "no_long_distance":
        payload["fusion_mode"] = "fixed"
    payload["fixed_eval_threshold"] = 0.24
    payload["epochs"] = 15
    payload["patience"] = 5
    out = out_root / f"{tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)
PY

run_variant() {
  local tag="$1"
  local config_file="${CONFIG_ROOT}/${tag}.json"
  local checkpoint="${CHECKPOINT_ROOT}/${tag}"
  local eval_dir="${RESULT_ROOT}/${tag}_eval"
  local test_dir="${RESULT_ROOT}/${tag}"

  if [[ "${FORCE_TRAIN}" == "1" || ! -f "${checkpoint}/document_relation.safetensors" ]]; then
    args=("${PYTHON_BIN}" train_document_relation.py
      --checkpoint "${LOCAL_CHECKPOINT}" --config "${CONFIG}"
      --training_config "${config_file}" --entity_source predicted
      --predictions_dir "${PREDICTIONS_DIR}" --output_dir "${checkpoint}"
      --load_document_checkpoint "${BASE_DOCUMENT_RE}"
      --epochs "${EPOCHS}" --patience "${PATIENCE}" --fixed_eval_threshold "${RELATION_THRESHOLD}")
    [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
    [[ "${OFFLINE}" == "1" ]] && args+=(--offline)
    [[ "${DEVICE}" == cuda:* ]] && args+=(--bfloat16)
    printf 'Training %s:' "${tag}"; printf ' %q' "${args[@]}"; printf '\n'
    "${args[@]}"
  else
    echo "Reusing RE checkpoint: ${checkpoint}"
  fi

  if [[ "${FORCE_TEST}" == "1" ]] || ! metrics_match_checkpoint "${test_dir}/test_metrics.json" "${checkpoint}"; then
    args=("${PYTHON_BIN}" infer_long_context.py
      --checkpoint "${LOCAL_CHECKPOINT}" --document_re_checkpoint "${checkpoint}"
      --config "${CONFIG}" --split test --batch_size "${BATCH_SIZE}"
      --ner_span_threshold "${NER_SPAN_THRESHOLD}" --ner_type_threshold "${NER_TYPE_THRESHOLD}"
      --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}" --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
      --relation_threshold "${RELATION_THRESHOLD}" --review_margin 0.0
      --output_dir "${test_dir}")
    [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
    [[ "${OFFLINE}" == "1" ]] && args+=(--offline)
    printf 'Testing %s:' "${tag}"; printf ' %q' "${args[@]}"; printf '\n'
    "${args[@]}"
  else
    echo "Reusing matching test metrics: ${test_dir}/test_metrics.json"
  fi
}

for tag in full no_learned_fusion no_distance_gate no_long_distance_head no_distance_experts no_pair_type no_direction no_window_position no_long_distance; do
  run_variant "${tag}"
done

"${PYTHON_BIN}" scripts/summarize_re_ablation.py \
  --root "${RESULT_ROOT}" \
  --expected-threshold "${RELATION_THRESHOLD}" \
  --output-csv "${ANALYSIS_ROOT}/re_ablation_results.csv" \
  --output-md "${ANALYSIS_ROOT}/re_ablation_results.md"
echo "RE ablation metrics: ${ANALYSIS_ROOT}/re_ablation_results.csv"
echo "RE ablation table: ${ANALYSIS_ROOT}/re_ablation_results.md"
