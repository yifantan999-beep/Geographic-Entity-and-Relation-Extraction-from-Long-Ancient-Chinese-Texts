#!/usr/bin/env bash
set -Eeuo pipefail

# Train/evaluate the isolated NER ablations and create one result table.
# The test split is evaluated with fixed thresholds; no test-time calibration
# is performed here.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
TRANSFORMER="${TRANSFORMER:-local-bert/google-bert/bert-base-chinese}"

# This is the current Full v8 NER configuration/checkpoint.  Override these
# variables when running the same ablation protocol on another dataset.
FULL_CONFIG="${FULL_CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_fixed_boundary_focus}"
BASE_CONFIG_FILE="${BASE_CONFIG_FILE:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_fixed_boundary_focus.json}"
FULL_CHECKPOINT="${FULL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_fixed_boundary_focus}"
# For a strict comparison, point this to the same pre-ablation checkpoint for
# Full and every variant. By default the existing Full checkpoint is reused
# for a quick run; set RETRAIN_FULL=1 and override INIT_CHECKPOINT for paper
# grade training from a common pre-trained/base checkpoint.
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${FULL_CHECKPOINT}}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/artifacts/inference/ner_ablation_v8_fixed_test}"
CONFIG_ROOT="${CONFIG_ROOT:-${PROJECT_ROOT}/cfg/wenyanwen}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${PROJECT_ROOT}/artifacts/analysis/ner_ablation_v8_fixed}"
ABLATION_CONFIG_PREFIX="${ABLATION_CONFIG_PREFIX:-ner_ablation_}"
RUN_BOUNDARY_AUX_ABLATION="${RUN_BOUNDARY_AUX_ABLATION:-1}"
SKIP_LEGACY_SUMMARY="${SKIP_LEGACY_SUMMARY:-0}"

NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.80}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.50}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-6}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-8}"
RETRAIN_FULL="${RETRAIN_FULL:-0}"

required=(
  "${BASE_CONFIG_FILE}"
  "${FULL_CHECKPOINT}/config.json"
  "${FULL_CHECKPOINT}/model.safetensors"
  "${INIT_CHECKPOINT}/config.json"
  "${INIT_CHECKPOINT}/model.safetensors"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 2; }
done

mkdir -p "${CONFIG_ROOT}" "${RESULT_ROOT}" "${ANALYSIS_ROOT}"

"${PYTHON_BIN}" - "${BASE_CONFIG_FILE}" "${CONFIG_ROOT}" "${ABLATION_CONFIG_PREFIX}" <<'PY'
import copy
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
config_root = Path(sys.argv[2])
config_prefix = sys.argv[3]
base = json.loads(base_path.read_text(encoding="utf-8"))

variants = {
    "no_type_aux": {
        "ner_type_loss_weight": 0.0,
        "ner_type_decode_weight": 0.0,
        "ner_type_prototype_loss_weight": 0.0,
        "ner_type_prototype_decode_weight": 0.0,
        "ner_type_focus_weight": 0.0,
        "ner_type_focus_types": [],
        "ner_type_confusion_pairs": [],
    },
    "no_boundary_aux": {
        "ner_boundary_loss_weight": 0.0,
        "ner_boundary_consistency_loss_weight": 0.0,
        "ner_boundary_decode_weight": 0.0,
        "ner_boundary_start_candidate_fusion": False,
        "ner_boundary_hard_negative_weight": 0.0,
        "ner_boundary_adjacent_negative_weight": 0.0,
        "ner_boundary_internal_negative_weight": 0.0,
    },
    "no_span_biaffine": {
        "ner_span_biaffine": False,
        "ner_span_width_embedding": False,
        "ner_span_learned_gate": False,
        "ner_span_biaffine_weight": 0.0,
    },
}

for tag, overrides in variants.items():
    payload = copy.deepcopy(base)
    dataset = payload.setdefault("dataset", {})
    dataset.update(overrides)
    dataset["name"] = f"{dataset.get('name', 'ner_ablation')}_{config_prefix}{tag}"
    training = payload.setdefault("training", {})
    training.update({
        "num_epochs": 20,
        "patience": 6,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "optimize_for": "ner",
    })
    out = config_root / f"{config_prefix}{tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)
PY

run_ner_test() {
  local tag="$1"
  local config_name="$2"
  local checkpoint="$3"
  local overlap="$4"
  local output_dir="${RESULT_ROOT}/${tag}"

  if [[ -f "${output_dir}/test_metrics.json" ]]; then
    echo "Reusing existing metrics: ${output_dir}/test_metrics.json"
    return
  fi
  [[ -f "${checkpoint}/config.json" && -f "${checkpoint}/model.safetensors" ]] || {
    echo "Missing checkpoint for ${tag}: ${checkpoint}/config.json or model.safetensors" >&2
    exit 2
  }
  mkdir -p "${output_dir}"
  args=(
    "${PYTHON_BIN}" infer_long_context.py
    --checkpoint "${checkpoint}"
    --config "${config_name}"
    --split test
    --batch_size "${BATCH_SIZE}"
    --ner_span_threshold "${NER_SPAN_THRESHOLD}"
    --ner_type_threshold "${NER_TYPE_THRESHOLD}"
    --ner_overlap_strategy "${overlap}"
    --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
    --rebuild_cache
    --output_dir "${output_dir}"
  )
  [[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
  [[ "${OFFLINE}" == "1" ]] && args+=(--offline)
  printf 'Running %s:' "${tag}"
  printf ' %q' "${args[@]}"
  printf '\n'
  "${args[@]}"
}

train_stage1() {
  local tag="$1"
  local config_name="$2"
  local checkpoint="$3"
  local load_checkpoint="$4"
  if [[ ! -f "${checkpoint}/config.json" || ! -f "${checkpoint}/model.safetensors" ]]; then
    args=(
      "${PYTHON_BIN}" train_long_context.py
      --transformer "${TRANSFORMER}"
      --config "${config_name}"
      --model_path "${checkpoint}"
      --load_ckpt "${load_checkpoint}"
      --epochs "${EPOCHS}"
      --patience "${PATIENCE}"
      --batch_size "${BATCH_SIZE}"
      --gradient_accumulation "${GRADIENT_ACCUMULATION}"
      --seed 42
      --rebuild_cache
      --skip_final_test
    )
    [[ "${OFFLINE}" == "1" ]] && args+=(--offline)
    [[ "${DEVICE}" == cuda:* ]] && args+=(--bfloat16)
    printf 'Training %s:' "${tag}"
    printf ' %q' "${args[@]}"
    printf '\n'
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" "${args[@]}"
  else
    echo "Reusing checkpoint: ${checkpoint}"
  fi
}

run_ner_train() {
  local tag="$1"
  local config_name="wenyanwen/${ABLATION_CONFIG_PREFIX}${tag}"
  local checkpoint="${MODEL_ROOT}/run_${ABLATION_CONFIG_PREFIX}${tag}"
  train_stage1 "${tag}" "${config_name}" "${checkpoint}" "${INIT_CHECKPOINT}"
  run_ner_test "${tag}" "${config_name}" "${checkpoint}" "${NER_OVERLAP_STRATEGY}"
}

# Full checkpoint under the standard NMS decoding.  Set RETRAIN_FULL=1 for a
# strict paper ablation in which Full and every variant share INIT_CHECKPOINT.
FULL_EVAL_CHECKPOINT="${FULL_CHECKPOINT}"
if [[ "${RETRAIN_FULL}" == "1" ]]; then
  FULL_EVAL_CHECKPOINT="${MODEL_ROOT}/run_ner_ablation_full"
  train_stage1 "full" "${FULL_CONFIG}" "${FULL_EVAL_CHECKPOINT}" "${INIT_CHECKPOINT}"
fi
run_ner_test full "${FULL_CONFIG}" "${FULL_EVAL_CHECKPOINT}" "${NER_OVERLAP_STRATEGY}"

# One variable is disabled per newly trained variant.
run_ner_train no_type_aux
if [[ "${RUN_BOUNDARY_AUX_ABLATION}" == "1" ]]; then
  run_ner_train no_boundary_aux
fi
run_ner_train no_span_biaffine

# NMS is a decoder-only ablation, so it reuses the exact Full checkpoint.
run_ner_test no_nms "${FULL_CONFIG}" "${FULL_EVAL_CHECKPOINT}" none

if [[ "${SKIP_LEGACY_SUMMARY}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/summarize_ner_ablation.py \
    --root "${RESULT_ROOT}" \
    --output-csv "${ANALYSIS_ROOT}/ner_ablation_results.csv" \
    --output-md "${ANALYSIS_ROOT}/ner_ablation_results.md"
fi

echo "NER ablation metrics are ready:"
echo "  ${ANALYSIS_ROOT}/ner_ablation_results.csv"
echo "  ${ANALYSIS_ROOT}/ner_ablation_results.md"
