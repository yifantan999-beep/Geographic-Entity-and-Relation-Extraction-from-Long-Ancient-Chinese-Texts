#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCAL_CHECKPOINT="${LOCAL_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux/bert-base-chinese/run_01}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/long_context_type_aux_decoder}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${LOCAL_CHECKPOINT}/document_relation_predicted_long}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/inference/long_context_type_aux_stage2_eval_default}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation.json}"

NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.5}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.3}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DEVICE="${DEVICE:-}"
OFFLINE="${OFFLINE:-1}"
BFLOAT16="${BFLOAT16:-0}"

if [[ ! -f "${LOCAL_CHECKPOINT}/config.json" ]]; then
  echo "Type-aux checkpoint is missing config.json: ${LOCAL_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${LOCAL_CHECKPOINT}/model.safetensors" && ! -f "${LOCAL_CHECKPOINT}/model.safetensors.index.json" ]]; then
  echo "Type-aux checkpoint weights were not found: ${LOCAL_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${TRAINING_CONFIG}" ]]; then
  echo "Document RE training config does not exist: ${TRAINING_CONFIG}" >&2
  exit 2
fi

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

"${PYTHON_BIN}" - "${PREDICTIONS_DIR}" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
for split in ("train", "eval"):
    path = directory / f"{split}_predictions.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing prediction file: {path}")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    document_ids = {int(record["document_id"]) for record in records}
    expected = len(records)
    expected_ids = set(range(expected))
    if not records or document_ids != expected_ids:
        raise SystemExit(
            f"{path}: expected {expected} documents with ids 0..{expected - 1}, "
            f"found {len(records)} rows and {len(document_ids)} unique ids"
        )
    print(f"Validated {split}: {len(records)} long-context documents")
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
train_args+=("$@")

printf 'Running:'
printf ' %q' "${train_args[@]}"
printf '\n'
"${train_args[@]}"

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
  --output_dir "${EVAL_OUTPUT_DIR}"
)
[[ -n "${DEVICE}" ]] && eval_args+=(--device "${DEVICE}")
[[ "${OFFLINE}" == "1" ]] && eval_args+=(--offline)

printf 'Running:'
printf ' %q' "${eval_args[@]}"
printf '\n'
"${eval_args[@]}"

echo "Stage 2 eval metrics: ${EVAL_OUTPUT_DIR}/eval_metrics.json"
echo "Do not run the Stage 2 test split until relation calibration is fixed on eval."
