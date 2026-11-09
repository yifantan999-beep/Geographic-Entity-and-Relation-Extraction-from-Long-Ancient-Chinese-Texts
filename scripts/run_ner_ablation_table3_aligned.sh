#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SUFFIX="${RUN_SUFFIX:-table3_aligned_v8}"
DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
TRANSFORMER="${TRANSFORMER:-local-bert/google-bert/bert-base-chinese}"

PRIMARY_CHECKPOINT="${PRIMARY_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_target}"
TRAIN_CONFIG_FILE="${TRAIN_CONFIG_FILE:-${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8.json}"
TEST_CONFIG="${TEST_CONFIG:-wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy}"
PRIMARY_METRICS="${PRIMARY_METRICS:-${PROJECT_ROOT}/artifacts/inference/ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_fixed_test/test_metrics.json}"

MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/ner_ablation_${RUN_SUFFIX}}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/artifacts/inference/ner_ablation_${RUN_SUFFIX}}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${PROJECT_ROOT}/artifacts/analysis/ner_ablation_${RUN_SUFFIX}}"
CONFIG_ROOT="${CONFIG_ROOT:-${PROJECT_ROOT}/cfg/wenyanwen}"
CONFIG_PREFIX="ner_ablation_${RUN_SUFFIX}_"

if [[ "${DEVICE}" != cuda:* ]]; then
  echo "This experiment requires a CUDA device; got DEVICE=${DEVICE}." >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to this Python environment; no ablation was started.")
print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
PY

required=(
  "${PRIMARY_CHECKPOINT}/config.json"
  "${PRIMARY_CHECKPOINT}/model.safetensors"
  "${TRAIN_CONFIG_FILE}"
  "${PRIMARY_METRICS}"
  "${PROJECT_ROOT}/cfg/${TEST_CONFIG}.json"
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 2; }
done

for path in "${MODEL_ROOT}" "${RESULT_ROOT}" "${ANALYSIS_ROOT}"; do
  [[ ! -e "${path}" ]] || {
    echo "Output already exists: ${path}" >&2
    echo "Set RUN_SUFFIX to a new value to create a separate run." >&2
    exit 2
  }
done

for tag in no_type_aux no_boundary_aux no_span_biaffine; do
  config_path="${CONFIG_ROOT}/${CONFIG_PREFIX}${tag}.json"
  [[ ! -e "${config_path}" ]] || {
    echo "Generated config already exists: ${config_path}" >&2
    echo "Set RUN_SUFFIX to a new value to create a separate run." >&2
    exit 2
  }
done

"${PYTHON_BIN}" - "${PROJECT_ROOT}" "${TRAIN_CONFIG_FILE}" "${TEST_CONFIG}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
train_config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
test_config_path = root / "cfg" / f"{sys.argv[3]}.json"
test_config = json.loads(test_config_path.read_text(encoding="utf-8"))
train_data = train_config["dataset"]["data_dir"]
test_data = test_config["dataset"]["data_dir"]
train_rows = json.loads((root / "datasets" / train_data / "test.json").read_text(encoding="utf-8"))
test_rows = json.loads((root / "datasets" / test_data / "test.json").read_text(encoding="utf-8"))

def digest(rows):
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

if digest(train_rows) != digest(test_rows):
    raise SystemExit(
        "The training config's test split differs from the Table 2 test split; "
        "refusing to produce non-comparable ablation metrics."
    )
print(f"Verified identical test split: {len(test_rows)} contexts ({digest(test_rows)[:12]}).")

dev_path = root / "datasets" / train_data / "dev.json"
if dev_path.is_file():
    dev_rows = json.loads(dev_path.read_text(encoding="utf-8"))
    if digest(dev_rows) == digest(test_rows):
        print(
            "WARNING: this diagnostic dataset reuses the same contexts for dev and test. "
            "These results are for aligned ablation comparison only, not an independent "
            "held-out test estimate."
        )
PY

echo "Full model checkpoint: ${PRIMARY_CHECKPOINT}"
echo "Training ablation configs from: ${TRAIN_CONFIG_FILE}"
echo "Test-time data config: ${TEST_CONFIG}"
echo "All NER decoding thresholds: span=0.80, type=0.50"

PYTHON_BIN="${PYTHON_BIN}" \
DEVICE="${DEVICE}" \
OFFLINE="${OFFLINE}" \
TRANSFORMER="${TRANSFORMER}" \
FULL_CONFIG="${TEST_CONFIG}" \
BASE_CONFIG_FILE="${TRAIN_CONFIG_FILE}" \
FULL_CHECKPOINT="${PRIMARY_CHECKPOINT}" \
INIT_CHECKPOINT="${PRIMARY_CHECKPOINT}" \
MODEL_ROOT="${MODEL_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}" \
CONFIG_ROOT="${CONFIG_ROOT}" \
ANALYSIS_ROOT="${ANALYSIS_ROOT}" \
ABLATION_CONFIG_PREFIX="${CONFIG_PREFIX}" \
RUN_BOUNDARY_AUX_ABLATION=0 \
SKIP_LEGACY_SUMMARY=1 \
NER_SPAN_THRESHOLD=0.80 \
NER_TYPE_THRESHOLD=0.50 \
NER_OVERLAP_STRATEGY=nms \
MAX_DOCUMENT_ENTITIES=128 \
BATCH_SIZE=1 \
EPOCHS=20 \
PATIENCE=6 \
GRADIENT_ACCUMULATION=8 \
RETRAIN_FULL=0 \
bash scripts/run_ner_ablation.sh

"${PYTHON_BIN}" scripts/summarize_ner_ablation_table3.py \
  --root "${RESULT_ROOT}" \
  --primary-metrics "${PRIMARY_METRICS}" \
  --output-dir "${ANALYSIS_ROOT}"

echo "Table 3 ablation results: ${ANALYSIS_ROOT}/table3_ner_ablation_rows.tex"
echo "Machine-readable results: ${ANALYSIS_ROOT}/table3_ner_ablation_results.csv"
