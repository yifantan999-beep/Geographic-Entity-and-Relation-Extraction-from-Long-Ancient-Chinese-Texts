#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Controlled follow-up to v8 recall.  It keeps the same Stage-1 checkpoint and
# dataset, while using the conservative Stage-2 configuration in which the
# >=128 expert is left to the global/long-distance heads.
export CONFIG="wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8_recall"
# This wrapper is a named ablation: do not inherit a stale TRAINING_CONFIG
# from an earlier shell session, or the saved checkpoint silently uses a
# different architecture than the experiment label claims.
export TRAINING_CONFIG="${PROJECT_ROOT}/cfg/wenyanwen/wenyanwen_document_relation_distance_experts_v8_recall_tuned.json"
export OLD_STAGE1="${OLD_STAGE1:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_target}"
export OLD_DOCUMENT_RE="${OLD_DOCUMENT_RE:-${OLD_STAGE1}/document_relation_distance_experts_v8}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced/bert-base-chinese/run_ner_diagnostic_easy_v8_recall_40ep}"
export DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${STAGE1_CHECKPOINT}/document_relation_distance_experts_v8_recall_tuned}"
export PREDICTIONS_DIR="${PREDICTIONS_DIR:-${PROJECT_ROOT}/artifacts/predicted_entities/diagnostic_easy_v8_recall_tuned_scores}"
export STAGE1_TEST_DIR="${STAGE1_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_recall_tuned_stage1_test}"
export STAGE2_EVAL_DIR="${STAGE2_EVAL_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_recall_tuned_stage2_eval}"
export FINAL_TEST_DIR="${FINAL_TEST_DIR:-${PROJECT_ROOT}/artifacts/inference/diagnostic_easy_v8_recall_tuned_final_test}"
export RECALL_HEAD_WEIGHT=0.10
export RECALL_MARGIN_LOSS_WEIGHT=0.15
export RECALL_SHORT_DISTANCE_WEIGHT=1.25
export EVAL_REVIEW_MARGIN=0.0

bash "${SCRIPT_DIR}/run_primary_diagnostic_easy_v8_recall.sh" "$@"

"${PYTHON_BIN:-python3}" - "${DOCUMENT_RE_DIR}/document_relation_config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "relation_adapter": False,
    "distance_expert_active_buckets": [0, 1, 2],
    "distance_expert_weight": 0.20,
    "long_distance_head_weight": 0.45,
    "recall_head_weight": 0.10,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"{path}: tuned configuration mismatch for {key}: "
            f"{payload.get(key)!r} != {value!r}"
        )
print(f"Validated tuned Stage-2 architecture: {path}")
PY
