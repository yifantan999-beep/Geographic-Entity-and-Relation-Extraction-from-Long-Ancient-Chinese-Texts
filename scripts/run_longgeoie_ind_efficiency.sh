#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible parameter-count and inference-efficiency benchmark for the
# paper's LongGeoIE (v8 fixed) model on the original IND test split.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda:0}"
OFFLINE="${OFFLINE:-1}"
CONFIG="${CONFIG:-wenyanwen/wenyanwen_ind_v8_fixed}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/models/wenyanwen_ind_v8_fixed/bert-base-chinese/run_01}"
DOCUMENT_RE_DIR="${DOCUMENT_RE_DIR:-${CHECKPOINT}/document_relation_distance_experts_v8}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${PROJECT_ROOT}/artifacts/analysis/longgeoie_ind_efficiency}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_DIR}/inference_output}"
BENCHMARK_JSON="${BENCHMARK_JSON:-${BENCHMARK_DIR}/benchmark.json}"

NER_SPAN_THRESHOLD="${NER_SPAN_THRESHOLD:-0.80}"
NER_TYPE_THRESHOLD="${NER_TYPE_THRESHOLD:-0.50}"
NER_OVERLAP_STRATEGY="${NER_OVERLAP_STRATEGY:-nms}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-0.24}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_DOCUMENT_ENTITIES="${MAX_DOCUMENT_ENTITIES:-128}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
REPEATS="${REPEATS:-3}"

for path in \
  "${CHECKPOINT}/config.json" \
  "${DOCUMENT_RE_DIR}/document_relation_config.json" \
  "${DOCUMENT_RE_DIR}/document_relation.safetensors"; do
  [[ -f "${path}" ]] || { echo "Required file does not exist: ${path}" >&2; exit 2; }
done

mkdir -p "${BENCHMARK_DIR}"

args=("${PYTHON_BIN}" infer_long_context.py
  --checkpoint "${CHECKPOINT}"
  --document_re_checkpoint "${DOCUMENT_RE_DIR}"
  --config "${CONFIG}"
  --split test
  --batch_size "${BATCH_SIZE}"
  --ner_span_threshold "${NER_SPAN_THRESHOLD}"
  --ner_type_threshold "${NER_TYPE_THRESHOLD}"
  --ner_overlap_strategy "${NER_OVERLAP_STRATEGY}"
  --relation_threshold "${RELATION_THRESHOLD}"
  --review_margin 0.0
  --max_document_entities "${MAX_DOCUMENT_ENTITIES}"
  --benchmark_output "${BENCHMARK_JSON}"
  --benchmark_warmup_runs "${WARMUP_RUNS}"
  --benchmark_repeats "${REPEATS}"
  --output_dir "${OUTPUT_DIR}")
[[ -n "${DEVICE}" ]] && args+=(--device "${DEVICE}")
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)

printf 'Running LongGeoIE IND efficiency benchmark:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"

"${PYTHON_BIN}" scripts/render_efficiency_benchmark.py \
  "${BENCHMARK_JSON}" \
  --output_dir "${BENCHMARK_DIR}"

printf 'Benchmark JSON: %s\n' "${BENCHMARK_JSON}"
printf 'Paper table: %s\n' "${BENCHMARK_DIR}/benchmark_table.tex"
