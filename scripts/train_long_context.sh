#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -z "${TRANSFORMER:-}" ]]; then
  if [[ -f "${PROJECT_ROOT}/local-bert/google-bert/bert-base-chinese/config.json" ]]; then
    TRANSFORMER="${PROJECT_ROOT}/local-bert/google-bert/bert-base-chinese"
  elif [[ -f "${PROJECT_ROOT}/local-bert/bert-base-chinese/config.json" ]]; then
    TRANSFORMER="${PROJECT_ROOT}/local-bert/bert-base-chinese"
  elif [[ -f "${PROJECT_ROOT}/local-bert/config.json" ]]; then
    TRANSFORMER="${PROJECT_ROOT}/local-bert"
  elif [[ -d "${PROJECT_ROOT}/pretrained_models/bert-base-chinese" ]]; then
    TRANSFORMER="${PROJECT_ROOT}/pretrained_models/bert-base-chinese"
  elif [[ -d "${PROJECT_ROOT}/models/pretrained/bert-base-chinese" ]]; then
    TRANSFORMER="${PROJECT_ROOT}/models/pretrained/bert-base-chinese"
  else
    TRANSFORMER="google-bert/bert-base-chinese"
  fi
fi
CONFIG="${CONFIG:-wenyanwen/wenyanwen_long_context_512}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/models/wenyanwen_long_context_stage1/bert-base-chinese/run_01}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-0}"
REBUILD_CACHE="${REBUILD_CACHE:-1}"
BFLOAT16="${BFLOAT16:-0}"
OFFLINE="${OFFLINE:-auto}"
SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE:-0}"
LOAD_CKPT="${LOAD_CKPT:-}"

if "${PYTHON_BIN}" scripts/check_transformer.py "${TRANSFORMER}" --quiet; then
  [[ "${OFFLINE}" == "auto" ]] && OFFLINE="1"
elif [[ -d "${TRANSFORMER}" || "${OFFLINE}" == "1" ]]; then
  cat >&2 <<EOF
The local transformer is incomplete: ${TRANSFORMER}

It must contain config.json, tokenizer files, and model weights. Point TRANSFORMER
to a complete local directory, for example:

  TRANSFORMER=/data/pretrained/bert-base-chinese OFFLINE=1 bash scripts/train_long_context.sh
EOF
  exit 2
elif [[ "${OFFLINE}" == "auto" ]]; then
  HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"
  HF_BASE="${HF_BASE%/}"
  if command -v curl >/dev/null 2>&1 && ! curl -fsSIL --connect-timeout 5 --max-time 10 \
      "${HF_BASE}/${TRANSFORMER}/resolve/main/config.json" >/dev/null; then
    cat >&2 <<EOF
The transformer '${TRANSFORMER}' is not cached and huggingface.co is unreachable.

Download it on a networked machine, or from a configured mirror:

  huggingface-cli download ${TRANSFORMER} \
    --local-dir /data/pretrained/bert-base-chinese

Then run:

  TRANSFORMER=/data/pretrained/bert-base-chinese \
  OFFLINE=1 \
  bash scripts/train_long_context.sh

If you use a Hugging Face mirror, set HF_ENDPOINT before running this script.
EOF
    exit 2
  fi
  OFFLINE="0"
fi

args=(
  "${PYTHON_BIN}" train_long_context.py
  --transformer "${TRANSFORMER}"
  --config "${CONFIG}"
  --model_path "${MODEL_DIR}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
)

[[ "${REBUILD_CACHE}" == "1" ]] && args+=(--rebuild_cache)
[[ "${BFLOAT16}" == "1" ]] && args+=(--bfloat16)
[[ "${OFFLINE}" == "1" ]] && args+=(--offline)
[[ "${SAVE_OPTIMIZER_STATE}" == "1" ]] && args+=(--save_optimizer_state)
[[ -n "${LOAD_CKPT}" ]] && args+=(--load_ckpt "${LOAD_CKPT}")
args+=("$@")

printf 'Running:'
printf ' %q' "${args[@]}"
printf '\n'
"${args[@]}"
