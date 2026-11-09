#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_ind_v8_fixed_stage1.sh"
"${SCRIPT_DIR}/run_ind_v8_fixed_stage2.sh"
"${SCRIPT_DIR}/run_ind_v8_fixed_final_test.sh"
