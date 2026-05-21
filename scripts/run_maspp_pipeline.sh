#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/jkp}"
OUT_ROOT="${2:-outputs/alphalife_mas_plus}"

python experiments/alphalife_mas_plus.py \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_ROOT}"

LATEST_RUN="$(ls -td "${OUT_ROOT}"/* | head -n 1)"

python experiments/alphalife_mas_plus_state_control.py \
  --data-root "${DATA_ROOT}" \
  --input-dir "${LATEST_RUN}"

echo "Completed MAS++ run: ${LATEST_RUN}"
