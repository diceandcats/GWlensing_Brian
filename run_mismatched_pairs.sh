#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 MODEL CSV CLUSTER" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CSV="$2"
[[ "$CSV" = /* ]] || CSV="$SCRIPT_DIR/$CSV"

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out_mismatched_models}"
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$SCRIPT_DIR/$OUT_DIR"
export OUT_DIR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2

mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

exec /home/dices/.venv/bin/python -u simulation_mismatched_pairs.py \
    --simulation-model "$1" \
    --csv "$CSV" \
    --cluster "$3"
