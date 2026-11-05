#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 ROW" >&2
    exit 2
fi

ROW="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${CSV:?CSV must be set in the Condor submit file}"
: "${OUT_DIR:=$SCRIPT_DIR/out_marginalize_models}"

[[ "$CSV" = /* ]] || CSV="$SCRIPT_DIR/$CSV"
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$SCRIPT_DIR/$OUT_DIR"

# Parallelism used by DE and MCMC
CONDOR_CPUS="${REQUEST_CPUS:-${PYTHON_CPU_COUNT:-${OMP_NUM_THREADS:-1}}}"
: "${DE_PROCS:=$CONDOR_CPUS}"
: "${MCMC_PROCS:=$CONDOR_CPUS}"

export OUT_DIR
export DE_PROCS
export MCMC_PROCS

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "simulation_marginalize_models.py"
echo "Row        = $ROW"
echo "CSV        = $CSV"
echo "OUT_DIR    = $OUT_DIR"
echo "DE_PROCS   = $DE_PROCS"
echo "MCMC_PROCS = $MCMC_PROCS"
echo "Python     = $PYTHON_BIN"
echo "============================================================"

exec "$PYTHON_BIN" -u "$SCRIPT_DIR/simulation_marginalize_models.py" \
    --csv "$CSV" \
    --row "$ROW"
