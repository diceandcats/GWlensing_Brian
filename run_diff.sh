#!/bin/bash
set -euo pipefail

ROW="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${CSV:?CSV must be set in the Condor submit file}"
: "${OUT_DIR:=$SCRIPT_DIR/out}"

# Make relative submit-file paths independent of the execute node's launch
# directory.
[[ "$CSV" = /* ]] || CSV="$SCRIPT_DIR/$CSV"
[[ "$OUT_DIR" = /* ]] || OUT_DIR="$SCRIPT_DIR/$OUT_DIR"

: "${SIMULATION_MODEL:=cats}"
: "${LOCALIZATION_MODEL:=keeton}"

: "${SIGMA_DT_FALLBACK:=0.1}"

: "${BURN_IN:=3000}"
: "${MAX_MAP_PIXELS:=1600}"

# Parallelism used by the underlying localization routines.
CONDOR_CPUS="${REQUEST_CPUS:-${PYTHON_CPU_COUNT:-${OMP_NUM_THREADS:-1}}}"

: "${DE_PROCS:=$CONDOR_CPUS}"
: "${MCMC_PROCS:=$CONDOR_CPUS}"

export OUT_DIR
export DE_PROCS
export MCMC_PROCS

# DE and MCMC use process pools.  Keep BLAS/OpenMP from creating another full
# set of threads in every worker process.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON_BIN="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "simulation_diff.py"
echo "Row                = $ROW"
echo "CSV                = $CSV"
echo "OUT_DIR            = $OUT_DIR"
echo "Simulation model   = $SIMULATION_MODEL"
echo "Localization model = $LOCALIZATION_MODEL"
echo "sigma_dt fallback  = $SIGMA_DT_FALLBACK"
echo "Burn-in            = $BURN_IN"
echo "Max map pixels     = $MAX_MAP_PIXELS"
echo "DE_PROCS           = $DE_PROCS"
echo "MCMC_PROCS         = $MCMC_PROCS"
echo "Python             = $PYTHON_BIN"
echo "============================================================"

exec "$PYTHON_BIN" -u "$SCRIPT_DIR/simulation_diff.py" \
    --csv "$CSV" \
    --row "$ROW" \
    --simulation-model "$SIMULATION_MODEL" \
    --localization-model "$LOCALIZATION_MODEL" \
    --sigma-dt-fallback "$SIGMA_DT_FALLBACK" \
    --burn-in "$BURN_IN" \
    --max-map-pixels "$MAX_MAP_PIXELS"
