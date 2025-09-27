#!/bin/bash
set -euo pipefail

# Activate environment if needed
# source /opt/miniconda/etc/profile.d/conda.sh
# conda activate lens

ROW="$1"
OUT_BASE="${OUT_DIR:-$PWD/out}"
export OUT_DIR="${OUT_BASE}/row${ROW}"
mkdir -p "$OUT_DIR" logs

# Map requested cores to the samplers (override in submit if you like)
: "${DE_PROCS:=${REQUEST_CPUS:-1}}"
: "${MCMC_PROCS:=${REQUEST_CPUS:-1}}"
export DE_PROCS MCMC_PROCS

echo "Row=$ROW OUT_DIR=$OUT_DIR DE_PROCS=$DE_PROCS MCMC_PROCS=$MCMC_PROCS"
exec python3 -u simulation.py --csv "$CSV" --row "$ROW"

: "${MCMC_BACKEND:=threads}"
export MCMC_BACKEND
# For single row testing
# set -euo pipefail
# ROW="$1"
# : "${CSV:?Set CSV in submit file environment or export it}"
# export OUT_DIR="${OUT_DIR:-$PWD/out/row${ROW}}"
# mkdir -p "$OUT_DIR" logs
# : "${DE_PROCS:=${REQUEST_CPUS:-1}}"
# : "${MCMC_PROCS:=${REQUEST_CPUS:-1}}"
# export DE_PROCS MCMC_PROCS
# exec python3 simulation.py --csv "$CSV" --row "$ROW"