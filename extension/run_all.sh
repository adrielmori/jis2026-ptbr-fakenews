#!/usr/bin/env bash
# Reproduce every artefact of the extended study from the published embeddings / hyper-parameters.
#
#   bash extension/run_all.sh            # full pipeline (creates extension/.venv if needed)
#   bash extension/run_all.sh --clean    # delete extension/results, figures, tables, logs first
#   PYTHON=/path/to/python bash extension/run_all.sh --no-venv   # use an existing interpreter
#
# Runs on Git Bash (Windows) and on Linux/macOS. CPU for the classifiers; the SVC-RBF Gram matrix uses the GPU when
# CUDA is available (identical results otherwise, only slower). See DECISIONS.md D-24.
set -euo pipefail

EXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXT_DIR"

CLEAN=0
USE_VENV=1
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --no-venv) USE_VENV=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ $CLEAN -eq 1 ]]; then
  echo "[run_all] cleaning previous outputs"
  rm -rf results figures tables logs
fi
mkdir -p results figures tables logs

if [[ $USE_VENV -eq 1 ]]; then
  if [[ ! -d .venv ]]; then
    BOOT="${PYTHON:-python}"
    echo "[run_all] creating virtual environment with $BOOT"
    "$BOOT" -m venv .venv
  fi
  if [[ -x .venv/Scripts/python.exe ]]; then PY=".venv/Scripts/python.exe"; else PY=".venv/bin/python"; fi
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements.txt
else
  PY="${PYTHON:-python}"
fi
echo "[run_all] python: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# Thread limits for BLAS inside the parallel workers (results do not depend on them).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

START=$(date +%s)
step() {
  local name="$1"; shift
  local t0=$(date +%s)
  echo "=============================================================="
  echo "[run_all] $(date '+%F %T') START $name"
  "$@" 2>&1 | tee "logs/${name}.stdout"
  echo "[run_all] $(date '+%F %T') DONE  $name in $(( $(date +%s) - t0 )) s"
}

step check_integrity   "$PY" check_integrity.py
step reproduce_table4  "$PY" reproduce_table4.py
step run_lodo          "$PY" run_lodo.py
step run_calibration   "$PY" run_calibration.py
step run_errors        "$PY" run_errors.py
step make_tables       "$PY" make_tables.py
step make_figures      "$PY" make_figures.py
step make_baseline_figures "$PY" make_baseline_figures.py

echo "=============================================================="
echo "[run_all] all steps finished in $(( $(date +%s) - START )) s"
echo "[run_all] results: extension/results | tables: extension/tables | figures: extension/figures | logs: extension/logs"
