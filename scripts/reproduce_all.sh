#!/usr/bin/env bash
# End-to-end reproduction script for Agri-Drift

set -e

echo "=== [1/4] Running automated unit and integration tests ==="
PYTHONPATH=src .venv/bin/pytest tests/

echo "=== [2/4] Generating fault-injection visualisations ==="
MPLCONFIGDIR=/tmp/matplotlib_cache PYTHONPATH=src .venv/bin/python scripts/generate_fault_figures.py

echo "=== [3/4] Running multi-seed comparative benchmark experiments ==="
MPLCONFIGDIR=/tmp/matplotlib_cache PYTHONPATH=src .venv/bin/python scripts/run_benchmarks.py

echo "=== [4/4] Verifying Streamlit Dashboard dependencies ==="
.venv/bin/streamlit --version

echo "=========================================================="
echo "Agri-Drift reproduction pipeline completed successfully!"
echo "Metrics saved in: results/metrics/"
echo "Figures saved in: results/figures/"
echo "To launch the interactive dashboard:"
echo "  .venv/bin/streamlit run src/agri_drift/dashboard/app.py"
echo "=========================================================="

