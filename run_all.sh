#!/usr/bin/env bash
#
# run_all.sh -- end-to-end reproducibility pipeline for the FAIR-PLR NSDUH paper.
#
# Prerequisites:
#   - Python 3.10+ with the packages in requirements.txt installed
#   - R with the glmnet package installed
#   - The NSDUH_DATA_DIR environment variable pointing at the directory
#     containing the raw NSDUH_<year>_Tab.txt/.tsv public-use files.
#
# Usage:
#   export NSDUH_DATA_DIR=/path/to/NSDUH_raw_files
#   ./run_all.sh
#
# Each step prints its status prefix [1/6], [2/6], ... so progress is visible.
# The script halts immediately on any step failure (set -e) so upstream errors
# are caught rather than silently propagated.
#
# This runs the modeling pipeline only. To render the manuscript figures from
# the resulting (or the shipped precomputed) results/, see the
# "Reproduce manuscript figures" section of README.md.

set -euo pipefail

if [[ -z "${NSDUH_DATA_DIR:-}" ]]; then
  echo "ERROR: NSDUH_DATA_DIR is not set. Export it to the path containing"
  echo "       NSDUH_<year>_Tab.txt/.tsv files before running this script."
  echo ""
  echo "  export NSDUH_DATA_DIR=/path/to/NSDUH_raw_files"
  echo "  ./run_all.sh"
  exit 2
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

export NSDUH_CLEAN_DIR="${NSDUH_CLEAN_DIR:-$SCRIPT_DIR/data_clean}"

mkdir -p "$NSDUH_CLEAN_DIR" \
         results/metrics results/coefficients results/predictions

echo "[1/6] Cleaning raw NSDUH files into $NSDUH_CLEAN_DIR ..."
python src/1_clean_data.py

echo "[2/6] Generating descriptive statistics tables (Tables 1-2, Supp. Table 1) ..."
jupyter nbconvert --to notebook --execute \
    src/2_generate_statistics_table.ipynb \
    --output 2_generate_statistics_table.executed.ipynb
jupyter nbconvert --to notebook --execute \
    src/2_generate_statistics_table_combined_predictor.ipynb \
    --output 2_generate_statistics_table_combined_predictor.executed.ipynb

echo "[3/6] Fitting Agnostic-PLR (single pooled elastic-net LR) ..."
python src/5_fit_agnostic_plr.py

echo "[4/6] Fitting Separate-PLR (one elastic-net LR per subgroup level) ..."
python src/6_fit_separate_plr.py --subgroup all
python src/6_fit_separate_plr.py --subgroup cross

echo "[5/6] Fitting FAIR-PLR (combined subgroup + survey weight) ..."
python src/7_fit_fair_plr.py --subgroup all
python src/7_fit_fair_plr.py --subgroup cross

echo "[6/6] Baselines + bootstrap CIs and DeLong tests (Table 3) ..."
python src/4_baselines_and_stats.py --subgroup all

echo ""
echo "Pipeline complete. Final artifacts:"
echo "  Cleaned per-year CSVs : $NSDUH_CLEAN_DIR/"
echo "  Evaluation tables     : results/metrics/"
echo "  Coefficients          : results/coefficients/"
echo "  Per-observation scores: results/predictions/"
echo ""
echo "Next: render figures with the scripts in README.md > Reproduce manuscript figures."
