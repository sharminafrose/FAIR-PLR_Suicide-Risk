# FAIR-PLR: Suicide Risk Prediction on NSDUH data (2013–2023)

Source code, derived results, and figures for the paper

> **Leveraging Functionally Adaptive Regularized Regression for Interpretable Suicide Risk Prediction and Subgroup Analysis**
> Afrose S., Kazanis W. H., Trafton J., Kapadia A., Peluso A.
> *npj Mental Health Research* (2026).
> [https://doi.org/10.1038/s44184-026-00229-y](https://doi.org/10.1038/s44184-026-00229-y)

The FAIR-PLR framework fits a single penalized logistic regression in
which every covariate interacts with a subgroup indicator and each
observation is weighted to equalize each subgroup's aggregate loss
contribution. It extends the linear-regression FAIR framework to binary
outcomes with a shared elastic-net penalty, and introduces a combined
subgroup-plus-survey weight so the NSDUH complex survey design can be
incorporated without sacrificing FAIR's cross-subgroup equalization
property.

Raw NSDUH public-use files are **not** in this repository. They are
freely available from SAMHSA (see *Data access* below).

## Quick start (TL;DR)

```bash
# 1. Clone and enter the repo
git clone <this-repo-url> FAIR-PLR_Suicide-Risk && cd FAIR-PLR_Suicide-Risk

# 2. Python + R dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
R -e 'install.packages("glmnet", repos="https://cloud.r-project.org/")'

# 3. Point at the raw NSDUH files you downloaded from SAMHSA
export NSDUH_DATA_DIR=/path/to/NSDUH_raw_files
export NSDUH_CLEAN_DIR=$(pwd)/data_clean

# 4. End-to-end reproduction (~2-4 h on a mid-range workstation)
./run_all.sh

# 5. Render every manuscript figure from the precomputed results/
bash -c '
  python src/build_subgroup_results.py        # tidy long-format metrics
  python src/figure3_cross_sectional.py       # main Fig. 3 + 4 (cross-sectional)
  python src/9_heatmap_crosssectional.py      # main Fig. 2 (cross-sectional coeff heatmap)
  python src/10_heatmap_pre_exposure.py       # Supp. Fig. 2 (pre-exposure coeff heatmap)
  python src/figureS_supp_cross_sectional.py  # Supp. Fig. 3-7
  python src/figureS_supp_pre_exposure.py     # Supp. Fig. 8-13
  python src/figureS_supp_coeff_cross_sectional.py  # Supp. Fig. 14-18
  python src/figureS_supp_coeff_pre_exposure.py     # Supp. Fig. 19 sub-panels
'
```

If you only want to verify figures from the precomputed `results/`
shipped with this repository, skip steps 3–4 and run step 5 directly.

## Directory layout

```
.
├── README.md                           <-- you are here
├── requirements.txt                    Python + R dependencies
├── run_all.sh                          One-command end-to-end modeling pipeline
├── run_script_clean_data.sh            Convenience wrapper for the cleaning step
├── .gitignore                          Exclusions (data, caches, venvs)
│
├── model/
│   ├── logistic_glmnet.py              FairElasticGlmNet (R glmnet via rpy2)
│   ├── baselines.py                    Standard LR, pooled EN (no weight), decision tree
│   └── joint_lasso_wrapper.py          Shared penalty wrapper used inside FAIR-PLR
│
├── src/                                See "Source layout" below
│
├── results/                            Precomputed evaluation outputs (tracked)
│   ├── metrics/                        Per-subgroup AUC / TPR / Precision tables
│   ├── coefficients/                   Per-model coefficient CSVs + heatmap inputs
│   └── predictions/                    Per-observation y_true / y_score per model
│
└── figures/                            Rendered figures (tracked)
    ├── main/                           Figures included in the main manuscript
    └── supplementary/                  Figures included in the supplementary materials
```

### Source layout (`src/`)

| File                                                     | Purpose                                                                         | Manuscript artifact                        |
| -------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------ |
| `1_clean_data.py`                                      | Raw NSDUH → harmonized per-year CSVs (+ survey weights)                        | Supp. Table 1                              |
| `2_generate_statistics_table.ipynb`                    | Descriptive statistics tables                                                   | Table 1, Table 2                           |
| `2_generate_statistics_table_combined_predictor.ipynb` | Combined-predictor descriptive statistics                                       | Supp. Table 1                              |
| `4_baselines_and_stats.py`                             | FAIR-PLR + three baselines with bootstrap CIs and DeLong tests                  | Table 3                                    |
| `5_fit_agnostic_plr.py`                                | Agnostic-PLR (single pooled elastic-net LR)                                     | (comparison model)                         |
| `6_fit_separate_plr.py`                                | Separate-PLR (one elastic-net LR per subgroup level)                            | (comparison model)                         |
| `7_fit_fair_plr.py`                                    | FAIR-PLR (combined subgroup + survey weighting)                                 | main model                                 |
| `6_disparity_calculation.ipynb`                        | Cross-subgroup disparity ratios                                                 | Table 4, Table 5                           |
| `9_heatmap_crosssectional.py`                          | Cross-sectional coefficient-variability heatmap                                 | Figure 2                                   |
| `10_heatmap_pre_exposure.py`                           | Pre-exposure coefficient-variability heatmap                                    | Supp. Figure 2                             |
| `build_subgroup_results.py`                            | Wide → long-format`subgroup_results_*.csv` for plotting                      | (preprocessing)                            |
| `figure3_cross_sectional.py`                           | FAIR-PLR vs. separate-PLR performance (cross-sectional axes)                    | Figures 3–4                               |
| `figureS_supp_cross_sectional.py`                      | Cross-sectional performance (combined Yes+No)                                   | Supp. Figures 3–7                         |
| `figureS_supp_pre_exposure.py`                         | Single-axis pre-exposure performance                                            | Supp. Figures 8–13                        |
| `figureS_supp_coeff_cross_sectional.py`                | Cross-sectional FAIR-PLR coefficient bar charts                                 | Supp. Figures 14–18                       |
| `figureS_supp_coeff_pre_exposure.py`                   | Pre-exposure coefficient bar charts                                             | Supp. Figure 19                            |
| `plotting.py`                                          | Generic grouped-bar-chart figure generator                                      | Figures 3–4, Supp. 8–13 (in-repo copies) |
| `stats_testing.py`                                     | Bootstrap CIs + DeLong paired AUC test                                          | Table 3                                    |
| `helper_function_fair.py`                              | `z_interact_multi_group()`, `get_n_k()`, `get_fair_plus_survey_weights()` | —                                         |
| `helper_functions_result.py`                           | `evaluate_model_by_group()` (AUC, TPR@k, Precision@k)                         | —                                         |
| `helper_function_disparity.py`                         | Disparity metrics (Δ_PM, max/min ratios)                                       | Tables 4–5                                |
| `helper_functions.py`                                  | Shared small utilities                                                          | —                                         |
| `_runtime.py`                                          | Shared loading/preprocessing path for`5/6/7_fit_*.py`                         | —                                         |
| `fair_subgroup_combined_weighting_example.md`          | Worked example of the combined subgroup + survey weight (Methods eq. 4)         | —                                         |

Figure 1 (main) and Supplementary Figure 1 are drawn directly in the
manuscript LaTeX (TikZ) and are not produced by this repository.

## Data access

Raw NSDUH public-use files (`NSDUH_<year>_Tab.txt` for 2020-2023, `.tsv`
for 2013-2019) can be downloaded from:

- SAMHSA data portal: [https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/datafiles](https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/datafiles)
- ICPSR NAHDAP (alternate): [https://www.datafiles.samhsa.gov/](https://www.datafiles.samhsa.gov/)

By default, the pipeline expects the raw files under `$NSDUH_DATA_DIR`
and writes harmonized per-year CSVs to `$NSDUH_CLEAN_DIR` (default
`./data_clean/`):

```bash
export NSDUH_DATA_DIR=/path/to/NSDUH_raw_files
export NSDUH_CLEAN_DIR=$(pwd)/data_clean
```

### NSDUH survey design variables (year-specific)

The NSDUH analysis-weight variable name changed twice during 2013-2023
in response to survey methodology redesigns. `src/1_clean_data.py`
handles this automatically:

| Years     | Weight           | Stratum         | Replicate |
| --------- | ---------------- | --------------- | --------- |
| 2013-2019 | `ANALWT_C`     | `VESTR`       | `VEREP` |
| 2020      | `ANALWTQ1Q4_C` | `VESTRQ1Q4_C` | `VEREP` |
| 2021-2023 | `ANALWT2_C`    | `VESTR_C`     | `VEREP` |

The cleaned CSVs harmonize these into columns `SURVEY_WEIGHT`,
`SURVEY_STRATUM`, `SURVEY_REPLICATE`, and `SURVEY_WEIGHT_TYPE` (the
latter records the original variable name so downstream code can
filter by methodology regime).

## Installation

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10–3.12.

### R (for FAIR-PLR via glmnet)

FAIR-PLR's model backend calls R's `glmnet` through `rpy2`. Install R
and the `glmnet` package:

```bash
# Debian/Ubuntu
sudo apt-get install r-base r-base-dev

# Install glmnet from within R:
R -e 'install.packages("glmnet", repos="https://cloud.r-project.org/")'
```

On macOS, `brew install R` followed by the same `install.packages` call
works.

## End-to-end reproduction

```bash
export NSDUH_DATA_DIR=/path/to/NSDUH_raw_files
export NSDUH_CLEAN_DIR=$(pwd)/data_clean

# 0. Harmonize raw NSDUH files (~5 min)
python src/1_clean_data.py
```

### Model fits

```bash
python src/5_fit_agnostic_plr.py                       # ~10 min
python src/6_fit_separate_plr.py --subgroup all        # ~30 min (6 pre-exposure axes)
python src/6_fit_separate_plr.py --subgroup cross      # ~25 min (5 cross-sectional axes)
python src/7_fit_fair_plr.py     --subgroup all        # ~45 min
python src/7_fit_fair_plr.py     --subgroup cross      # ~30 min
```

Each script writes to the canonical layout:

- `results/coefficients/coeff_<model>_<axis>_l1_<mix>.csv`
- `results/predictions/<model>_<axis>_l1_<mix>.csv` (per-observation y_true / y_pred / y_score)
- `results/metrics/<model>_<axis>_l1_<mix>.csv` (per-subgroup evaluation)
- `models/<model>_<axis>_l1_<mix>.pkl` (not tracked in git; regenerated locally)

### Baselines and statistical comparison (Table 3)

`src/4_baselines_and_stats.py` fits the main FAIR-PLR alongside three
baselines and produces the overall performance table with 95% bootstrap
confidence intervals and DeLong two-sided p-values:

- Standard logistic regression (no penalty, no interactions)
- Pooled elastic net with interactions but no subgroup-size weighting
- Decision tree (CART) with `max_depth` tuned by 5-fold CV

```bash
# All six single-subgroup axes (~1-2 h, dominated by the bootstrap)
python src/4_baselines_and_stats.py --subgroup all

# Or one axis at a time
python src/4_baselines_and_stats.py --subgroup Age

# Faster smoke test (fewer bootstrap iterations)
python src/4_baselines_and_stats.py --subgroup Age --n-bootstrap 200
```

Outputs:

- `results/predictions/scores_<model>_<axis>.csv`
- `results/metrics/bootstrap_delong_<axis>.csv` (AUC / TPR / Precision point estimates, 95% bootstrap CIs, DeLong p-values vs. FAIR-PLR)

### One-command orchestration

```bash
./run_all.sh
```

Runs cleaning, descriptive tables, the three model fits, and the
baseline + statistical comparison in sequence (~2–4 h on a mid-range
workstation, dominated by `cv.glmnet` with 20-fold CV for each FAIR-PLR
configuration).

`run_all.sh` does **not** automatically run the figure scripts — those
are listed separately in *Reproduce manuscript figures* below so
reviewers can re-render figures without re-running the multi-hour
modeling pipeline.

## Reproduce manuscript figures

The figure scripts read the precomputed CSVs under `results/` (tracked
in this repository), so **figures can be reproduced without re-running
the modeling pipeline**.

### 1. Performance figures (grouped-bar)

| Script                                                                                                        | Manuscript artifact                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python src/build_subgroup_results.py`                                                                      | (preprocessing) — converts wide-format metric CSVs into the long-format`subgroup_results_<axis>.csv` files used by every figure script below     |
| `python src/figure3_cross_sectional.py`                                                                     | Main Figures 3 and 4 (cross-sectional axes: Marital × SPD, Rurality × Treatment, Sex × SPD, Age × SPD, BMI × SPD), Yes-only and No-only panels |
| `python src/figureS_supp_cross_sectional.py`                                                                | Supp. Figures 3–7 (combined Yes+No cross-sectional)                                                                                                |
| `python src/figureS_supp_pre_exposure.py`                                                                   | Supp. Figures 8–13 (single-axis pre-exposure: Sex, Race, Age, BMI, Health Insurance, Rurality)                                                     |
| `python -m src.plotting --input results/metrics --output figures --metric tpr_top5 --ylabel "TPR (top 5%)"` | In-repo copies of the grouped-bar performance figures (`figures/`)                                                                                |

### 2. Coefficient figures (heatmaps and per-axis bar charts)

| Script                                               | Manuscript artifact                                                                        |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `python src/9_heatmap_crosssectional.py`           | Figure 2 (cross-sectional coefficient-variability heatmap)                                 |
| `python src/10_heatmap_pre_exposure.py`            | Supp. Figure 2 (pre-exposure coefficient-variability heatmap)                              |
| `python src/figureS_supp_coeff_cross_sectional.py` | Supp. Figures 14–18 (cross-sectional FAIR-PLR coefficient bar charts, SPD-Yes vs. SPD-No) |
| `python src/figureS_supp_coeff_pre_exposure.py`    | Supp. Figure 19 sub-panels (pre-exposure coefficient bar charts)                           |

### Note on figure output paths

The figure scripts write into **sibling** manuscript directories —
`../npj_NSDUH/low_res/`, `../npj_NSDUH/high_res/`, and
`../npj_supplementary_NSDUH/...` — so the rendered PNGs land where the
LaTeX `\includegraphics{...}` paths expect them. Each script
auto-creates its destination via `os.makedirs(..., exist_ok=True)`, so
cloning only this repository will create those sibling directories
beside the repo. To redirect the figures into the in-repo `figures/`
tree instead, edit the `_REPO`/`_MAN_LOW`/`_SUPP_LOW` path constants at
the top of each script (search for `os.path.join(_REPO, ...)`).

## Citing

If you use this code, please cite:

```bibtex
@article{afrose2026fairplr,
  title={Leveraging Functionally Adaptive Regularized Regression for
         Interpretable Suicide Risk Prediction and Subgroup Analysis},
  author={Afrose, Sharmin and Kazanis, William H. and Trafton, Jodie
          and Kapadia, Anuj and Peluso, Alina},
  journal={npj Mental Health Research},
  year={2026},
  doi={10.1038/s44184-026-00229-y}
}
```

## License

Code: MIT License (see `LICENSE`).
Data: NSDUH public-use files are subject to SAMHSA's terms of use; this
repository does not redistribute them.
