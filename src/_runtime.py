"""Shared loading / preprocessing utilities for the standalone model scripts.

Provides the shared data-loading and feature-engineering pipeline so the
standalone scripts (5_fit_agnostic_plr.py, 6_fit_separate_plr.py,
7_fit_fair_plr.py) can share one canonical preprocessing path.

Path resolution
---------------
The cleaned per-year CSVs are expected at the path returned by
get_clean_dir(), which is:

    $NSDUH_CLEAN_DIR  if set
    else  <repo>/data_clean   (sibling of NSDUH_code/)

Output directory layout
-----------------------
All scripts write to a single canonical results tree under <repo>/NSDUH_code:

    results/coefficients/  -- per-model coefficient tables (CSV)
    results/predictions/   -- per-observation y_true / y_pred / y_score CSVs
    results/metrics/       -- per-model evaluation tables (output of evaluate_model_by_group)
    models/                -- pickled fitted models
"""

from __future__ import annotations

import glob
import os
from typing import Tuple

import numpy as np
import pandas as pd

from helper_functions import combine_data, get_decoded_data_and_analyse, preprocess_data
from helper_functions_result import evaluate_model_by_group


# -----------------------------------------------------------------------------
# Constants matching the notebook configuration
# -----------------------------------------------------------------------------

PREDICTORS = [
    "Year",
    "CATAG6_decode", "IRSEX_decode", "NEWRACE2_decode", "IRMARIT_decode",
    "EDUHIGHCAT_decode", "POVERTY3_decode", "UD5ILALANY_decode",
    "SPDPSTMON_decode", "SPDPSTYR_decode", "RCVSUTOMHT_decode",
    "AMDEYR_decode", "IRWRKSTAT18_decode", "INCOME_decode",
    "Health_Coverage_decode", "COUTYP4_decode",
    "IRPYUD5ALC_decode", "IRPYUD5MRJ_decode", "STMWYNORX_decode",
    "SEDWYNORX_decode", "IRPYUD5COC_decode", "IRPYUD5HER_decode",
    "IRPYUD5HAL_decode", "IRPYUD5INH_decode", "OXYCNANYYR_decode",
    "UD5ILLANY_decode", "BNGDRKMON_decode", "HVYDRKMON_decode",
    "IRIMPRESP_decode", "AD_MDEA6_decode", "AD_MDEA7_decode",
    "BMI2_decode",
]
OUTCOMES = ["ADSUITPAYR", "IRSUICTHNK", "IRSUIPLANYR", "IRSUITRYYR"]

BINARY_COLUMNS = [
    "UD5ILALANY", "SPDPSTMON", "SPDPSTYR", "RCVSUTOMHT", "AMDEYR",
    "IRPYUD5ALC", "IRPYUD5MRJ", "STMWYNORX", "SEDWYNORX",
    "IRPYUD5COC", "IRPYUD5HER", "IRPYUD5HAL", "IRPYUD5INH",
    "OXYCNANYYR", "UD5ILLANY", "BNGDRKMON", "HVYDRKMON",
    "IRIMPRESP", "AD_MDEA6", "AD_MDEA7",
]
CATEGORICAL_COLUMNS = [
    "CATAG6", "IRSEX", "NEWRACE2", "IRMARIT", "EDUHIGHCAT", "POVERTY3",
    "IRWRKSTAT18", "INCOME", "Health_Coverage", "COUTYP4", "BMI2",
]

COLUMN_DICT = {
    "CATAG6": "Age",
    "IRSEX": "Gender",
    "NEWRACE2": "Race",
    "IRMARIT": "Marital Status",
    "EDUHIGHCAT": "Education",
    "POVERTY3": "Poverty",
    "IRWRKSTAT18": "Employment Status",
    "INCOME": "Family Income",
    "Health_Coverage": "Health Insurance",
    "COUTYP4": "Urban Residence",
    "BMI2": "BMI",
    "UD5ILALANY": "Drug or Alcohol Use Disorder (any past year)",
    "SPDPSTMON": "Serious Psychological Distress (any past month)",
    "SPDPSTYR": "Serious Psychological Distress (any past year)",
    "RCVSUTOMHT": "Received Substance Use or Mental Health Treatment (any past year)",
    "AMDEYR": "Major Depressive Episode (any past year)",
    "IRPYUD5ALC": "Alcohol Use Disorder (any past year)",
    "IRPYUD5MRJ": "Marijuana Use Disorder (any past year)",
    "STMWYNORX": "Stimulant Use w/o RX (any past year)",
    "SEDWYNORX": "Sedative Use w/o RX (any past year)",
    "IRPYUD5COC": "Cocaine Use Disorder (any past year)",
    "IRPYUD5HER": "Heroin Use Disorder (any past year)",
    "IRPYUD5HAL": "Hallucinogen Use Disorder (any past year)",
    "IRPYUD5INH": "Inhalant Use Disorder (any past year)",
    "OXYCNANYYR": "Oxycontin Use (any past year)",
    "UD5ILLANY": "Drug Use Disorder (any past year)",
    "BNGDRKMON": "Binge Alcohol Use (any past month)",
    "HVYDRKMON": "Heavy Alcohol Use (any past month)",
    "IRIMPRESP": "Difficulty in Work Response (any past year)",
    "AD_MDEA6": "Felt Tired/Low Energy (nearly every day)",
    "AD_MDEA7": "Felt Worthless (nearly every day)",
}

# Reference (most prevalent) class per subgroup axis.
# Used by FAIR-PLR for the interaction expansion's reference block.
REFERENCE_BY_SUBGROUP = {
    "Age": "18-25 Years Old",
    "Gender": "Female",
    "Race": "NonHisp White",
    "BMI": "Healthy",
    "Health Insurance": "Private plan",
    "Urban Residence": "Large Metropolitan",
    "Age:SPD": "18-25 Years Old:No",
    "Gender:SPD": "Female:No",
    "Marital Status:SPD": "Married:No",
    "Urban Residence:Treatment": "Large Metropolitan:No",
    "BMI:SPD": "Healthy:No",
}

# Binary columns we want preserved (in raw form, alongside one-hot dummies)
# in the prediction CSVs so subgroup stratification still works downstream.
EXTRA_DEMOG_BINARIES = [
    "Drug Use Disorder (any past year)",
    "Serious Psychological Distress (any past year)",
    "Felt Worthless (nearly every day)",
    "Major Depressive Episode (any past year)",
    "Received Substance Use or Mental Health Treatment (any past year)",
]


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)


def get_clean_dir() -> str:
    """Return the directory containing per-year clean_data_<year>.csv files."""
    default = os.path.abspath(os.path.join(_REPO_ROOT, "..", "data_clean"))
    return os.environ.get("NSDUH_CLEAN_DIR", default)


def get_results_root() -> str:
    """Return <repo>/NSDUH_code/results, creating subdirectories if absent."""
    root = os.path.join(_REPO_ROOT, "results")
    for sub in ("coefficients", "predictions", "metrics"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


def get_models_root() -> str:
    """Return <repo>/NSDUH_code/models, creating it if absent."""
    root = os.path.join(_REPO_ROOT, "models")
    os.makedirs(root, exist_ok=True)
    return root


# -----------------------------------------------------------------------------
# Data loading + preprocessing (mirrors notebook cells 2-10)
# -----------------------------------------------------------------------------

def load_and_preprocess(combine_features: list = None) -> Tuple[pd.DataFrame, list]:
    """Load all cleaned per-year CSVs and apply the same preprocessing as the notebook.

    Parameters
    ----------
    combine_features : list of two strings, optional
        If provided, creates an interaction column (e.g. "Age:SPD") by
        concatenating the two named columns. Matches the notebook's
        cell 9 logic. Pass None for single-attribute analyses.

    Returns
    -------
    data_selected : pd.DataFrame
        Preprocessed DataFrame with one-hot dummies, binary columns mapped
        to 0/1, and (optionally) an interaction column.
    categorical_columns : list of str
        The list of categorical columns (potentially with the new
        interaction feature appended) for downstream processing.
    """
    clean_dir = get_clean_dir()
    file_path = os.path.join(clean_dir, "*.csv")
    print(f"[load] reading per-year CSVs from {clean_dir}")

    data = combine_data(file_path)
    predictors = list(PREDICTORS)
    categorical_columns = list(CATEGORICAL_COLUMNS)

    data_selected = get_decoded_data_and_analyse(data, COLUMN_DICT, predictors, OUTCOMES)

    # Snapshot the raw survey-design columns; they are re-attached AFTER
    # preprocess_data (post-hoc) to avoid interfering with pd.get_dummies /
    # pd.concat, which on pandas 3.x + Apple Silicon can segfault when a
    # string-dtype column (SURVEY_WEIGHT_TYPE) is mixed with Int64 / dummy
    # columns during wide concatenation.
    assert len(data_selected) == len(data), (
        "row-count mismatch between data and data_selected; cannot align "
        "survey-design columns positionally"
    )
    _design_snapshot = {
        c: data[c].values
        for c in ("SURVEY_WEIGHT", "SURVEY_STRATUM", "SURVEY_REPLICATE",
                  "SURVEY_WEIGHT_TYPE")
        if c in data.columns
    }

    # Apply optional cross-sectional combination (e.g. Age x SPD)
    if combine_features:
        f1, f2 = combine_features
        out = _make_combined_label(f1, f2)
        if out is not None:
            new_col, new_dict = out
            spd_col = COLUMN_DICT.get("SPDPSTYR", "Serious Psychological Distress (any past year)")
            data_selected[new_col] = (
                data_selected[COLUMN_DICT[_lookup_dict_key(f1)]] + ":" +
                data_selected[COLUMN_DICT[_lookup_dict_key(f2)]].astype(str)
            )
            predictors.append(new_col)
            categorical_columns.append(new_col)
            COLUMN_DICT[new_col] = new_col
            print(f"[load] created cross-sectional column '{new_col}'")
            print(data_selected[new_col].value_counts())

    data_selected = preprocess_data(categorical_columns, BINARY_COLUMNS, data_selected, COLUMN_DICT)

    # Re-attach survey-design columns now that all dummy-encoding is done.
    # SURVEY_WEIGHT is the numeric weight used by FAIR-PLR; the others are
    # carried for downstream design-aware analyses. Length is guaranteed to
    # match by the snapshot assertion above (preprocess_data preserves row order).
    if len(data_selected) != len(next(iter(_design_snapshot.values()), [])) and _design_snapshot:
        raise RuntimeError(
            "preprocess_data changed the row count; cannot reattach survey design columns"
        )
    for _c, _vals in _design_snapshot.items():
        data_selected[_c] = _vals

    return data_selected, categorical_columns


def _lookup_dict_key(human_name: str) -> str:
    """Reverse-lookup the raw NSDUH variable name from its human-readable label."""
    for k, v in COLUMN_DICT.items():
        if v == human_name:
            return k
    raise KeyError(f"No raw column matches human label {human_name!r}")


def _make_combined_label(f1: str, f2: str):
    """Returns (new_col_name, new_dict_entry) or None if the combination is not recognized."""
    pair = (f1, f2)
    canonical = {
        ("Age", "Serious Psychological Distress (any past year)"): "Age:SPD",
        ("Gender", "Serious Psychological Distress (any past year)"): "Sex:SPD",
        ("Marital Status", "Serious Psychological Distress (any past year)"): "Marital Status:SPD",
        ("Urban Residence", "Received Substance Use or Mental Health Treatment (any past year)"): "Urban Residence:Treatment",
        ("BMI", "Serious Psychological Distress (any past year)"): "BMI:SPD",
    }
    for key, name in canonical.items():
        if pair == key:
            return name, name
    return None


# -----------------------------------------------------------------------------
# Convenience: tidy CSV writer for predictions
# -----------------------------------------------------------------------------

def save_predictions_and_metrics(
    df_demographic: pd.DataFrame,
    y_test, y_pred, y_score,
    demographic_cols: list,
    out_stem: str,
) -> None:
    """Write a per-observation prediction CSV and a per-subgroup metrics CSV."""
    results = get_results_root()
    df_demographic = df_demographic.copy()
    df_demographic["actual"] = pd.Series(y_test).astype(int).values
    df_demographic["predicted"] = pd.Series(y_pred).astype(int).values
    if y_score.ndim == 2:
        df_demographic["score"] = y_score[:, 1]
    else:
        df_demographic["score"] = y_score

    pred_path = os.path.join(results, "predictions", f"{out_stem}.csv")
    df_demographic.to_csv(pred_path, index=False)

    metrics_df = evaluate_model_by_group(
        df_demographic,
        demographic_cols,
        actual_col="actual",
        predicted_col="predicted",
        score_col="score",
    )
    metrics_path = os.path.join(results, "metrics", f"{out_stem}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[save] {pred_path}\n[save] {metrics_path}")


SURVEY_DESIGN_COLUMNS = [
    "SURVEY_WEIGHT", "SURVEY_STRATUM", "SURVEY_REPLICATE", "SURVEY_WEIGHT_TYPE",
]

# SAMHSA multi-year pooling divisor: number of years in the pooled analysis
# (2013-2023 inclusive = 11 years). Dividing each respondent's raw survey weight
# by this divisor rescales the pooled sample to a single-year-equivalent U.S.
# population total, eliminating the 11-fold double-counting that arises from
# concatenating annually-calibrated weights.
N_POOL_YEARS = 11


__all__ = [
    "PREDICTORS", "OUTCOMES", "BINARY_COLUMNS", "CATEGORICAL_COLUMNS",
    "COLUMN_DICT", "REFERENCE_BY_SUBGROUP", "EXTRA_DEMOG_BINARIES",
    "SURVEY_DESIGN_COLUMNS", "N_POOL_YEARS",
    "get_clean_dir", "get_results_root", "get_models_root",
    "load_and_preprocess", "save_predictions_and_metrics",
]
