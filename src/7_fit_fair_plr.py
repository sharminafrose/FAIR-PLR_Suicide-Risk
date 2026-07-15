"""Fit FAIR-PLR (the main model) for each subgroup axis.

Fits a single penalized logistic regression with group-covariate interaction
terms and the subgroup-size weight w_i = n_min / n_{k(i)}.

Backend: R's glmnet via rpy2 (FairElasticGlmNet wrapper). Requires R and
the glmnet package installed.

Usage
-----
    # All six pre-exposure subgroup axes, mixing alpha=0.5
    python src/7_fit_fair_plr.py --subgroup all

    # One axis
    python src/7_fit_fair_plr.py --subgroup Age

    # Cross-sectional axis
    python src/7_fit_fair_plr.py --subgroup "BMI:SPD"

Outputs (under NSDUH_code/results/):
    coefficients/FAIR_<subgroup>_l1_0.5.csv  -- per-subgroup-level rows with
                                                 group-specific coefficients
    predictions/FAIR_<subgroup>_l1_0.5.csv
    metrics/FAIR_<subgroup>_l1_0.5.csv

and a pickled FairElasticGlmNet under NSDUH_code/models/.
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from _runtime import (
    COLUMN_DICT,
    OUTCOMES,
    CATEGORICAL_COLUMNS,
    REFERENCE_BY_SUBGROUP,
    SURVEY_DESIGN_COLUMNS,
    N_POOL_YEARS,
    get_models_root,
    get_results_root,
    load_and_preprocess,
    save_predictions_and_metrics,
)
from helper_function_fair import (
    z_interact_multi_group,
    get_n_k,
    get_fair_plus_survey_weights,
)


PRE_EXPOSURE_SUBGROUPS = ["Age", "Gender", "Race", "BMI", "Health Insurance", "Urban Residence"]
CROSS_SECTIONAL_SUBGROUPS = ["Age:SPD", "Sex:SPD", "Marital Status:SPD",
                              "Urban Residence:Treatment", "BMI:SPD"]

_CROSS_SECTIONAL_PAIRS = {
    "Age:SPD": ("Age", "Serious Psychological Distress (any past year)"),
    "Sex:SPD": ("Gender", "Serious Psychological Distress (any past year)"),
    "Marital Status:SPD": ("Marital Status", "Serious Psychological Distress (any past year)"),
    "Urban Residence:Treatment": ("Urban Residence", "Received Substance Use or Mental Health Treatment (any past year)"),
    "BMI:SPD": ("BMI", "Serious Psychological Distress (any past year)"),
}


def _fit_one_subgroup(data_selected: pd.DataFrame, subgroup: str,
                      mixing_param: float, target: str = "ADSUITPAYR") -> None:
    # Lazy import so users without R can at least run --help
    from model.logistic_glmnet import FairElasticGlmNet

    original_categorical = [COLUMN_DICT[c] for c in CATEGORICAL_COLUMNS]
    if subgroup in CROSS_SECTIONAL_SUBGROUPS:
        original_categorical = original_categorical + [subgroup]

    df = data_selected.dropna(subset=[target]).copy()
    # Exclude outcomes and the carry-through complex-survey design columns from X;
    # the survey weight is consumed separately when building curr_n_k below.
    _EXCLUDE_FROM_X = set(OUTCOMES) | set(SURVEY_DESIGN_COLUMNS)
    X_columns = [c for c in df.columns if c not in _EXCLUDE_FROM_X]
    X_columns_refined = [
        c for c in X_columns
        if c not in original_categorical and not c.startswith(subgroup)
    ]

    X = df[[*X_columns, *[c for c in SURVEY_DESIGN_COLUMNS if c in df.columns]]]
    y = df[target]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    X_train = X_train_raw[X_columns_refined].copy()
    X_test = X_test_raw[X_columns_refined].copy()

    # Standardize non-binary columns (matches notebook cell 22)
    non_binary_cols = [c for c in X_train.columns if not (X[c].min() == 0 and X[c].max() == 1)]
    scaler = StandardScaler()
    X_train[non_binary_cols] = scaler.fit_transform(X_train[non_binary_cols])
    X_test[non_binary_cols] = scaler.transform(X_test[non_binary_cols])

    le = LabelEncoder()
    z_train = le.fit_transform(X_train_raw[subgroup])
    z_test = le.transform(X_test_raw[subgroup])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    ref_human = REFERENCE_BY_SUBGROUP.get(subgroup)
    if ref_human is None or ref_human not in label_mapping:
        # Fallback: most prevalent class in training
        values, counts = np.unique(z_train, return_counts=True)
        ref_class = int(values[np.argmax(counts)])
        ref_human = list(label_mapping.keys())[list(label_mapping.values()).index(ref_class)]
        print(f"  [warn] no reference defined for subgroup={subgroup!r}; using most prevalent={ref_human!r}")
    else:
        ref_class = int(label_mapping[ref_human])
    print(f"[fair-plr] subgroup={subgroup}  reference_class={ref_class} ({ref_human!r})  "
          f"classes={list(le.classes_)}")

    X_train_int = z_interact_multi_group(X_train.values, z_train, ref_class)
    X_test_int = z_interact_multi_group(X_test.values, z_test, ref_class)
    X_test_int = np.asarray(X_test_int, dtype=np.float64)

    # Build the combined survey + FAIR subgroup weight. When SURVEY_WEIGHT is
    # available (cleaned data from 1_clean_data.py with the survey-design
    # columns preserved), multiply each respondent's per-year-rescaled survey
    # weight (s_i / Y, Y = 11 pooled years) by the FAIR subgroup-balance factor
    # min_k(n_hat_k) / n_hat_{k(i)}. This retains national representativeness
    # within each subgroup while equalizing aggregate loss contribution across
    # subgroups. If SURVEY_WEIGHT is absent (legacy CSVs), fall back to the
    # unweighted FAIR rule get_n_k().
    if "SURVEY_WEIGHT" in X_train_raw.columns:
        s_train = X_train_raw["SURVEY_WEIGHT"].to_numpy(dtype=np.float64)
        curr_n_k = get_fair_plus_survey_weights(
            z_train, s_train, n_pool_years=N_POOL_YEARS,
        )
        _weight_mode = f"survey+FAIR (Y={N_POOL_YEARS})"
    else:
        curr_n_k = get_n_k(z_train, size_weighting=True)
        _weight_mode = "FAIR only (no survey weights)"

    print(f"  augmented feature dim = {X_train_int.shape[1]}; weight={_weight_mode}; "
          f"curr_n_k range [{curr_n_k.min():.4g}, {curr_n_k.max():.4g}]")

    model = FairElasticGlmNet()
    model.fit(X_train_int, y_train.values,
              curr_n_k=curr_n_k, l1=True, logistic=True, alpha=mixing_param)

    out_stem = f"FAIR_{subgroup.replace(' ', '_').replace(':', '_')}_l1_{mixing_param}"
    joblib.dump(model, os.path.join(get_models_root(), f"{out_stem}.pkl"))

    # Save coefficients as a (K rows by 1 + p_features) frame matching the notebook layout
    n_groups = len(label_mapping)
    coef = model.estimated_betas.reshape((n_groups, len(X_columns_refined) + 1))
    feature_names = ["intercept"] + X_columns_refined
    coef_df = pd.DataFrame(coef, columns=feature_names)
    index_list = [ref_human] + [c for c in label_mapping if c != ref_human]
    coef_df.index = index_list
    coef_df.to_csv(os.path.join(get_results_root(), "coefficients", f"coeff_{out_stem}.csv"))

    # Predictions and metrics on the test set
    y_pred = model.predict(X_test_int)
    y_score = model.predict_proba(X_test_int)
    demog = X_test_raw[[subgroup]].reset_index(drop=True)
    save_predictions_and_metrics(
        demog, y_test.values, y_pred, y_score, [subgroup], out_stem,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subgroup", default="all",
                    help="One of: " + ", ".join(PRE_EXPOSURE_SUBGROUPS + CROSS_SECTIONAL_SUBGROUPS) +
                         ", or 'all' for the six pre-exposure axes, or 'cross' for the five cross-sectional axes.")
    ap.add_argument("--mixing", type=float, default=0.5,
                    help="Elastic-net L1 ratio (default 0.5, matching the original paper)")
    ap.add_argument("--target", default="ADSUITPAYR")
    args = ap.parse_args()

    if args.subgroup == "all":
        subgroups = PRE_EXPOSURE_SUBGROUPS
    elif args.subgroup == "cross":
        subgroups = CROSS_SECTIONAL_SUBGROUPS
    else:
        subgroups = [args.subgroup]

    for sg in subgroups:
        if sg in _CROSS_SECTIONAL_PAIRS:
            data_selected, _ = load_and_preprocess(combine_features=list(_CROSS_SECTIONAL_PAIRS[sg]))
        else:
            data_selected, _ = load_and_preprocess()
        _fit_one_subgroup(data_selected, sg, args.mixing, args.target)


if __name__ == "__main__":
    main()
