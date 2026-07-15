"""Fit Separate-PLR baseline (one elastic-net logistic regression per subgroup).

For each subgroup axis
(Age, Gender, Race, BMI, Health Insurance, Urban Residence, plus the
cross-sectional combinations), trains an independent elastic-net logistic
regression on each subgroup level and evaluates only on test rows in that
subgroup.

Usage
-----
    # All six pre-exposure subgroups
    python src/6_fit_separate_plr.py --subgroup all

    # One axis
    python src/6_fit_separate_plr.py --subgroup Age

    # Cross-sectional axis (e.g. Sex x SPD)
    python src/6_fit_separate_plr.py --subgroup "Sex:SPD"

Outputs (under NSDUH_code/results/):
    coefficients/Separate_<subgroup>_<level>_l1_0.5.csv
    predictions/Separate_<subgroup>_<level>_l1_0.5.csv
    metrics/Separate_<subgroup>_<level>_l1_0.5.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from _runtime import (
    COLUMN_DICT,
    OUTCOMES,
    CATEGORICAL_COLUMNS,
    SURVEY_DESIGN_COLUMNS,
    N_POOL_YEARS,
    get_models_root,
    get_results_root,
    load_and_preprocess,
    save_predictions_and_metrics,
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
    original_categorical = [COLUMN_DICT[c] for c in CATEGORICAL_COLUMNS]
    if subgroup in CROSS_SECTIONAL_SUBGROUPS:
        original_categorical = original_categorical + [subgroup]

    df = data_selected.dropna(subset=[target]).copy()
    _EXCLUDE_FROM_X = set(OUTCOMES) | set(SURVEY_DESIGN_COLUMNS)
    X_columns = [c for c in df.columns if c not in _EXCLUDE_FROM_X]
    X_columns_refined = [c for c in X_columns if c not in original_categorical]
    # Drop columns starting with the subgroup name (so each subgroup model
    # is fit on only the non-subgroup features within its slice)
    X_columns_refined = [c for c in X_columns_refined if not c.startswith(subgroup)]

    X = df[[*X_columns, *[c for c in SURVEY_DESIGN_COLUMNS if c in df.columns]]]
    y = df[target]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    X_train = X_train_raw[X_columns_refined]
    X_test = X_test_raw[X_columns_refined]

    # SAMHSA multi-year rescaling. No FAIR subgroup-balance factor here: each
    # Separate-PLR model is already fit on a single subgroup slice, so
    # between-subgroup balancing is not a concept for this baseline.
    if "SURVEY_WEIGHT" in X_train_raw.columns:
        sw_train_full = X_train_raw["SURVEY_WEIGHT"].to_numpy(dtype=np.float64) / float(N_POOL_YEARS)
        _weight_mode = f"survey (ANALWT/{N_POOL_YEARS})"
    else:
        sw_train_full = None
        _weight_mode = "unweighted"

    le = LabelEncoder()
    z_train = le.fit_transform(X_train_raw[subgroup])
    z_test = le.transform(X_test_raw[subgroup])
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"[separate] subgroup={subgroup}  classes={list(le.classes_)}  weight={_weight_mode}")

    passthrough_cols = [c for c in X_train.columns if c != "Year"]
    transformer = ColumnTransformer([
        ("scale", StandardScaler(), ["Year"]),
        ("pass", "passthrough", passthrough_cols),
    ])
    pipe = Pipeline([
        ("transformer", transformer),
        ("classifier", LogisticRegression(
            penalty="elasticnet", solver="saga", max_iter=500, l1_ratio=mixing_param,
        )),
    ])
    param_grid = {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]}

    for level, encoded in label_mapping.items():
        mask_tr = z_train == encoded
        mask_te = z_test == encoded
        if mask_tr.sum() == 0 or y_train[mask_tr].sum() == 0:
            print(f"  [skip] {level}: 0 training rows or 0 positives")
            continue

        print(f"  [fit] level={level!r}  n_train={int(mask_tr.sum()):,}  positives={int(y_train[mask_tr].sum())}")
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
        _fit_kwargs = (
            {"classifier__sample_weight": sw_train_full[mask_tr]}
            if sw_train_full is not None else {}
        )
        grid.fit(X_train[mask_tr], y_train[mask_tr], **_fit_kwargs)
        best = grid.best_estimator_

        safe_level = str(level).replace("/", "").replace(" ", "_").replace(":", "_")
        out_stem = f"Separate_{subgroup.replace(' ', '_').replace(':', '_')}_{safe_level}_l1_{mixing_param}"

        joblib.dump(best, os.path.join(get_models_root(), f"{out_stem}.pkl"))

        coef = best.named_steps["classifier"].coef_.flatten()
        pd.Series(coef, index=X_train.columns).sort_values(key=np.abs, ascending=False).to_csv(
            os.path.join(get_results_root(), "coefficients", f"coeff_{out_stem}.csv"),
            header=["coefficient"],
        )

        demog = X_test_raw[mask_te][[subgroup]].reset_index(drop=True)
        y_pred = best.predict(X_test[mask_te])
        y_proba = best.predict_proba(X_test[mask_te])
        save_predictions_and_metrics(
            demog,
            y_test[mask_te].values,
            y_pred,
            y_proba,
            [subgroup],
            out_stem,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subgroup", default="all",
                    help="One of: " + ", ".join(PRE_EXPOSURE_SUBGROUPS + CROSS_SECTIONAL_SUBGROUPS) +
                         ", or 'all' for all six pre-exposure subgroups, or 'cross' for all five cross-sectional.")
    ap.add_argument("--mixing", type=float, default=0.5,
                    help="Elastic-net L1 ratio (default: 0.5)")
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
