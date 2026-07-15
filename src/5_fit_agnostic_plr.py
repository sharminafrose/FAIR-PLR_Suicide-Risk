"""Fit Agnostic-PLR baseline (single global elastic-net logistic regression).

Trains one pooled elastic-net logistic regression on all training data, with
optional removal of a "sensitive" attribute family for the agnostic-by-feature
variant.

Usage
-----
    # Default: pooled model, no feature removal, all three mixing values
    python src/5_fit_agnostic_plr.py

    # Agnostic-by-Race (drop all Race columns from features)
    python src/5_fit_agnostic_plr.py --agnostic-of "Race"

    # Single mixing value
    python src/5_fit_agnostic_plr.py --mixing 0.5

Outputs (under NSDUH_code/results/):
    coefficients/Agnostic_<variable>_l1_<mix>.csv
    predictions/Agnostic_<variable>_l1_<mix>.csv
    metrics/Agnostic_<variable>_l1_<mix>.csv

and a pickled model under NSDUH_code/models/.
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
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from _runtime import (
    COLUMN_DICT,
    EXTRA_DEMOG_BINARIES,
    OUTCOMES,
    CATEGORICAL_COLUMNS,
    SURVEY_DESIGN_COLUMNS,
    N_POOL_YEARS,
    get_models_root,
    load_and_preprocess,
    save_predictions_and_metrics,
)


def _fit_one(data_selected: pd.DataFrame, agnostic_of: str | None, target: str,
             mixing_param: float) -> None:
    original_categorical = [COLUMN_DICT[c] for c in CATEGORICAL_COLUMNS]

    df = data_selected.dropna(subset=[target]).copy()
    _EXCLUDE_FROM_X = set(OUTCOMES) | set(SURVEY_DESIGN_COLUMNS)
    X_columns = [c for c in df.columns if c not in _EXCLUDE_FROM_X]
    X_columns_refined = [c for c in X_columns if c not in original_categorical]

    if agnostic_of:
        X_columns_refined = [c for c in X_columns_refined if not c.startswith(agnostic_of)]

    X = df[[*X_columns, *[c for c in SURVEY_DESIGN_COLUMNS if c in df.columns]]]
    y = df[target]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    X_train = X_train_raw[X_columns_refined]
    X_test = X_test_raw[X_columns_refined]

    # SAMHSA multi-year rescaling: ANALWT / Y (Y=11 pooled years). Passed to
    # LogisticRegression.fit(sample_weight=...) so the pooled model is trained
    # on the same per-respondent loss scale as FAIR-PLR (minus the FAIR
    # subgroup-balance factor, which does not apply to a non-subgroup-aware
    # pooled baseline).
    if "SURVEY_WEIGHT" in X_train_raw.columns:
        sw_train = X_train_raw["SURVEY_WEIGHT"].to_numpy(dtype=np.float64) / float(N_POOL_YEARS)
        _weight_mode = f"survey (ANALWT/{N_POOL_YEARS})"
    else:
        sw_train = None
        _weight_mode = "unweighted"

    print(f"[agnostic] mixing={mixing_param}  agnostic_of={agnostic_of}  "
          f"X_train={X_train.shape}  X_test={X_test.shape}  positives={int(y_train.sum())}  "
          f"weight={_weight_mode}")

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
    grid = GridSearchCV(
        pipe,
        param_grid={"classifier__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]},
        cv=5, scoring="roc_auc", n_jobs=-1,
    )
    _fit_kwargs = {"classifier__sample_weight": sw_train} if sw_train is not None else {}
    grid.fit(X_train, y_train, **_fit_kwargs)
    best = grid.best_estimator_
    print(f"[agnostic] best C = {grid.best_params_['classifier__C']}")

    label = agnostic_of if agnostic_of else "pooled"
    out_stem = f"Agnostic_{label.replace(' ', '_')}_l1_{mixing_param}"

    joblib.dump(best, os.path.join(get_models_root(), f"{out_stem}.pkl"))

    coef = best.named_steps["classifier"].coef_.flatten()
    pd.Series(coef, index=X_train.columns).sort_values(key=np.abs, ascending=False).to_csv(
        os.path.join(_results_dir(), "coefficients", f"coeff_{out_stem}.csv"),
        header=["coefficient"],
    )

    demog_cols = original_categorical + EXTRA_DEMOG_BINARIES
    df_demog = X_test_raw[demog_cols].reset_index(drop=True)
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)
    save_predictions_and_metrics(
        df_demog, y_test.values, y_pred, y_proba, demog_cols, out_stem,
    )


def _results_dir():
    from _runtime import get_results_root
    return get_results_root()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agnostic-of", default=None,
                    help="Drop columns starting with this human-readable prefix "
                         "(e.g. 'Race', 'Age', 'Gender'). Default: keep all features (pooled).")
    ap.add_argument("--mixing", type=float, default=None,
                    help="Single mixing value to fit; if omitted, sweeps {0, 0.5, 1.0}.")
    ap.add_argument("--target", default="ADSUITPAYR")
    args = ap.parse_args()

    data_selected, _ = load_and_preprocess()

    mixing_grid = [args.mixing] if args.mixing is not None else [0.0, 0.5, 1.0]
    for m in mixing_grid:
        _fit_one(data_selected, args.agnostic_of, args.target, m)


if __name__ == "__main__":
    main()
