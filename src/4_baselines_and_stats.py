"""Driver for the baseline models and the statistical comparison table.

This single-file, self-contained driver produces the artifacts behind
Table 3 of the manuscript (overall FAIR-PLR vs. three baselines with 95%
bootstrap confidence intervals and DeLong two-sided p-values):

  1. Main FAIR-PLR fit with the combined subgroup + survey weight
       w_i = s_i * min_k(n_hat_k) / n_hat_{k(i)}   (Methods, equation 4).

  2. Three baseline models:
       - Standard logistic regression (no penalty, no interactions)
       - Pooled elastic net with interactions but no subgroup-size weighting
       - Decision tree (CART) with max_depth tuned by 5-fold CV

  3. Bootstrap 95% CIs and DeLong paired AUC tests, saved as a tidy
     long-format CSV per subgroup axis.

Inputs
------
Cleaned per-year CSVs from $NSDUH_CLEAN_DIR (default ./data_clean), as
produced by 1_clean_data.py.

Outputs
-------
Everything lands under ./results/:

    results/predictions/
        scores_FAIR-PLR_<subgroup>.csv
        scores_Standard-LR_<subgroup>.csv
        scores_Pooled-EN-no-weight_<subgroup>.csv
        scores_Decision-Tree_<subgroup>.csv

    results/metrics/
        bootstrap_delong_<subgroup>.csv

How to invoke
-------------
From the repository root with the venv active and R + glmnet available:

    # Run everything for one subgroup axis
    python src/4_baselines_and_stats.py --subgroup Age

    # Run all six single-subgroup axes
    python src/4_baselines_and_stats.py --subgroup all

    # Faster smoke test (fewer bootstrap iterations)
    python src/4_baselines_and_stats.py --subgroup Age --n-bootstrap 200

Expected runtime per subgroup: 10-20 minutes, dominated by the bootstrap
(1000 iterations x 4 models x N subgroup categories).

This script is deliberately verbose in its logging so you can tell at a
glance which stage is executing.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Repo-relative imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from src.helper_function_fair import (
    z_interact_multi_group,
    get_n_k,
    get_fair_plus_survey_weights,
)
from src.stats_testing import (
    compare_models_on_subgroups,
)
from model.baselines import (
    fit_standard_logreg,
    fit_pooled_en_interactions_no_weight,
    fit_decision_tree,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CLEAN_DIR = os.environ.get(
    "NSDUH_CLEAN_DIR",
    os.path.abspath(os.path.join(_REPO_ROOT, "..", "data_clean")),
)
RESULTS_DIR = os.path.abspath(os.path.join(_REPO_ROOT, "results"))
FIGURES_DIR = os.path.abspath(os.path.join(_REPO_ROOT, "figures"))

# Subgroup axis specifications. For each subgroup we record:
#   - the decoded (string) column in the cleaned CSVs that defines it
#   - the "most prevalent" level that acts as the FAIR-PLR reference group
_SUBGROUP_SPECS = {
    "Age":               {"col": "CATAG6_decode",         "reference_value": "18-25 Years Old"},
    "Gender":            {"col": "IRSEX_decode",          "reference_value": "Female"},
    "Race":              {"col": "NEWRACE2_decode",       "reference_value": "NonHisp White"},
    "BMI":               {"col": "BMI2_decode",           "reference_value": "Healthy"},
    "Health Insurance":  {"col": "Health_Coverage_decode","reference_value": "Private plan"},
    "Urban Residence":   {"col": "COUTYP4_decode",        "reference_value": "Large Metropolitan"},
}

# Target outcome column
TARGET = "ADSUITPAYR"

# Predictor columns -- use the DECODED (string) versions from the cleaned CSVs so
# continuous raw codes (especially BMI2, which is a continuous BMI number) don't
# get one-hot-expanded into thousands of dummy columns. Using _decode versions
# keeps the categorical feature space at ~80 columns total after one-hot.
_PREDICTOR_COLS = [
    "CATAG6_decode", "IRSEX_decode", "NEWRACE2_decode", "IRMARIT_decode",
    "EDUHIGHCAT_decode", "POVERTY3_decode", "UD5ILALANY_decode",
    "SPDPSTMON_decode", "SPDPSTYR_decode", "RCVSUTOMHT_decode",
    "BMI2_decode", "AMDEYR_decode", "IRWRKSTAT18_decode", "INCOME_decode",
    "Health_Coverage_decode", "COUTYP4_decode",
    "IRPYUD5ALC_decode", "IRPYUD5MRJ_decode", "STMWYNORX_decode",
    "SEDWYNORX_decode", "IRPYUD5COC_decode", "IRPYUD5HER_decode",
    "IRPYUD5HAL_decode", "IRPYUD5INH_decode", "OXYCNANYYR_decode",
    "UD5ILLANY_decode", "BNGDRKMON_decode", "HVYDRKMON_decode",
    "IRIMPRESP_decode", "AD_MDEA6_decode", "AD_MDEA7_decode",
]

# Columns that are genuinely categorical (few discrete string values). These
# become one-hot. BMI2_decode is here instead of raw numeric BMI2 so the
# feature matrix stays compact.
_CATEGORICAL_COLS = [
    "CATAG6_decode", "IRSEX_decode", "NEWRACE2_decode", "IRMARIT_decode",
    "EDUHIGHCAT_decode", "POVERTY3_decode", "BMI2_decode",
    "IRWRKSTAT18_decode", "INCOME_decode", "Health_Coverage_decode",
    "COUTYP4_decode",
]

# The remaining predictor columns (everything in _PREDICTOR_COLS minus the
# categoricals) are binary-style 0/1 or "Yes"/"No" decoded; we map them below.
_BINARY_DECODE_COLS = [c for c in _PREDICTOR_COLS if c not in _CATEGORICAL_COLS]


# -----------------------------------------------------------------------------
# Data loading + feature engineering
# -----------------------------------------------------------------------------

def load_pooled_data(clean_dir: str = CLEAN_DIR) -> pd.DataFrame:
    """Load all per-year cleaned CSVs and concatenate into one DataFrame.

    Adds a 'Year' column (extracted from the filename) so the model can
    include a secular-trend effect. Drops rows with a missing target.
    """
    frames = []
    for path in sorted(glob.glob(os.path.join(clean_dir, "clean_data_*.csv"))):
        year = int(os.path.basename(path).split("_")[-1].split(".")[0])
        df = pd.read_csv(path)
        df["Year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No clean_data_*.csv files found in {clean_dir}")
    df_all = pd.concat(frames, axis=0, ignore_index=True)
    df_all = df_all.dropna(subset=[TARGET])
    df_all[TARGET] = df_all[TARGET].astype(int)
    return df_all


def engineer_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Build the modeling feature matrix from the cleaned per-year CSVs.

    All predictor columns are the `_decode` (string-typed) versions from
    `1_clean_data.py`. Categoricals become one-hot dummies; binary "Yes"/"No"
    strings are mapped to 1/0; Year is standardized to zero mean unit variance.

    Returns a float numpy array and the list of feature names.
    """
    X = df[_PREDICTOR_COLS + ["Year"]].copy()

    # Binary decoded columns: map "Yes"/"No" -> 1/0. Works on object-dtype,
    # pandas 2.x/3.x string-dtype (`StringArray`), and already-numeric columns
    # (in which case the map returns NaN which we coerce back via a fallback).
    for col in _BINARY_DECODE_COLS:
        mapped = X[col].astype("string").map({"Yes": 1, "No": 0})
        # If the column was actually numeric 0/1 already, the .map() above
        # returned NaN; recover those values.
        fallback = pd.to_numeric(X[col], errors="coerce")
        X[col] = mapped.fillna(fallback).astype("Int64")

    # Categorical decoded columns: one-hot with clean string dummies
    dummies = pd.get_dummies(
        X[_CATEGORICAL_COLS].astype(str),
        prefix=_CATEGORICAL_COLS,
        drop_first=False,
    )
    X = X.drop(columns=_CATEGORICAL_COLS).join(dummies)

    # Standardize Year
    X["Year"] = (X["Year"] - X["Year"].mean()) / X["Year"].std(ddof=0)

    # Cast to float; any remaining NaN is filled with 0 so downstream glmnet
    # doesn't choke (consistent with the main pipeline's behavior)
    X = X.astype(float).fillna(0.0)
    feature_names = X.columns.tolist()
    return X.values, feature_names


def split_and_subgroup(
    df: pd.DataFrame,
    X: np.ndarray,
    subgroup: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """Produce the 80/20 train/test split and label-encoded subgroup vectors.

    The subgroup column is taken from the cleaned data BEFORE one-hot
    encoding, so the FAIR interaction construction in z_interact_multi_group
    gets integer-encoded subgroup labels as expected.
    """
    spec = _SUBGROUP_SPECS[subgroup]
    subgroup_series = df[spec["col"]]

    # Label-encode so subgroup labels are always 0..K-1
    le = LabelEncoder()
    z_all = le.fit_transform(subgroup_series.astype(str))

    # Find the reference class's encoded index
    ref_raw = str(spec["reference_value"])
    if ref_raw in le.classes_:
        reference_class = int(np.where(le.classes_ == ref_raw)[0][0])
    else:
        # Fallback: pick the most prevalent class
        values, counts = np.unique(z_all, return_counts=True)
        reference_class = int(values[np.argmax(counts)])
        print(f"  [warn] reference_value {ref_raw!r} not found in subgroup {subgroup!r}; "
              f"falling back to most-prevalent (encoded={reference_class}, raw={le.classes_[reference_class]!r})")

    y = df[TARGET].values.astype(int)
    survey_w = df["SURVEY_WEIGHT"].values.astype(float) if "SURVEY_WEIGHT" in df.columns else None
    year = df["Year"].values.astype(int) if "Year" in df.columns else None

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "X_train": X[idx_tr],
        "X_test": X[idx_te],
        "y_train": y[idx_tr],
        "y_test": y[idx_te],
        "z_train": z_all[idx_tr],
        "z_test": z_all[idx_te],
        "subgroup_labels_train": subgroup_series.values[idx_tr],
        "subgroup_labels_test": subgroup_series.values[idx_te],
        "reference_class": reference_class,
        "label_classes": le.classes_,
        "survey_weight_train": survey_w[idx_tr] if survey_w is not None else None,
        "survey_weight_test": survey_w[idx_te] if survey_w is not None else None,
        "year_train": year[idx_tr] if year is not None else None,
        "year_test": year[idx_te] if year is not None else None,
    }


# -----------------------------------------------------------------------------
# Baselines and FAIR-PLR for one subgroup axis
# -----------------------------------------------------------------------------

def run_baselines(split: Dict[str, np.ndarray], subgroup: str, out_dir: str) -> Dict[str, np.ndarray]:
    """Fit the three baselines and return their predicted scores on the test set."""
    os.makedirs(out_dir, exist_ok=True)
    X_tr, y_tr = split["X_train"], split["y_train"]
    X_te = split["X_test"]
    # Per-respondent survey weight rescaled by SAMHSA Y=11 convention, passed
    # to Standard-LR and Decision-Tree so they share the same loss scaling as
    # FAIR-PLR. Pooled-EN-no-weight intentionally stays unweighted (its purpose
    # is to isolate the subgroup-weighting contribution).
    sw_tr = (
        np.asarray(split["survey_weight_train"], dtype=float) / 11.0
        if split.get("survey_weight_train") is not None else None
    )

    scores = {}
    print(f"[baselines] fitting Standard-LR on subgroup={subgroup} ...")
    res_lr = fit_standard_logreg(X_tr, y_tr, X_te, sample_weight=sw_tr)
    scores["Standard-LR"] = res_lr["y_score"]
    pd.DataFrame({
        "subgroup_label": split["subgroup_labels_test"],
        "y_true": split["y_test"],
        "y_score": res_lr["y_score"],
    }).to_csv(os.path.join(out_dir, f"scores_Standard-LR_{subgroup}.csv"), index=False)

    print(f"[baselines] fitting Pooled-EN-no-weight on subgroup={subgroup} ...")
    res_en = fit_pooled_en_interactions_no_weight(
        X_tr, y_tr, X_te, split["z_train"], split["z_test"],
        reference_class=split["reference_class"],
    )
    scores["Pooled-EN-no-weight"] = res_en["y_score"]
    pd.DataFrame({
        "subgroup_label": split["subgroup_labels_test"],
        "y_true": split["y_test"],
        "y_score": res_en["y_score"],
    }).to_csv(os.path.join(out_dir, f"scores_Pooled-EN-no-weight_{subgroup}.csv"), index=False)

    print(f"[baselines] fitting Decision-Tree on subgroup={subgroup} ...")
    res_dt = fit_decision_tree(X_tr, y_tr, X_te, sample_weight=sw_tr)
    scores["Decision-Tree"] = res_dt["y_score"]
    print(f"  best_max_depth = {res_dt['best_max_depth']}")
    pd.DataFrame({
        "subgroup_label": split["subgroup_labels_test"],
        "y_true": split["y_test"],
        "y_score": res_dt["y_score"],
    }).to_csv(os.path.join(out_dir, f"scores_Decision-Tree_{subgroup}.csv"), index=False)

    return scores


def run_fairplr_main(split: Dict[str, np.ndarray], subgroup: str, out_dir: str) -> np.ndarray:
    """Fit the main FAIR-PLR with combined survey + FAIR weight (Y=11) and save test-set scores."""
    from model.logistic_glmnet import FairElasticGlmNet
    os.makedirs(out_dir, exist_ok=True)

    print(f"[fair-plr] fitting FAIR-PLR (subgroup-weighted) on subgroup={subgroup} ...")
    X_train_int = z_interact_multi_group(split["X_train"], split["z_train"], split["reference_class"])
    X_test_int = z_interact_multi_group(split["X_test"], split["z_test"], split["reference_class"])
    if split.get("survey_weight_train") is not None:
        curr_n_k = get_fair_plus_survey_weights(
            split["z_train"], split["survey_weight_train"], n_pool_years=11,
        )
        _weight_mode = "survey+FAIR (Y=11)"
    else:
        curr_n_k = get_n_k(split["z_train"], size_weighting=True)
        _weight_mode = "FAIR only (no survey weights)"
    print(f"  weight={_weight_mode}; curr_n_k range [{curr_n_k.min():.4g}, {curr_n_k.max():.4g}]")

    m = FairElasticGlmNet()
    m.fit(X_train_int, split["y_train"], curr_n_k=curr_n_k, logistic=True, alpha=0.5)
    y_score = m.predict_proba(X_test_int)

    pd.DataFrame({
        "subgroup_label": split["subgroup_labels_test"],
        "y_true": split["y_test"],
        "y_score": y_score,
    }).to_csv(os.path.join(out_dir, f"scores_FAIR-PLR_{subgroup}.csv"), index=False)

    # Save the model too
    import joblib
    models_dir = os.path.abspath(os.path.join(_REPO_ROOT, "models"))
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(m, os.path.join(models_dir, f"fair_{subgroup}_alpha_0.5.pkl"))
    return y_score


# -----------------------------------------------------------------------------
# Bootstrap CIs and DeLong tests across all fitted models
# -----------------------------------------------------------------------------

def run_stats_testing(
    split: Dict[str, np.ndarray],
    scores_by_model: Dict[str, np.ndarray],
    subgroup: str,
    out_dir: str,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Compute bootstrap CIs and DeLong tests for all model-by-subgroup cells."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"[stats] bootstrap ({n_bootstrap} iter) + DeLong across "
          f"{len(scores_by_model)} models x subgroup '{subgroup}' ...")
    cmp_df = compare_models_on_subgroups(
        y_true=split["y_test"],
        scores_by_model=scores_by_model,
        subgroup_labels=split["subgroup_labels_test"],
        reference_model="FAIR-PLR",
        n_bootstrap=n_bootstrap,
    )
    cmp_df.to_csv(os.path.join(out_dir, f"bootstrap_delong_{subgroup}.csv"), index=False)
    return cmp_df


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def run_one_subgroup(subgroup: str, df_all: pd.DataFrame, X_all: np.ndarray,
                     n_bootstrap: int = 1000) -> None:
    t0 = time.time()
    print(f"\n=== Subgroup: {subgroup} ===")

    split = split_and_subgroup(df_all, X_all, subgroup)
    print(f"[split] train={len(split['y_train']):,}  test={len(split['y_test']):,}  "
          f"subgroup_classes={list(split['label_classes'])}  reference={split['reference_class']}")

    # 1. Main FAIR-PLR
    score_fair = run_fairplr_main(split, subgroup, os.path.join(RESULTS_DIR, "predictions"))

    # 2. Three baselines (Standard-LR, Pooled-EN-no-weight, Decision-Tree)
    baseline_scores = run_baselines(split, subgroup, os.path.join(RESULTS_DIR, "predictions"))
    scores_by_model = {"FAIR-PLR": score_fair, **baseline_scores}

    # 3. Bootstrap + DeLong across all models
    run_stats_testing(split, scores_by_model, subgroup,
                      os.path.join(RESULTS_DIR, "metrics"),
                      n_bootstrap=n_bootstrap)

    print(f"=== Subgroup '{subgroup}' done in {time.time() - t0:.1f}s ===\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subgroup", default="all",
                    help="One of: Age, Gender, Race, BMI, 'Health Insurance', 'Urban Residence', or 'all'")
    ap.add_argument("--n-bootstrap", type=int, default=1000,
                    help="Number of bootstrap iterations for stat testing (default 1000)")
    args = ap.parse_args()

    print(f"Loading pooled NSDUH cleaned data from {CLEAN_DIR} ...")
    df_all = load_pooled_data(CLEAN_DIR)
    print(f"  pooled records: {len(df_all):,}")
    print(f"  survey-weight regimes: {df_all['SURVEY_WEIGHT_TYPE'].value_counts().to_dict() if 'SURVEY_WEIGHT_TYPE' in df_all.columns else '<missing>'}")

    print("Engineering features (one-hot encoding + Year scaling) ...")
    X_all, feature_names = engineer_features(df_all)
    print(f"  feature matrix shape: {X_all.shape}")

    if args.subgroup == "all":
        subgroups = list(_SUBGROUP_SPECS.keys())
    else:
        subgroups = [args.subgroup]

    for sg in subgroups:
        run_one_subgroup(sg, df_all, X_all, n_bootstrap=args.n_bootstrap)

    print("\nAll requested experiments complete.")
    print(f"Results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
