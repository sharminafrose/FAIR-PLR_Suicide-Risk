"""Baseline models for FAIR-PLR comparison.

Implements the three comparison baselines:

  1. Standard (unpenalized) logistic regression, pooled, no interactions.

  2. Pooled elastic-net logistic regression with group-covariate interaction
     terms but NO subgroup-size weighting. This ablation isolates whether the
     subgroup-size weighting is what drives FAIR-PLR's fairness gains (as
     opposed to the interaction terms alone). Reuses the FairElasticGlmNet
     backend with `curr_n_k=None` (uniform weights).

  3. Decision tree (CART) with max_depth tuned via 5-fold CV. A
     representative interpretable non-linear baseline.

All three baselines use the same 80/20 train/test split and the same
evaluate_model_by_group() function as the main FAIR-PLR runs, so their
outputs are directly comparable column-by-column in the results tables.

Usage
-----
    from model.baselines import (
        fit_standard_logreg,
        fit_pooled_en_interactions_no_weight,
        fit_decision_tree,
    )

    preds_lr = fit_standard_logreg(X_train, y_train, X_test)
    preds_en = fit_pooled_en_interactions_no_weight(
        X_train, y_train, X_test, z_train, z_test, reference_class=most_prevalent_class
    )
    preds_dt = fit_decision_tree(X_train, y_train, X_test)

Each fitter returns a dict with keys `y_pred` (binary) and `y_score`
(probability) aligned with X_test rows.
"""

from __future__ import annotations

import sys
import os
from typing import Dict

import numpy as np

# Allow running as a script from NSDUH_code/ with `python -m model.baselines`
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from src.helper_function_fair import z_interact_multi_group

# `FairElasticGlmNet` is imported lazily inside the one function that needs it,
# so that environments without R/rpy2 (e.g. CI runners, pure-Python baseline
# checks) can still use `fit_standard_logreg` and `fit_decision_tree`.


# -----------------------------------------------------------------------------
# 1. Standard logistic regression (no penalty, no interactions)
# -----------------------------------------------------------------------------

def fit_standard_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    max_iter: int = 1000,
    sample_weight: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Fit a plain logistic regression and return predictions on the test set.

    This is the simplest interpretable baseline: unpenalized maximum-
    likelihood logistic regression on the 33 harmonized features, with no
    subgroup interaction terms and no per-observation weighting.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices. Both should be standardized and encoded as they are
        fed to the other models (one-hot categorical, binary 0/1).
    y_train : np.ndarray
        Binary outcome (0/1) for the training set.
    max_iter : int, default 1000
        Solver iteration cap. Raised above the sklearn default because the
        full NSDUH feature matrix has ~100 one-hot columns and can take a few
        hundred iterations to converge under `lbfgs`.

    Returns
    -------
    dict with keys:
        y_pred  : ndarray of 0/1 class predictions on X_test (threshold 0.5)
        y_score : ndarray of predicted probabilities of the positive class
        model   : fitted sklearn estimator
    """
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=max_iter,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return {"y_pred": y_pred, "y_score": y_score, "model": model}


# -----------------------------------------------------------------------------
# 2. Pooled elastic net WITH interactions but WITHOUT subgroup-size weighting
#    (weighting ablation)
# -----------------------------------------------------------------------------

def fit_pooled_en_interactions_no_weight(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    reference_class: int,
    alpha: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Fit pooled elastic-net logistic regression with group-covariate
    interactions but uniform (unit) observation weights.

    This is the subgroup-weighting ablation. It uses the same augmented
    feature vector as FAIR-PLR (interaction terms between the
    subgroup indicator and all covariates) but drops the subgroup-size
    weighting, so the only difference from FAIR-PLR is the observation
    weight vector. Comparing this baseline against FAIR-PLR directly
    isolates the contribution of the `w_i = n_min / n_{k(i)}` weighting
    scheme.

    The backend is the same R `cv.glmnet` wrapper (`FairElasticGlmNet`) as
    the main FAIR-PLR runs, invoked with `curr_n_k=None` (which internally
    defaults to a vector of ones).

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Covariate matrices (same features as the main FAIR-PLR run).
    y_train : np.ndarray
        Binary outcome.
    z_train, z_test : np.ndarray
        Subgroup labels (integer-encoded) for train and test.
    reference_class : int
        Index of the reference subgroup (typically the most prevalent one).
    alpha : float, default 0.5
        Elastic-net mixing parameter (0 = Ridge, 1 = Lasso, 0.5 = balanced).
        Matches the FAIR-PLR default so comparisons are apples-to-apples.

    Returns
    -------
    dict with keys y_pred, y_score, model (same contract as fit_standard_logreg).
    """
    # Lazy import so Python-only callers can import baselines.py without R.
    from model.logistic_glmnet import FairElasticGlmNet

    X_train_interact = z_interact_multi_group(X_train, z_train, reference_class)
    X_test_interact = z_interact_multi_group(X_test, z_test, reference_class)

    model = FairElasticGlmNet()
    model.fit(
        X_train_interact,
        y_train,
        curr_n_k=None,         # uniform weights -- the ablation
        penalty_factor=None,   # no differential penalization
        logistic=True,
        alpha=alpha,
    )
    y_score = model.predict_proba(X_test_interact)
    y_pred = (y_score >= 0.5).astype(int)
    return {"y_pred": y_pred, "y_score": y_score, "model": model}


# -----------------------------------------------------------------------------
# 3. Decision tree (CART), max_depth tuned by 5-fold CV
# -----------------------------------------------------------------------------

def fit_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cv_folds: int = 5,
    max_depth_grid: tuple = (3, 5, 7, 10, 15),
    random_state: int = 42,
    sample_weight: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Fit a CART decision tree with max_depth tuned via cross-validation.

    Representative inherently interpretable non-linear baseline.
    Uses AUC as the CV scoring criterion so the tuning objective matches
    the evaluation metric used throughout the paper.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices. Decision trees do not require standardization; the
        same encoded feature matrix used for the other models is fine.
    y_train : np.ndarray
        Binary outcome.
    cv_folds : int, default 5
        Number of cross-validation folds for max_depth selection.
    max_depth_grid : tuple of int, default (3, 5, 7, 10, 15)
        Candidate max_depth values searched via grid search.
    random_state : int, default 42
        For reproducibility of the decision-tree internal splitting.

    Returns
    -------
    dict with keys y_pred, y_score, model, best_max_depth.
    """
    base = DecisionTreeClassifier(random_state=random_state)
    grid = GridSearchCV(
        estimator=base,
        param_grid={"max_depth": list(max_depth_grid)},
        cv=cv_folds,
        scoring="roc_auc",
        n_jobs=-1,
    )
    _fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    grid.fit(X_train, y_train, **_fit_kwargs)
    best = grid.best_estimator_
    y_score = best.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "y_pred": y_pred,
        "y_score": y_score,
        "model": best,
        "best_max_depth": grid.best_params_["max_depth"],
    }
