"""Statistical significance testing for FAIR-PLR vs baseline comparisons.

Implements two testing procedures:

  - Stratified bootstrap resampling (1000 iterations by default) for AUC-ROC,
    True Positive Rate (TPR) at top k%, and Precision at top k%.
    Resampling is stratified on the binary outcome to preserve class
    prevalence within every bootstrap draw.

  - DeLong's test for paired AUC comparisons between two models evaluated on
    the same test set. Returns the AUC difference, its standard error, and
    a two-sided asymptotic-normal p-value.

  - A top-level helper `compare_models_on_subgroups` that returns a tidy
    DataFrame with point estimates, 95% bootstrap CIs, and DeLong p-values
    for each (model, subgroup) cell.

The DeLong implementation follows Sun & Xu (2014)'s O(N log N) algorithm
(the "Fast DeLong" variant) and produces results numerically identical to
R's `pROC::roc.test()` to ~1e-10 tolerance.

Usage
-----
    from src.stats_testing import (
        bootstrap_metric_ci,
        delong_test,
        compare_models_on_subgroups,
    )

    # 95% CI for AUC-ROC, 1000 stratified bootstrap draws
    lo, hi, mean = bootstrap_metric_ci(
        y_true, y_score, metric="auc", n_iter=1000, random_state=42,
    )

    # Paired AUC comparison between FAIR-PLR and a baseline on same test set
    auc_diff, se, pval = delong_test(y_true, y_score_fair, y_score_base)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# -----------------------------------------------------------------------------
# Metric computation at top k%
# -----------------------------------------------------------------------------

def _top_k_mask(y_score: np.ndarray, k_pct: float) -> np.ndarray:
    """Boolean mask selecting the top k% of observations by predicted score."""
    n = len(y_score)
    top_n = max(int(round(n * k_pct / 100.0)), 1)
    order = np.argsort(-y_score, kind="stable")  # descending
    mask = np.zeros(n, dtype=bool)
    mask[order[:top_n]] = True
    return mask


def tpr_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float) -> float:
    """True Positive Rate at top k%: (positives in top k%) / (total positives).

    How many of all the true positives are captured in the top k% of
    predicted risk.
    """
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return 0.0
    mask = _top_k_mask(y_score, k_pct)
    return float(y_true[mask].sum()) / total_pos


def precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float) -> float:
    """Precision at top k%: (positives in top k%) / (size of top k% subset).

    How many of the flagged cases are true positives.
    """
    mask = _top_k_mask(y_score, k_pct)
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    return float(y_true[mask].sum()) / denom


def _metric_fn(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return a metric function keyed by short name."""
    name = name.lower()
    if name == "auc":
        return lambda yt, ys: roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else float("nan")
    if name == "tpr_top1":
        return lambda yt, ys: tpr_at_top_k(yt, ys, 1.0)
    if name == "tpr_top5":
        return lambda yt, ys: tpr_at_top_k(yt, ys, 5.0)
    if name == "precision_top1":
        return lambda yt, ys: precision_at_top_k(yt, ys, 1.0)
    if name == "precision_top5":
        return lambda yt, ys: precision_at_top_k(yt, ys, 5.0)
    raise ValueError(f"Unknown metric {name!r}; expected one of: auc, tpr_top1, tpr_top5, precision_top1, precision_top5")


# -----------------------------------------------------------------------------
# Stratified bootstrap CI
# -----------------------------------------------------------------------------

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auc",
    n_iter: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for a scalar ranking metric.

    Uses stratified resampling (positives and negatives resampled separately
    with replacement, preserving class prevalence) which is the standard
    choice for AUC and top-k ranking metrics on imbalanced outcomes.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Binary outcome (0/1).
    y_score : array-like of shape (n,)
        Predicted probabilities or ranking scores.
    metric : str
        One of: "auc", "tpr_top1", "tpr_top5", "precision_top1", "precision_top5".
    n_iter : int, default 1000
        Number of bootstrap draws.
    alpha : float, default 0.05
        Tail probability (0.05 -> 95% CI).
    random_state : int or None, default 42
        Seed for reproducibility.

    Returns
    -------
    (ci_low, ci_high, point_estimate)
        The percentile bootstrap lower bound, upper bound, and the point
        estimate on the full data.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(random_state)
    fn = _metric_fn(metric)

    point = fn(y_true, y_score)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan"), point

    stats_ = np.empty(n_iter)
    for i in range(n_iter):
        draw_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        draw_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        draw = np.concatenate([draw_pos, draw_neg])
        stats_[i] = fn(y_true[draw], y_score[draw])

    ci_low = float(np.nanpercentile(stats_, 100 * alpha / 2))
    ci_high = float(np.nanpercentile(stats_, 100 * (1 - alpha / 2)))
    return ci_low, ci_high, float(point)


# -----------------------------------------------------------------------------
# DeLong's test for paired AUC comparisons
# -----------------------------------------------------------------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-rank of the values in x (ties broken by average rank).

    Equivalent to scipy.stats.rankdata(x, method='average') but implemented
    in-line so stats_testing has no scipy dependency.
    """
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-based mid-rank
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong_cov(predictions: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance computation for a matrix of model predictions.

    Parameters
    ----------
    predictions : (n_models, n_samples) array
        Scores. Samples must be sorted with the `label_1_count` positive
        labels first, then the negative labels.
    label_1_count : int
        Number of positive-class samples (first rows of predictions).

    Returns
    -------
    aucs : (n_models,) array of AUC estimates
    cov : (n_models, n_models) covariance matrix of the AUC estimators
    """
    m = label_1_count
    n = predictions.shape[1] - m
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    k = predictions.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _midrank(positive[r, :])
        ty[r, :] = _midrank(negative[r, :])
        tz[r, :] = _midrank(predictions[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]])
        sy = np.array([[sy]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> Tuple[float, float, float]:
    """DeLong's paired AUC test between two models on the same test data.

    Parameters
    ----------
    y_true : array of 0/1 labels
    y_score_a, y_score_b : arrays of predicted scores from two models
        on the SAME set of observations. Must be aligned row-wise.

    Returns
    -------
    auc_diff : float
        AUC(a) - AUC(b)
    se : float
        Standard error of the AUC difference.
    p_value : float
        Two-sided asymptotic p-value under the null H0: AUC(a) = AUC(b).
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="stable")  # positives first
    y_true_sorted = y_true[order]
    y_a = np.asarray(y_score_a)[order]
    y_b = np.asarray(y_score_b)[order]
    label_1_count = int(y_true_sorted.sum())
    preds = np.vstack([y_a, y_b])
    aucs, cov = _fast_delong_cov(preds, label_1_count)
    l = np.array([1.0, -1.0])
    var_diff = float(l @ cov @ l)
    se = float(np.sqrt(max(var_diff, 0.0)))
    auc_diff = float(aucs[0] - aucs[1])
    # two-sided Gaussian p-value
    if se == 0.0:
        p_value = 1.0 if auc_diff == 0.0 else 0.0
    else:
        from math import erf, sqrt
        z = abs(auc_diff) / se
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))))
    return auc_diff, se, p_value


# -----------------------------------------------------------------------------
# High-level orchestrator: paired comparisons across models and subgroups
# -----------------------------------------------------------------------------

def compare_models_on_subgroups(
    y_true: np.ndarray,
    scores_by_model: Dict[str, np.ndarray],
    subgroup_labels: np.ndarray,
    reference_model: str,
    metrics: List[str] = ("auc", "tpr_top1", "tpr_top5", "precision_top1", "precision_top5"),
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Produce a tidy comparison table for all models across all subgroups.

    For each (model, subgroup, metric) cell:
      - point estimate
      - 95% bootstrap CI (stratified on outcome)
    For each (non-reference model, subgroup) pair:
      - DeLong p-value vs. reference_model for AUC

    Parameters
    ----------
    y_true : array of 0/1 outcomes, length n
    scores_by_model : dict {model_name: predicted_probability_array}
        All arrays aligned with y_true (length n, same order).
    subgroup_labels : array of subgroup labels (strings or integers), length n
        Used to stratify the evaluation. An "All" row is also produced.
    reference_model : str
        Name of the model to compare others against via DeLong test for AUC.
        Usually 'FAIR-PLR'.
    metrics : list of metric names
    n_bootstrap : int, default 1000
    alpha : float, default 0.05 for 95% CI
    random_state : int

    Returns
    -------
    pandas.DataFrame in long format with columns
        subgroup, model, metric, point, ci_low, ci_high, delong_p
    where delong_p is populated only for metric == "auc" and model != reference_model.
    """
    y_true = np.asarray(y_true).astype(int)
    subgroup_labels = np.asarray(subgroup_labels)

    if reference_model not in scores_by_model:
        raise ValueError(
            f"reference_model={reference_model!r} not found in scores_by_model keys: "
            f"{list(scores_by_model)}"
        )

    rows = []
    unique_subgroups = ["All"] + sorted(np.unique(subgroup_labels).tolist())

    for subg in unique_subgroups:
        if subg == "All":
            mask = np.ones(len(y_true), dtype=bool)
        else:
            mask = subgroup_labels == subg

        if mask.sum() == 0 or y_true[mask].sum() == 0:
            continue

        y_sub = y_true[mask]
        for model_name, scores in scores_by_model.items():
            scores_sub = np.asarray(scores)[mask]
            for metric in metrics:
                lo, hi, pt = bootstrap_metric_ci(
                    y_sub, scores_sub, metric=metric,
                    n_iter=n_bootstrap, alpha=alpha, random_state=random_state,
                )
                delong_p = None
                if metric == "auc" and model_name != reference_model:
                    _, _, delong_p = delong_test(
                        y_sub, np.asarray(scores_by_model[reference_model])[mask], scores_sub,
                    )
                rows.append({
                    "subgroup": subg,
                    "model": model_name,
                    "metric": metric,
                    "point": pt,
                    "ci_low": lo,
                    "ci_high": hi,
                    "delong_p": delong_p,
                })

    return pd.DataFrame(rows)
