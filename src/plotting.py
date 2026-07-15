"""Grouped bar-chart figure generator for FAIR-PLR subgroup performance.

Renders two-color comparison panels for subgroup performance. Key design
choices:

  - Subgroups on the x-axis (the thing being compared).
  - Only two colors per panel: FAIR-PLR (always trained on all data) vs
    Separate-PLR (trained on the matching subgroup).
  - Separate subpanels for SPD=Yes and SPD=No (or Treatment=Yes/No) where
    the cross-sectional design applies.
  - Caption explicitly states the training-data difference between the two
    color groups.

The module is data-source agnostic: it reads result CSVs produced by the
existing pipeline (`Result_FAIR_*.csv`, `Result_PLR_*.csv`) and produces
PNG figures in `figures/main/` and `figures/supplementary/`.

Usage
-----
    # Regenerate one figure
    python -m src.plotting --input results/metrics/ --output figures/supplementary/ --subgroup sex_spd

    # Regenerate all eleven figures
    python -m src.plotting --input results/metrics/ --output figures/
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Color palette: only two model colors, deliberately kept simple
# -----------------------------------------------------------------------------

_COLOR_FAIR = "#1f77b4"        # blue
_COLOR_SEPARATE = "#d62728"    # red
_COLOR_AGNOSTIC = "#2ca02c"    # green (reserved for Supp Table 2 comparison)


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------

def load_subgroup_results(
    metrics_dir: str,
    subgroup_name: str,
) -> pd.DataFrame:
    """Load the tidy long-format results CSV for a given subgroup axis.

    Expected schema (long format):
        subgroup : str   -- subgroup label (e.g. "Male:Yes", "Female:No")
        model    : str   -- "FAIR-PLR" or "Separate-PLR"
        metric   : str   -- one of {auc, tpr_top1, tpr_top5, precision_top1, precision_top5}
        point    : float -- point estimate
        ci_low   : float -- 95% CI lower (from stratified bootstrap)
        ci_high  : float -- 95% CI upper

    If the path does not exist or the file is empty, returns an empty DataFrame
    so the plotting helper can gracefully skip that subgroup.
    """
    path = os.path.join(metrics_dir, f"subgroup_results_{subgroup_name}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# Grouped bar chart (the core redesign)
# -----------------------------------------------------------------------------

def grouped_bar_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    subpanel_splitter: Optional[str] = None,
    models: Iterable[str] = ("FAIR-PLR", "Separate-PLR"),
    colors: Optional[Dict[str, str]] = None,
    ax=None,
) -> plt.Figure:
    """Render a grouped bar chart with subgroups on the x-axis.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format results DataFrame (see load_subgroup_results).
    metric : str
        Which metric to plot (must match a value in df['metric']).
    title, ylabel : str
        Axis labels.
    subpanel_splitter : str, optional
        A substring by which to split subgroup labels into two subpanels
        (e.g. ":Yes" vs ":No" for cross-sectional SPD groupings). If None,
        a single panel is drawn with all subgroups on the x-axis.
    models : iterable of str, default ("FAIR-PLR", "Separate-PLR")
        Model names to plot. First name is drawn first (leftmost within
        each group).
    colors : dict, optional
        Override the default per-model colors.
    ax : matplotlib Axes, optional
        If provided, draw into this axes rather than creating a new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = df[df["metric"] == metric].copy()
    if df.empty:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    color_map = {
        "FAIR-PLR": _COLOR_FAIR,
        "Separate-PLR": _COLOR_SEPARATE,
        "Agnostic-PLR": _COLOR_AGNOSTIC,
    }
    if colors:
        color_map.update(colors)

    if subpanel_splitter is not None:
        # Split subgroups into two panels based on the splitter substring
        yes_mask = df["subgroup"].str.contains(f"{subpanel_splitter}Yes", case=False)
        no_mask = df["subgroup"].str.contains(f"{subpanel_splitter}No", case=False)
        df_yes = df[yes_mask]
        df_no = df[no_mask]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        _draw_panel(axes[0], df_yes, metric, models, color_map, f"{title} (Yes)", ylabel)
        _draw_panel(axes[1], df_no, metric, models, color_map, f"{title} (No)", ylabel=None)
        axes[0].legend(frameon=False, loc="upper right", fontsize=9)
        fig.tight_layout()
        return fig

    # Single-panel path
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    _draw_panel(ax, df, metric, models, color_map, title, ylabel)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax.figure


def _draw_panel(ax, df_panel, metric, models, color_map, title, ylabel):
    """Render a single grouped-bar panel into a matplotlib Axes."""
    subgroups = sorted(df_panel["subgroup"].unique().tolist())
    n_sub = len(subgroups)
    n_mod = len(list(models))
    if n_sub == 0 or n_mod == 0:
        return

    bar_width = 0.8 / max(n_mod, 1)
    x = np.arange(n_sub)
    for i, model in enumerate(models):
        sub = df_panel[df_panel["model"] == model].set_index("subgroup")
        # Align rows with the subgroup ordering above
        vals = np.array([sub.loc[s, "point"] if s in sub.index else np.nan for s in subgroups])
        # CI error bars, if present
        if "ci_low" in sub.columns and "ci_high" in sub.columns:
            yerr_low = np.array([vals[j] - sub.loc[s, "ci_low"] if s in sub.index else 0 for j, s in enumerate(subgroups)])
            yerr_high = np.array([sub.loc[s, "ci_high"] - vals[j] if s in sub.index else 0 for j, s in enumerate(subgroups)])
            yerr = np.vstack([np.maximum(yerr_low, 0), np.maximum(yerr_high, 0)])
        else:
            yerr = None
        offset = (i - (n_mod - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width,
            yerr=yerr,
            capsize=2.5,
            color=color_map.get(model, "#888888"),
            label=model,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, rotation=25, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)


# -----------------------------------------------------------------------------
# Figure bundle: regenerate all eleven subgroup-performance figures
# -----------------------------------------------------------------------------

_FIGURE_SPECS = [
    # (filename_stem,               subgroup_csv_stem,          subpanel_splitter, dest_dir,          main_title)
    ("main_sex_spd",                "sex_spd",                  ":",              "main",            "Sex x SPD"),
    ("main_age_spd",                "age_spd",                  ":",              "main",            "Age x SPD"),
    ("main_marital_spd",            "marital_spd",              ":",              "main",            "Marital Status x SPD"),
    ("main_bmi_spd",                "bmi_spd",                  ":",              "main",            "BMI x SPD"),
    ("main_rurality_treatment",     "rurality_treatment",       ":",              "main",            "Rurality x Treatment"),
    ("supp_age",                    "age",                      None,             "supplementary",   "Age"),
    ("supp_sex",                    "sex",                      None,             "supplementary",   "Sex"),
    ("supp_race",                   "race",                     None,             "supplementary",   "Race/Ethnicity"),
    ("supp_bmi",                    "bmi",                      None,             "supplementary",   "BMI"),
    ("supp_insurance",              "insurance",                None,             "supplementary",   "Health Insurance"),
    ("supp_rurality",               "rurality",                 None,             "supplementary",   "Rurality"),
]


def regenerate_all_figures(
    metrics_dir: str,
    figures_root: str,
    metric: str = "tpr_top5",
    ylabel: str = "TPR (top 5%)",
) -> None:
    """Regenerate all eleven subgroup-performance figures for one metric.

    Writes PNG files into <figures_root>/main/ and <figures_root>/supplementary/
    using the _FIGURE_SPECS registry above.

    Parameters
    ----------
    metrics_dir : str
        Directory holding subgroup_results_<stem>.csv files.
    figures_root : str
        Root output directory. Subdirectories main/ and supplementary/ are
        created if they don't already exist.
    metric : str
        Which metric to plot across all eleven figures (one PNG per figure per
        call). Loop over metrics to produce the full set.
    """
    os.makedirs(os.path.join(figures_root, "main"), exist_ok=True)
    os.makedirs(os.path.join(figures_root, "supplementary"), exist_ok=True)

    for stem, csv_stem, splitter, dest, title in _FIGURE_SPECS:
        df = load_subgroup_results(metrics_dir, csv_stem)
        if df.empty:
            print(f"[skip] {stem}: no data at {csv_stem}")
            continue
        fig = grouped_bar_chart(
            df,
            metric=metric,
            title=title,
            ylabel=ylabel,
            subpanel_splitter=splitter,
        )
        out = os.path.join(figures_root, dest, f"{stem}_{metric}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok]   {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Directory containing subgroup_results_<stem>.csv files")
    ap.add_argument("--output", required=True,
                    help="Destination figures/ directory")
    ap.add_argument("--metric", default="tpr_top5",
                    help="Metric to plot (default: tpr_top5)")
    ap.add_argument("--ylabel", default="TPR (top 5%)",
                    help="Y-axis label (default: 'TPR (top 5%)')")
    args = ap.parse_args()

    regenerate_all_figures(
        metrics_dir=args.input,
        figures_root=args.output,
        metric=args.metric,
        ylabel=args.ylabel,
    )


if __name__ == "__main__":
    main()
