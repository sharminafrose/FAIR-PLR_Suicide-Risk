"""Render combined-Yes+No cross-sectional supplementary figures (Supp Fig 3-7)
using the same legacy ML4H-style conventions as figure3_cross_sectional.py
(which generates main manuscript Figures 3 and 4).

Conventions (identical to figure3_cross_sectional.py):
  - Two stacked panels per figure: AUC-ROC (Overall) on top and
    TP% (Top 5% Risk) on bottom.
  - One color pair per subgroup level (light = Yes, slightly darker = No).
  - FAIR-PLR bar uses '.' hatch + light color; PLR bar uses '/' hatch +
    dark color. Edges and hatches in dark gray.
  - Black star marks the better-performing model within each subgroup.
  - Value labels above each bar, rotated 90°.
  - Legend explicit: "FAIR-PLR: Train All, Test <subgroup>:Yes" /
    "PLR: Train <subgroup>:Yes, Test <subgroup>:Yes" etc.

Difference from figure3_cross_sectional.py:
  - Combines :Yes and :No in a single figure (2 * n_levels groups), so
    the supplementary shows the full cross-sectional axis on one plot
    instead of split into two sub-figures.

Output:
  - npj_supplementary_NSDUH/low_res/{SEX_SPD,Age_SPD,Marital_status_SPD,
    Rurality_RCVDTreatment,BMI_SPD}.png
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_HERE = os.path.dirname(os.path.abspath(__file__))
_NSDUH_CODE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_NSDUH_CODE)
_METRICS_DIR = os.path.join(_NSDUH_CODE, "results", "metrics")
_SUPP_LOW = os.path.join(_REPO, "npj_supplementary_NSDUH", "low_res")


# ---------------------------------------------------------------------------
# Axis specifications — same level + color pairings as the main figure script.
# ---------------------------------------------------------------------------

_Axis = Tuple[str, str, str, List[str], List[str], Dict[str, str]]

_AXES: List[_Axis] = [
    (
        "Age_SPD",
        "Age:SPD",
        "FAIR_Age_SPD_l1_0.5.csv",
        ["18-25 Years Old", "26-34 Years Old", "35-49 Years Old",
         "50-64 Years Old", "65 or Older"],
        [
            "#b2df8a", "#33a02c",   # green
            "#fdbf6f", "#ff7f00",   # orange
            "#a6cee3", "#1f78b4",   # blue
            "#cab2d6", "#975fd4",   # purple
            "#fb9a99", "#df7173",   # red
        ],
        {
            "18-25 Years Old": "18-25Y",
            "26-34 Years Old": "26-34Y",
            "35-49 Years Old": "35-49Y",
            "50-64 Years Old": "50-64Y",
            "65 or Older": "65Y or Older",
        },
    ),
    (
        "BMI_SPD",
        "BMI:SPD",
        "FAIR_BMI_SPD_l1_0.5.csv",
        ["Underweight", "Healthy", "Overweight", "Obesity",
         "Severe Obesity", "Unknown"],
        [
            "#b2df8a", "#33a02c",
            "#fdbf6f", "#ff7f00",
            "#a6cee3", "#1f78b4",
            "#cab2d6", "#975fd4",
            "#fb9a99", "#df7173",
            "#ffffb3", "#9a962e",
        ],
        {},
    ),
    (
        "Marital_status_SPD",
        "Marital Status:SPD",
        "FAIR_Marital_Status_SPD_l1_0.5.csv",
        ["Married", "Widowed", "Divorced or Separated", "Never Been Married"],
        [
            "#b2df8a", "#33a02c",
            "#a6cee3", "#1f78b4",
            "#cab2d6", "#975fd4",
            "#ffffb3", "#9a962e",
        ],
        {},
    ),
    (
        "Rurality_RCVDTreatment",
        "Urban Residence:Treatment",
        "FAIR_Urban_Residence_Treatment_l1_0.5.csv",
        ["Large Metropolitan", "Nonmetropolitan", "Small Metropolitan"],
        [
            "#b2df8a", "#33a02c",
            "#a6cee3", "#1f78b4",
            "#cab2d6", "#975fd4",
        ],
        {},
    ),
    (
        "SEX_SPD",
        "Sex:SPD",
        "FAIR_Sex_SPD_l1_0.5.csv",
        ["Male", "Female"],
        [
            "#b2df8a", "#33a02c",
            "#fdbf6f", "#ff7f00",
        ],
        {},
    ),
]


# ---------------------------------------------------------------------------
# Data loading (identical to figure3_cross_sectional.py)
# ---------------------------------------------------------------------------


def _separate_prefix_for(fair_filename: str) -> str:
    return fair_filename.replace("FAIR_", "Separate_").replace("_l1_0.5.csv", "_")


def _load_fair(path: str, axis_group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["Group"] == axis_group].copy()


def _load_separate(metrics_dir: str, axis_group: str, sep_prefix: str) -> pd.DataFrame:
    pattern = os.path.join(metrics_dir, f"{sep_prefix}*_l1_0.5.csv")
    frames: List[pd.DataFrame] = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        df = df[df["Group"] == axis_group].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _value_for(df: pd.DataFrame, subset: str, group_value: str, column: str) -> Optional[float]:
    hit = df[(df["Subset"] == subset) & (df["Group Value"] == group_value)]
    if hit.empty:
        return None
    val = hit[column].iloc[0]
    return None if pd.isna(val) else float(val)


# ---------------------------------------------------------------------------
# Panel rendering (mirrors figure3_cross_sectional._draw_panel exactly,
# only the bar count changes because we feed both :Yes and :No levels)
# ---------------------------------------------------------------------------


def _abbreviate(level: str, abbrev: Dict[str, str]) -> str:
    return abbrev.get(level, level)


def _draw_panel(
    ax,
    values_fair: List[Optional[float]],
    values_plr: List[Optional[float]],
    full_labels: List[str],   # already ":Yes" / ":No" suffixed
    colors: List[str],        # length == 2 * len(full_labels)
    metric_label: str,
    value_fmt: str,
    ylim: Tuple[float, float],
    x_label: str,
    legend_train_labels: List[str],   # what comes after "Train " in legend
):
    n = len(full_labels)
    bar_w = 0.90 / (2.25 * n - 0.25) if n > 0 else 0.10
    pair_w = 2.0 * bar_w
    gap_w = 0.25 * bar_w
    total_w = n * pair_w + (n - 1) * gap_w
    x_start = 0.5 - total_w / 2.0
    centers = np.array([x_start + pair_w / 2.0 + i * (pair_w + gap_w) for i in range(n)])
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.5])
    ax.set_xticklabels([x_label], fontsize=16, fontweight="bold")

    hatch_fair = "."
    hatch_plr = "/"
    hatch_color = "#111111"

    legend_handles: list = []
    legend_texts: list = []

    for i, label in enumerate(full_labels):
        light = colors[2 * i]
        dark = colors[2 * i + 1]
        fair_val = values_fair[i]
        plr_val = values_plr[i]
        train_label = legend_train_labels[i]

        bar_fair = ax.bar(
            centers[i] - bar_w / 2,
            fair_val if fair_val is not None else 0.0,
            bar_w,
            color=light,
            edgecolor=hatch_color,
            linewidth=1.2,
            hatch=hatch_fair,
            label=f"FAIR-PLR: Train All, Test {label}",
        )
        bar_plr = ax.bar(
            centers[i] + bar_w / 2,
            plr_val if plr_val is not None else 0.0,
            bar_w,
            color=dark,
            edgecolor=hatch_color,
            linewidth=1.2,
            hatch=hatch_plr,
            label=f"PLR: Train {train_label}, Test {label}",
        )

        legend_handles.extend([bar_fair, bar_plr])
        legend_texts.extend([
            f"FAIR-PLR: Train All, Test {label}",
            f"PLR: Train {train_label}, Test {label}",
        ])

        label_pad = (ylim[1] - ylim[0]) * 0.02
        if fair_val is not None:
            ax.text(
                centers[i] - bar_w / 2,
                fair_val + label_pad,
                value_fmt.format(fair_val),
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", rotation=90,
            )
        if plr_val is not None:
            ax.text(
                centers[i] + bar_w / 2,
                plr_val + label_pad,
                value_fmt.format(plr_val),
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", rotation=90,
            )

        if fair_val is not None and plr_val is not None:
            star_x = centers[i] - bar_w / 2 if fair_val >= plr_val else centers[i] + bar_w / 2
            ax.text(
                star_x,
                (ylim[1] - ylim[0]) * 0.03,
                "★",
                ha="center", va="bottom",
                fontsize=28, color="#111111", fontweight="bold",
            )

    ax.set_ylim(*ylim)
    ax.set_ylabel(metric_label, fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    if metric_label == "AUC-ROC":
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax._legend_handles = legend_handles
    ax._legend_texts = legend_texts


# ---------------------------------------------------------------------------
# Figure builder — combined Yes + No
# ---------------------------------------------------------------------------


def _build_combined_figure(
    axis_name: str,
    axis_group: str,
    fair_filename: str,
    levels: List[str],
    colors: List[str],
    abbrev: Dict[str, str],
    out_paths: List[str],
) -> None:
    fair_df = _load_fair(os.path.join(_METRICS_DIR, fair_filename), axis_group)
    sep_df = _load_separate(_METRICS_DIR, axis_group, _separate_prefix_for(fair_filename))

    # Build the combined level list: all :Yes first, then all :No.
    full_labels: List[str] = []
    full_colors: List[str] = []
    legend_train_labels: List[str] = []
    auc_fair, auc_plr, tpp_fair, tpp_plr = [], [], [], []
    for suffix in (":Yes", ":No"):
        for i, level in enumerate(levels):
            ab = _abbreviate(level, abbrev)
            full_label = f"{ab}{suffix}"
            full_labels.append(full_label)
            full_colors.extend([colors[2 * i], colors[2 * i + 1]])
            legend_train_labels.append(full_label)
            gv = f"{level}{suffix}"
            auc_fair.append(_value_for(fair_df, "Overall", gv, "AUC-ROC"))
            auc_plr.append(_value_for(sep_df, "Overall", gv, "AUC-ROC"))
            tpp_fair.append(_value_for(fair_df, "Top 5% Risk", gv, "TP %"))
            tpp_plr.append(_value_for(sep_df, "Top 5% Risk", gv, "TP %"))

    # Drop levels with no data anywhere
    keep_idx = [
        i for i in range(len(full_labels))
        if not (auc_fair[i] is None and auc_plr[i] is None
                and tpp_fair[i] is None and tpp_plr[i] is None)
    ]
    full_labels_k = [full_labels[i] for i in keep_idx]
    full_colors_k: List[str] = []
    for i in keep_idx:
        full_colors_k.extend([full_colors[2 * i], full_colors[2 * i + 1]])
    legend_train_labels_k = [legend_train_labels[i] for i in keep_idx]
    auc_fair_k = [auc_fair[i] for i in keep_idx]
    auc_plr_k = [auc_plr[i] for i in keep_idx]
    tpp_fair_k = [tpp_fair[i] for i in keep_idx]
    tpp_plr_k = [tpp_plr[i] for i in keep_idx]

    n = len(full_labels_k)
    # Layout: legend BAND on top spanning the full width, a SPACER row,
    # then two stacked metric panels. Legend entry count is 2*n (FAIR +
    # PLR rows per subgroup) so we choose ncol/height to keep it compact.
    n_entries = 2 * n
    if n_entries <= 8:
        legend_ncol = 2
    elif n_entries <= 16:
        legend_ncol = 3
    else:
        legend_ncol = 4
    legend_rows = -(-n_entries // legend_ncol)  # ceil-div
    legend_band = 0.06 * legend_rows + 0.4      # inches per row + padding
    spacer_band = 0.6                            # inches between legend & panels

    fig_w = max(14.0, 1.6 + 1.0 * n)
    fig_h = 10.5 + legend_band + spacer_band
    fig = plt.figure(figsize=(fig_w, fig_h))
    legend_frac = legend_band / fig_h
    spacer_frac = spacer_band / fig_h
    panel_frac = (1 - legend_frac - spacer_frac) / 2.0
    gs = fig.add_gridspec(
        4, 1,
        height_ratios=[legend_frac, spacer_frac, panel_frac, panel_frac],
        left=0.06, right=0.985,
        top=0.985, bottom=0.07,
        hspace=0.30,
    )
    legend_ax = fig.add_subplot(gs[0, 0])
    spacer_ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[2, 0])
    ax_bot = fig.add_subplot(gs[3, 0])
    legend_ax.axis("off")
    spacer_ax.axis("off")

    _draw_panel(
        ax_top,
        auc_fair_k, auc_plr_k, full_labels_k, full_colors_k,
        metric_label="AUC-ROC",
        value_fmt="{:.3f}",
        ylim=(0.0, 1.25),
        x_label="Overall",
        legend_train_labels=legend_train_labels_k,
    )
    finite_tpp = [v for v in tpp_fair_k + tpp_plr_k if v is not None]
    tpp_upper = max(30.0, max(finite_tpp) * 1.65) if finite_tpp else 30.0
    _draw_panel(
        ax_bot,
        tpp_fair_k, tpp_plr_k, full_labels_k, full_colors_k,
        metric_label="TP %",
        value_fmt="{:.2f}",
        ylim=(0.0, tpp_upper),
        x_label="Top 5% Risk",
        legend_train_labels=legend_train_labels_k,
    )

    handles = getattr(ax_top, "_legend_handles", [])
    texts = getattr(ax_top, "_legend_texts", [])
    if handles:
        # Font sizes tuned for the on-top legend band.
        fontsize = (
            16 if n_entries <= 8
            else 13 if n_entries <= 16
            else 11
        )
        legend_ax.legend(
            handles, texts,
            loc="center", ncol=legend_ncol, frameon=True,
            fontsize=fontsize,
            handlelength=2.0, handleheight=1.2,
            labelspacing=0.4, borderpad=0.6,
            columnspacing=1.5,
        )

    for out in out_paths:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        # Lower DPI keeps the supplementary PDF small (250 -> 120 cuts file
        # size by ~3x while remaining sharp at the embedded 6.5 in width).
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"[ok] {out}")
    plt.close(fig)


def render_all() -> None:
    for axis_name, axis_group, fair_filename, levels, colors, abbrev in _AXES:
        out_paths = [os.path.join(_SUPP_LOW, f"{axis_name}.png")]
        _build_combined_figure(
            axis_name, axis_group, fair_filename, levels, colors, abbrev, out_paths,
        )


if __name__ == "__main__":
    render_all()
