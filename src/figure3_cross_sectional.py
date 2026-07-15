"""Render cross-sectional performance figures (manuscript Figure 3 and its
No-only counterpart) in the legacy ML4H-style layout:

  - Two stacked panels per figure: AUC-ROC (Overall) on top and
    TP% (Top 5% Risk) on bottom.
  - One color pair per subgroup (light / dark), with hatch `o` for the
    FAIR-PLR bar (``Train All, Test <subgroup>``) and hatch `/` for the
    corresponding Separate PLR bar (``Train <subgroup>, Test <subgroup>``).
  - A black star marks the better-performing model within each subgroup.
  - Legend uses the full ``FAIR-PLR: Train ... / PLR: Train ...`` description
    so the training / testing distinction is explicit, matching the original
    figure pattern.

The script loads the wide FAIR / Separate metric CSVs under
``results/metrics/`` and writes PNGs into ``npj_NSDUH/low_res/`` and
``npj_NSDUH/high_res/`` under the legacy manuscript filenames so the
existing LaTeX ``\\includegraphics`` calls keep working.
"""

from __future__ import annotations

import glob
import os
from itertools import cycle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_HERE = os.path.dirname(os.path.abspath(__file__))
_NSDUH_CODE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_NSDUH_CODE)
_METRICS_DIR = os.path.join(_NSDUH_CODE, "results", "metrics")
_MAN_LOW = os.path.join(_REPO, "npj_NSDUH", "low_res")
_MAN_HIGH = os.path.join(_REPO, "npj_NSDUH", "high_res")


# ---------------------------------------------------------------------------
# Axis specifications
# ---------------------------------------------------------------------------
# Each entry describes one cross-sectional axis figure. ``group_value`` is the
# label that appears in the FAIR CSV's ``Group`` column. ``levels`` is the
# canonical ordering of subgroup levels used to assign colors (before
# suffixing with :Yes / :No). ``colors`` is a flat list with one light / dark
# pair per level (len == 2 * len(levels)). ``abbrev`` rewrites some long
# subgroup labels so they fit in the legend.

_Axis = Tuple[str, str, str, List[str], List[str], Dict[str, str]]

_AXES: List[_Axis] = [
    (
        "Age_SPD",
        "Age:SPD",
        "FAIR_Age_SPD_l1_0.5.csv",
        ["18-25 Years Old", "26-34 Years Old", "35-49 Years Old", "50-64 Years Old", "65 or Older"],
        [
            "#b2df8a", "#33a02c",   # light/dark green
            "#fdbf6f", "#ff7f00",   # light/dark orange
            "#a6cee3", "#1f78b4",   # light/dark blue
            "#cab2d6", "#975fd4",   # light/dark purple
            "#fb9a99", "#df7173",   # light/dark red
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
        ["Underweight", "Healthy", "Overweight", "Obesity", "Severe Obesity", "Unknown"],
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
# Data loading
# ---------------------------------------------------------------------------


def _separate_prefix_for(fair_filename: str) -> str:
    """Map a FAIR CSV filename to the matching Separate CSV prefix."""
    stem = fair_filename.replace("FAIR_", "Separate_").replace("_l1_0.5.csv", "_")
    return stem


def _load_fair(path: str, axis_group: str) -> pd.DataFrame:
    """Return the FAIR rows for the requested axis (one row per subset+level)."""
    df = pd.read_csv(path)
    df = df[df["Group"] == axis_group].copy()
    df["model"] = "FAIR"
    return df


def _load_separate(metrics_dir: str, axis_group: str, sep_prefix: str) -> pd.DataFrame:
    """Stack every Separate-<level> CSV for the axis into one frame."""
    pattern = os.path.join(metrics_dir, f"{sep_prefix}*_l1_0.5.csv")
    frames: List[pd.DataFrame] = []
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        df = df[df["Group"] == axis_group].copy()
        df["model"] = "PLR"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _value_for(df: pd.DataFrame, subset: str, group_value: str, column: str) -> Optional[float]:
    hit = df[(df["Subset"] == subset) & (df["Group Value"] == group_value)]
    if hit.empty:
        return None
    val = hit[column].iloc[0]
    if pd.isna(val):
        return None
    return float(val)


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------


def _abbreviate(level: str, abbrev: Dict[str, str]) -> str:
    return abbrev.get(level, level)


def _draw_panel(
    ax,
    values_fair: List[Optional[float]],
    values_plr: List[Optional[float]],
    levels: List[str],
    suffix: str,
    colors: List[str],
    abbrev: Dict[str, str],
    metric_label: str,
    value_fmt: str,
    ylim: Tuple[float, float],
    x_label: str,
    include_legend: bool,
    legend_col_labels: Optional[List[str]] = None,
):
    """Render one panel with per-subgroup color pairs and FAIR/PLR hatches.

    Bars are ordered so that within each subgroup the FAIR bar comes first,
    immediately followed by the PLR bar. Groups are separated by a small
    gap so color pairs stay visually coherent.
    """
    n = len(levels)
    # Layout: every bar uses ~90% of the panel width centered on 0.5,
    # so pairs cluster sensibly for small n and scale down for large n.
    # bar_w solves: n * (2 * bar_w) + (n-1) * (0.25 * bar_w) = 0.90
    if n > 0:
        bar_w = 0.90 / (2.25 * n - 0.25)
    else:
        bar_w = 0.10
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

    legend_handles = []
    legend_texts = []

    for i, level in enumerate(levels):
        light = colors[2 * i]
        dark = colors[2 * i + 1]
        fair_val = values_fair[i]
        plr_val = values_plr[i]
        abbrev_level = _abbreviate(level, abbrev)

        # FAIR bar (light color, 'o' hatch)
        bar_fair = ax.bar(
            centers[i] - bar_w / 2,
            fair_val if fair_val is not None else 0.0,
            bar_w,
            color=light,
            edgecolor=hatch_color,
            linewidth=1.2,
            hatch=hatch_fair,
            label=f"FAIR-PLR: Train All, Test {abbrev_level}{suffix}",
        )
        # PLR bar (dark color, '/' hatch)
        bar_plr = ax.bar(
            centers[i] + bar_w / 2,
            plr_val if plr_val is not None else 0.0,
            bar_w,
            color=dark,
            edgecolor=hatch_color,
            linewidth=1.2,
            hatch=hatch_plr,
            label=f"PLR: Train {abbrev_level}{suffix}, Test {abbrev_level}{suffix}",
        )

        legend_handles.extend([bar_fair, bar_plr])
        legend_texts.extend([
            f"FAIR-PLR: Train All, Test {abbrev_level}{suffix}",
            f"PLR: Train {abbrev_level}{suffix}, Test {abbrev_level}{suffix}",
        ])

        # Value labels above each bar (rotated vertical, with padding gap
        # between the bar top and the first character of the label).
        label_pad = (ylim[1] - ylim[0]) * 0.02
        if fair_val is not None:
            ax.text(
                centers[i] - bar_w / 2,
                fair_val + label_pad,
                value_fmt.format(fair_val),
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                rotation=90,
            )
        if plr_val is not None:
            ax.text(
                centers[i] + bar_w / 2,
                plr_val + label_pad,
                value_fmt.format(plr_val),
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                rotation=90,
            )

        # Star on the better-performing model within this subgroup (placed
        # near the base of the bar, just above the x-axis).
        if fair_val is not None and plr_val is not None:
            if fair_val >= plr_val:
                star_x = centers[i] - bar_w / 2
            else:
                star_x = centers[i] + bar_w / 2
            ax.text(
                star_x,
                (ylim[1] - ylim[0]) * 0.03,
                "★",
                ha="center",
                va="bottom",
                fontsize=28,
                color="#111111",
                fontweight="bold",
            )

    ax.set_ylim(*ylim)
    ax.set_ylabel(metric_label, fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=14)
    # For AUC-ROC clamp tick labels to [0, 1.0] (still leave ylim headroom
    # so rotated value labels don't clip at the top).
    if metric_label == "AUC-ROC":
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # Legend is placed at the figure level by the caller (not per-axes)
    # so it sits at the top spanning both panels. We return the collected
    # handles/texts via attributes on the axes for the caller to read.
    ax._legend_handles = legend_handles
    ax._legend_texts = legend_texts


def _build_figure(
    axis_name: str,
    axis_group: str,
    fair_filename: str,
    levels: List[str],
    colors: List[str],
    abbrev: Dict[str, str],
    suffix: str,  # ":Yes" or ":No"
    out_paths: List[str],
    legend_pos: str = "right",  # "top" or "right"
) -> None:
    fair_df = _load_fair(os.path.join(_METRICS_DIR, fair_filename), axis_group)
    sep_df = _load_separate(_METRICS_DIR, axis_group, _separate_prefix_for(fair_filename))

    # Collect per-level values for AUC-ROC (Overall) and TP % (Top 5% Risk)
    auc_fair, auc_plr = [], []
    tpp_fair, tpp_plr = [], []
    for level in levels:
        gv = f"{level}{suffix}"
        auc_fair.append(_value_for(fair_df, "Overall", gv, "AUC-ROC"))
        auc_plr.append(_value_for(sep_df, "Overall", gv, "AUC-ROC"))
        tpp_fair.append(_value_for(fair_df, "Top 5% Risk", gv, "TP %"))
        tpp_plr.append(_value_for(sep_df, "Top 5% Risk", gv, "TP %"))

    # Skip levels that have no data at all (e.g., zero-cell strata)
    keep_idx = [
        i for i in range(len(levels))
        if not (auc_fair[i] is None and auc_plr[i] is None
                and tpp_fair[i] is None and tpp_plr[i] is None)
    ]
    levels_k = [levels[i] for i in keep_idx]
    colors_k: List[str] = []
    for i in keep_idx:
        colors_k.extend([colors[2 * i], colors[2 * i + 1]])
    auc_fair_k = [auc_fair[i] for i in keep_idx]
    auc_plr_k = [auc_plr[i] for i in keep_idx]
    tpp_fair_k = [tpp_fair[i] for i in keep_idx]
    tpp_plr_k = [tpp_plr[i] for i in keep_idx]

    # Layout depends on legend_pos. Both variants keep fonts large so the
    # figure reads well after scaling down to its textwidth slot.
    #   * "right" (landscape): legend sits in a wide right column; used for
    #     the larger row-2 sub-figures (Age, BMI) that have 5-6 subgroups.
    #   * "top" (near-square): legend spans the top above two stacked
    #     panels; used for the smaller row-1 sub-figures (Marital,
    #     Rurality, Sex) where a tall-ish aspect packs better.
    n = len(levels_k)
    if legend_pos == "top":
        fig_w = 8.0
        fig_h = 10.0
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            3, 1,
            height_ratios=[0.9, 1.0, 1.0],
            left=0.13, right=0.97,
            top=0.98, bottom=0.07,
            hspace=0.30,
        )
        legend_ax = fig.add_subplot(gs[0, 0])
        ax_top = fig.add_subplot(gs[1, 0])
        ax_bot = fig.add_subplot(gs[2, 0])
    else:  # "right"
        fig_w = 14.0
        fig_h = 11.5
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            2, 2,
            width_ratios=[1.6, 1.0],
            height_ratios=[1.0, 1.0],
            left=0.05, right=0.985,
            top=0.95, bottom=0.09,
            wspace=0.22, hspace=0.30,
        )
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0])
        legend_ax = fig.add_subplot(gs[:, 1])
    legend_ax.axis("off")

    # AUC-ROC panel (extra headroom for vertical value labels and stars)
    _draw_panel(
        ax_top,
        auc_fair_k, auc_plr_k,
        levels_k, suffix, colors_k, abbrev,
        metric_label="AUC-ROC",
        value_fmt="{:.3f}",
        ylim=(0.0, 1.25),
        x_label="Overall",
        include_legend=True,
    )

    # TP % panel
    finite_tpp = [v for v in tpp_fair_k + tpp_plr_k if v is not None]
    if finite_tpp:
        max_tpp = max(finite_tpp)
        tpp_upper = max(30.0, max_tpp * 1.65)
    else:
        tpp_upper = 30.0
    tpp_ylim = (0.0, tpp_upper)

    _draw_panel(
        ax_bot,
        tpp_fair_k, tpp_plr_k,
        levels_k, suffix, colors_k, abbrev,
        metric_label="TP %",
        value_fmt="{:.2f}",
        ylim=tpp_ylim,
        x_label="Top 5% Risk",
        include_legend=False,
    )

    # Legend drawn inside the dedicated legend_ax.
    handles = getattr(ax_top, "_legend_handles", [])
    texts = getattr(ax_top, "_legend_texts", [])
    if handles:
        legend_rows = len(handles)
        if legend_pos == "top":
            # One legend entry per row. Font size eases down a touch for
            # axes with more entries so 8 rows still fit in the top band.
            ncol = 1
            legend_fontsize = (
                16 if legend_rows <= 4
                else 14 if legend_rows <= 6
                else 13
            )
        else:  # "right"
            ncol = 1
            legend_fontsize = (
                26 if legend_rows <= 4
                else 23 if legend_rows <= 6
                else 21 if legend_rows <= 8
                else 19 if legend_rows <= 10
                else 17
            )
        legend_ax.legend(
            handles,
            texts,
            loc="center",
            ncol=ncol,
            frameon=True,
            fontsize=legend_fontsize,
            handlelength=2.2,
            handleheight=1.4,
            labelspacing=0.6,
            borderpad=0.8,
            columnspacing=1.2,
        )

    for out in out_paths:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=250, bbox_inches="tight")
        print(f"[ok] {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_all() -> None:
    # Row-1 sub-figures (rendered smaller in LaTeX) use a legend-on-top
    # layout; row-2 sub-figures (Age, BMI) use a landscape legend-on-right
    # layout so the larger legend fits beside the two metric panels.
    legend_position = {
        "Marital_status_SPD": "top",
        "Rurality_RCVDTreatment": "top",
        "SEX_SPD": "top",
        "Age_SPD": "right",
        "BMI_SPD": "right",
    }
    for axis_name, axis_group, fair_filename, levels, colors, abbrev in _AXES:
        for suffix, tag in ((":Yes", "yes_only"), (":No", "no_only")):
            out_paths = [
                os.path.join(_MAN_LOW, f"{axis_name}_{tag}.png"),
                os.path.join(_MAN_HIGH, f"{axis_name}_{tag}.png"),
            ]
            _build_figure(
                axis_name,
                axis_group,
                fair_filename,
                levels,
                colors,
                abbrev,
                suffix,
                out_paths,
                legend_pos=legend_position.get(axis_name, "right"),
            )


if __name__ == "__main__":
    render_all()
