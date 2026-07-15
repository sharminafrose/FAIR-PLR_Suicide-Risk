"""Render single-axis pre-exposure supplementary figures (Supp Fig 8-13)
using the same legacy ML4H-style conventions as the cross-sectional
sibling script (figureS_supp_cross_sectional.py) and main manuscript
Figures 3 and 4.

Conventions:
  - Three panels per figure, side-by-side: AUC-ROC (Overall),
    TP% (Top 5% Risk), TP% (Top 1% Risk).
  - One color pair per subgroup level. FAIR-PLR bar uses light color +
    '.' hatch; PLR bar uses dark color + '/' hatch.
  - Black star marks the better-performing model within each subgroup.
  - Value labels above each bar, rotated 90 degrees.
  - Legend sits in a band on top spanning the full figure width.

Single-axis means each subgroup level appears once (no :Yes/:No suffix),
so the bar count is just 2 * n_levels (vs. 4 * n_levels in the
cross-sectional combined figures).

Output paths (matching supplementary.tex \\includegraphics calls):
  npj_supplementary_NSDUH/Figures/low_res/gender.png            (Sex)
  npj_supplementary_NSDUH/Figures/low_res/race.png              (Race)
  npj_supplementary_NSDUH/Figures/low_res/age.png               (Age)
  npj_supplementary_NSDUH/Figures/low_res/region.png            (Rurality)
  npj_supplementary_NSDUH/Figures/low_res/health_insurance.png  (Insurance)
  npj_supplementary_NSDUH/Figures/low_res/bmi.png               (BMI)
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
_SUPP_FIGS_LOW = os.path.join(_REPO, "npj_supplementary_NSDUH", "Figures", "low_res")


# (output_filename, axis_group_name, fair_csv, levels, color_pairs, abbrev)
_Axis = Tuple[str, str, str, List[str], List[str], Dict[str, str]]

_AXES: List[_Axis] = [
    (
        "gender.png",
        "Gender",
        "FAIR_Gender_l1_0.5.csv",
        ["Male", "Female"],
        [
            "#b2df8a", "#33a02c",   # green
            "#fdbf6f", "#ff7f00",   # orange
        ],
        {},
    ),
    (
        "race.png",
        "Race",
        "FAIR_Race_l1_0.5.csv",
        [
            "NonHisp White",
            "NonHisp Black/Afr Am",
            "NonHisp Native Am/AK Native",
            "NonHisp Native HI/Other Pac Isl",
            "NonHisp Asian",
            "NonHisp more than",
            "Hispanic",
        ],
        [
            "#a6cee3", "#1f78b4",   # blue
            "#b2df8a", "#33a02c",   # green
            "#fb9a99", "#df7173",   # red
            "#fdbf6f", "#ff7f00",   # orange
            "#cab2d6", "#975fd4",   # purple
            "#ffffb3", "#9a962e",   # yellow / olive
            "#fdcce5", "#f768a1",   # pink
        ],
        {
            "NonHisp White":                  "NonHisp White",
            "NonHisp Black/Afr Am":           "NonHisp Black/Afr Am",
            "NonHisp Native Am/AK Native":    "NonHisp Native Am/AK Native",
            "NonHisp Native HI/Other Pac Isl": "NonHisp Native HI/Other Pac Isl",
            "NonHisp Asian":                  "NonHisp Asian",
            "NonHisp more than":              "NonHisp more than one race",
            "Hispanic":                       "Hispanic",
        },
    ),
    (
        "age.png",
        "Age",
        "FAIR_Age_l1_0.5.csv",
        ["18-25 Years Old", "26-34 Years Old", "35-49 Years Old",
         "50-64 Years Old", "65 or Older"],
        [
            "#b2df8a", "#33a02c",
            "#fdbf6f", "#ff7f00",
            "#a6cee3", "#1f78b4",
            "#cab2d6", "#975fd4",
            "#fb9a99", "#df7173",
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
        "region.png",
        "Urban Residence",
        "FAIR_Urban_Residence_l1_0.5.csv",
        ["Large Metropolitan", "Nonmetropolitan", "Small Metropolitan"],
        [
            "#b2df8a", "#33a02c",
            "#fdbf6f", "#ff7f00",
            "#a6cee3", "#1f78b4",
        ],
        {},
    ),
    (
        "health_insurance.png",
        "Health Insurance",
        "FAIR_Health_Insurance_l1_0.5.csv",
        ["Medicaid/CHIP", "Medicare", "Other", "Private plan", "Uninsured"],
        [
            "#b2df8a", "#33a02c",
            "#fdbf6f", "#ff7f00",
            "#a6cee3", "#1f78b4",
            "#cab2d6", "#975fd4",
            "#fb9a99", "#df7173",
        ],
        {},
    ),
    (
        "bmi.png",
        "BMI",
        "FAIR_BMI_l1_0.5.csv",
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
]


# ---------------------------------------------------------------------------
# Data loading
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
        # Skip cross-sectional files for the corresponding pre-exposure axis
        # (e.g. Separate_Age_SPD_*) by requiring no "_SPD_" / "_Treatment_"
        # in the filename after the axis prefix.
        base = os.path.basename(path)
        tail = base[len(sep_prefix):]
        if "_SPD_" in tail or "_Treatment_" in tail:
            continue
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
# Panel rendering (mirrors cross-sectional script)
# ---------------------------------------------------------------------------


def _abbreviate(level: str, abbrev: Dict[str, str]) -> str:
    return abbrev.get(level, level)


def _draw_panel(
    ax,
    values_fair: List[Optional[float]],
    values_plr: List[Optional[float]],
    full_labels: List[str],
    colors: List[str],
    metric_label: str,
    value_fmt: str,
    ylim: Tuple[float, float],
    x_label: str,
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

        bar_fair = ax.bar(
            centers[i] - bar_w / 2,
            fair_val if fair_val is not None else 0.0,
            bar_w,
            color=light, edgecolor=hatch_color, linewidth=1.2,
            hatch=hatch_fair,
            label=f"FAIR-PLR: Train All, Test {label}",
        )
        bar_plr = ax.bar(
            centers[i] + bar_w / 2,
            plr_val if plr_val is not None else 0.0,
            bar_w,
            color=dark, edgecolor=hatch_color, linewidth=1.2,
            hatch=hatch_plr,
            label=f"PLR: Train {label}, Test {label}",
        )
        legend_handles.extend([bar_fair, bar_plr])
        legend_texts.extend([
            f"FAIR-PLR: Train All, Test {label}",
            f"PLR: Train {label}, Test {label}",
        ])

        label_pad = (ylim[1] - ylim[0]) * 0.02
        if fair_val is not None:
            ax.text(
                centers[i] - bar_w / 2,
                fair_val + label_pad,
                value_fmt.format(fair_val),
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", rotation=90,
            )
        if plr_val is not None:
            ax.text(
                centers[i] + bar_w / 2,
                plr_val + label_pad,
                value_fmt.format(plr_val),
                ha="center", va="bottom",
                fontsize=12, fontweight="bold", rotation=90,
            )

        if fair_val is not None and plr_val is not None:
            star_x = centers[i] - bar_w / 2 if fair_val >= plr_val else centers[i] + bar_w / 2
            ax.text(
                star_x,
                (ylim[1] - ylim[0]) * 0.03,
                "★",
                ha="center", va="bottom",
                fontsize=24, color="#111111", fontweight="bold",
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
# Figure builder — single axis, three side-by-side metric panels
# ---------------------------------------------------------------------------


def _build_axis_figure(
    out_filename: str,
    axis_group: str,
    fair_filename: str,
    levels: List[str],
    colors: List[str],
    abbrev: Dict[str, str],
) -> None:
    fair_df = _load_fair(os.path.join(_METRICS_DIR, fair_filename), axis_group)
    sep_df = _load_separate(_METRICS_DIR, axis_group, _separate_prefix_for(fair_filename))

    full_labels: List[str] = []
    full_colors: List[str] = []
    auc_fair, auc_plr, tpp5_fair, tpp5_plr, tpp1_fair, tpp1_plr = [], [], [], [], [], []
    for i, level in enumerate(levels):
        ab = _abbreviate(level, abbrev)
        full_labels.append(ab)
        full_colors.extend([colors[2 * i], colors[2 * i + 1]])
        auc_fair.append(_value_for(fair_df, "Overall", level, "AUC-ROC"))
        auc_plr.append(_value_for(sep_df, "Overall", level, "AUC-ROC"))
        tpp5_fair.append(_value_for(fair_df, "Top 5% Risk", level, "TP %"))
        tpp5_plr.append(_value_for(sep_df, "Top 5% Risk", level, "TP %"))
        tpp1_fair.append(_value_for(fair_df, "Top 1% Risk", level, "TP %"))
        tpp1_plr.append(_value_for(sep_df, "Top 1% Risk", level, "TP %"))

    keep_idx = [
        i for i in range(len(full_labels))
        if not all(v is None for v in (
            auc_fair[i], auc_plr[i], tpp5_fair[i], tpp5_plr[i], tpp1_fair[i], tpp1_plr[i]))
    ]
    full_labels_k = [full_labels[i] for i in keep_idx]
    full_colors_k: List[str] = []
    for i in keep_idx:
        full_colors_k.extend([full_colors[2 * i], full_colors[2 * i + 1]])
    auc_fair_k = [auc_fair[i] for i in keep_idx]
    auc_plr_k = [auc_plr[i] for i in keep_idx]
    tpp5_fair_k = [tpp5_fair[i] for i in keep_idx]
    tpp5_plr_k = [tpp5_plr[i] for i in keep_idx]
    tpp1_fair_k = [tpp1_fair[i] for i in keep_idx]
    tpp1_plr_k = [tpp1_plr[i] for i in keep_idx]

    n = len(full_labels_k)
    n_entries = 2 * n
    if n_entries <= 8:
        legend_ncol = 2
    elif n_entries <= 14:
        legend_ncol = 3
    else:
        legend_ncol = 4
    legend_rows = -(-n_entries // legend_ncol)
    legend_band = 0.06 * legend_rows + 0.4

    spacer_band = 0.6  # inches of breathing room between legend and panels

    fig_w = max(15.0, 4.0 + 1.0 * n)
    fig_h = 7.5 + legend_band + spacer_band
    fig = plt.figure(figsize=(fig_w, fig_h))
    legend_frac = legend_band / fig_h
    spacer_frac = spacer_band / fig_h
    panel_frac = 1.0 - legend_frac - spacer_frac
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[legend_frac, spacer_frac, panel_frac],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.06, right=0.985,
        top=0.985, bottom=0.10,
        hspace=0.05, wspace=0.18,
    )
    legend_ax = fig.add_subplot(gs[0, :])
    spacer_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    spacer_ax.axis("off")
    ax_auc = fig.add_subplot(gs[2, 0])
    ax_tp5 = fig.add_subplot(gs[2, 1])
    ax_tp1 = fig.add_subplot(gs[2, 2])

    _draw_panel(
        ax_auc, auc_fair_k, auc_plr_k, full_labels_k, full_colors_k,
        metric_label="AUC-ROC", value_fmt="{:.3f}",
        ylim=(0.0, 1.25), x_label="Overall",
    )

    finite5 = [v for v in tpp5_fair_k + tpp5_plr_k if v is not None]
    tp5_upper = max(50.0, max(finite5) * 1.55) if finite5 else 50.0
    _draw_panel(
        ax_tp5, tpp5_fair_k, tpp5_plr_k, full_labels_k, full_colors_k,
        metric_label="TP %", value_fmt="{:.2f}",
        ylim=(0.0, tp5_upper), x_label="Top 5% Risk",
    )

    finite1 = [v for v in tpp1_fair_k + tpp1_plr_k if v is not None]
    tp1_upper = max(30.0, max(finite1) * 1.55) if finite1 else 30.0
    _draw_panel(
        ax_tp1, tpp1_fair_k, tpp1_plr_k, full_labels_k, full_colors_k,
        metric_label="TP %", value_fmt="{:.2f}",
        ylim=(0.0, tp1_upper), x_label="Top 1% Risk",
    )

    handles = getattr(ax_auc, "_legend_handles", [])
    texts = getattr(ax_auc, "_legend_texts", [])
    if handles:
        fontsize = (
            16 if n_entries <= 8
            else 13 if n_entries <= 14
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

    out_path = os.path.join(_SUPP_FIGS_LOW, out_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Lower DPI keeps the supplementary PDF small (250 -> 120 cuts file
    # size by ~3x while remaining sharp at the embedded figure width).
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[ok] {out_path}")
    plt.close(fig)


def render_all() -> None:
    for out_filename, axis_group, fair_filename, levels, colors, abbrev in _AXES:
        _build_axis_figure(out_filename, axis_group, fair_filename, levels, colors, abbrev)


if __name__ == "__main__":
    render_all()
