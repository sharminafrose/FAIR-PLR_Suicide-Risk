"""Render cross-sectional FAIR-PLR coefficient (log-odds) figures
(Supplementary Figures 14-18) using the FAIR fits and the same color
palette as the cross-sectional performance figures.

Each figure has two side-by-side panels (left = SPD/Treatment Yes,
right = No). Each panel is a horizontal grouped-bar chart:
  - Y-axis: features in the canonical order from
    9_heatmap_crosssectional.py (intercept and Year at the bottom).
  - X-axis: FAIR-PLR coefficient (log-odds).
  - Within each feature row, one colored bar per subgroup level. Colors
    match the level palette used in figureS_supp_cross_sectional.py
    (light shade per level), so the same color = same level across all
    supplementary cross-sectional figures.

Output: npj_supplementary_NSDUH/low_res/{coeff_age_spd,coeff_sex_spd,
coeff_marital_spd,coeff_rurality_received_treatment,coeff_bmi_spd}.png
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_HERE = os.path.dirname(os.path.abspath(__file__))
_NSDUH_CODE = os.path.dirname(_HERE)
_REPO = os.path.dirname(_NSDUH_CODE)
_COEF_DIR = os.path.join(_NSDUH_CODE, "results", "coefficients")
_SUPP_LOW = os.path.join(_REPO, "npj_supplementary_NSDUH", "low_res")


# (output_filename, FAIR coefficient CSV stem, base levels in order, level colors, abbrev)
_AXES: List[Tuple[str, str, List[str], List[str], Dict[str, str]]] = [
    (
        "coeff_age_spd.png",
        "coeff_FAIR_Age_SPD_l1_0.5.csv",
        ["18-25 Years Old", "26-34 Years Old", "35-49 Years Old",
         "50-64 Years Old", "65 or Older"],
        ["#33a02c", "#ff7f00", "#1f78b4", "#975fd4", "#df7173"],
        {
            "18-25 Years Old": "18-25Y",
            "26-34 Years Old": "26-34Y",
            "35-49 Years Old": "35-49Y",
            "50-64 Years Old": "50-64Y",
            "65 or Older":     "65Y or Older",
        },
    ),
    (
        "coeff_sex_spd.png",
        "coeff_FAIR_Sex_SPD_l1_0.5.csv",
        ["Male", "Female"],
        ["#33a02c", "#ff7f00"],
        {},
    ),
    (
        "coeff_marital_spd.png",
        "coeff_FAIR_Marital_Status_SPD_l1_0.5.csv",
        ["Married", "Widowed", "Divorced or Separated", "Never Been Married"],
        ["#33a02c", "#1f78b4", "#975fd4", "#9a962e"],
        {},
    ),
    (
        "coeff_rurality_received_treatment.png",
        "coeff_FAIR_Urban_Residence_Treatment_l1_0.5.csv",
        ["Large Metropolitan", "Nonmetropolitan", "Small Metropolitan"],
        ["#33a02c", "#1f78b4", "#975fd4"],
        {},
    ),
    (
        "coeff_bmi_spd.png",
        "coeff_FAIR_BMI_SPD_l1_0.5.csv",
        ["Underweight", "Healthy", "Overweight", "Obesity",
         "Severe Obesity", "Unknown"],
        ["#33a02c", "#ff7f00", "#1f78b4", "#975fd4", "#df7173", "#9a962e"],
        {},
    ),
]


# Feature display order — top of the chart at index 0 (last on the y-axis
# after .invert_yaxis()), bottom is the intercept. Mirrors the canonical
# ordering used in 10_heatmap_pre_exposure.py / 9_heatmap_crosssectional.py.
_FEATURE_ORDER = [
    "intercept",
    "Year",
    "Drug or Alcohol Use Disorder (any past year)",
    "Serious Psychological Distress (any past month)",
    "Serious Psychological Distress (any past year)",
    "Received Substance Use or Mental Health Treatment (any past year)",
    "Major Depressive Episode (any past year)",
    "Alcohol Use Disorder (any past year)",
    "Marijuana Use Disorder (any past year)",
    "Stimulant Use w/o RX (any past year)",
    "Sedative Use w/o RX (any past year)",
    "Cocaine Use Disorder (any past year)",
    "Heroin Use Disorder (any past year)",
    "Hallucinogen Use Disorder (any past year)",
    "Inhalant Use Disorder (any past year)",
    "Oxycontin Use (any past year)",
    "Drug Use Disorder (any past year)",
    "Binge Alcohol Use (any past month)",
    "Heavy Alcohol Use (any past month)",
    "Difficulty in Work Response (any past year)",
    "Felt Tired/Low Energy (nearly every day)",
    "Felt Worthless (nearly every day)",
    "Age_18-25 Years Old",
    "Age_26-34 Years Old",
    "Age_35-49 Years Old",
    "Age_50-64 Years Old",
    "Age_65 or Older",
    "Gender_Female",
    "Gender_Male",
    "Race_Hispanic",
    "Race_NonHisp Asian",
    "Race_NonHisp Black/Afr Am",
    "Race_NonHisp Native Am/AK Native",
    "Race_NonHisp Native HI/Other Pac Isl",
    "Race_NonHisp White",
    "Race_NonHisp more than",
    "Marital Status_Divorced or Separated",
    "Marital Status_Married",
    "Marital Status_Never Been Married",
    "Marital Status_Widowed",
    "Education_College graduate",
    "Education_High school grad",
    "Education_Less high school",
    "Education_Some coll/Assoc Dg",
    "Poverty_Living in Poverty",
    "Poverty_Income Up to 2X Fed Pov Thresh",
    "Poverty_Income More Than 2X Fed Pov Thresh",
    "Employment Status_Employed full time",
    "Employment Status_Employed part time",
    "Employment Status_Other",
    "Employment Status_Unemployed",
    "Family Income_Less than $20,000",
    "Family Income_$20,000 - $49,999",
    "Family Income_$50,000 - $74,999",
    "Family Income_$75,000 or More",
    "Health Insurance_Medicaid/CHIP",
    "Health Insurance_Medicare",
    "Health Insurance_Other",
    "Health Insurance_Private plan",
    "Health Insurance_Uninsured",
    "Urban Residence_Large Metropolitan",
    "Urban Residence_Nonmetropolitan",
    "Urban Residence_Small Metropolitan",
    "BMI_Healthy",
    "BMI_Obesity",
    "BMI_Overweight",
    "BMI_Severe Obesity",
    "BMI_Underweight",
    "BMI_Unknown",
]


def _abbrev(level: str, mp: Dict[str, str]) -> str:
    return mp.get(level, level)


def _draw_panel(
    ax,
    df_panel: pd.DataFrame,         # rows = features (in _FEATURE_ORDER), cols = levels
    levels: List[str],
    colors: List[str],
    title: str,
    show_yticklabels: bool = True,
):
    n_features = len(df_panel.index)
    n_levels = len(levels)
    bar_h = 0.85 / n_levels  # leave 0.15 of vertical space between feature groups
    y_centers = np.arange(n_features)

    for i, level in enumerate(levels):
        offset = (i - (n_levels - 1) / 2.0) * bar_h
        if level not in df_panel.columns:
            continue
        vals = df_panel[level].values
        ax.barh(
            y_centers + offset,
            vals,
            height=bar_h,
            color=colors[i],
            edgecolor="#222222",
            linewidth=0.3,
            label=level,
        )
    ax.axvline(0.0, color="#666666", linewidth=0.7)
    ax.set_yticks(y_centers)
    if show_yticklabels:
        ax.set_yticklabels(df_panel.index, fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=11)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.6)


def _build_axis_figure(
    out_filename: str,
    fair_csv: str,
    base_levels: List[str],
    level_colors: List[str],
    abbrev: Dict[str, str],
    yes_label: str = "Yes",
    no_label: str = "No",
) -> None:
    df = pd.read_csv(os.path.join(_COEF_DIR, fair_csv), index_col=0)
    # Normalize column names by stripping accidental newlines
    df.columns = [c.replace("\n", "") for c in df.columns]

    yes_rows: Dict[str, pd.Series] = {}
    no_rows: Dict[str, pd.Series] = {}
    for level in base_levels:
        yes_key = f"{level}:Yes"
        no_key = f"{level}:No"
        if yes_key in df.index:
            yes_rows[level] = df.loc[yes_key]
        if no_key in df.index:
            no_rows[level] = df.loc[no_key]

    # Build per-panel feature-level matrices in canonical row order.
    feature_idx = [f for f in _FEATURE_ORDER if f in df.columns]
    label_index = ["Intercept" if f == "intercept" else f for f in feature_idx]

    yes_mat = pd.DataFrame(
        {_abbrev(lvl, abbrev): yes_rows[lvl][feature_idx].values for lvl in base_levels if lvl in yes_rows},
        index=label_index,
    )
    no_mat = pd.DataFrame(
        {_abbrev(lvl, abbrev): no_rows[lvl][feature_idx].values for lvl in base_levels if lvl in no_rows},
        index=label_index,
    )

    abbrev_levels = [_abbrev(l, abbrev) for l in base_levels]

    n_features = len(label_index)
    n_levels = len(base_levels)
    # Figure is portrait double-panel; aspect chosen so when scaled to
    # 0.7*linewidth in LaTeX the height saturates at 0.85*textheight,
    # which gives y-axis labels enough vertical room to render at ~7pt.
    fig_h = max(14.0, 0.27 * n_features)
    fig_w = 9.0  # narrow double-panel keeps bars proportional
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.04, 0.96],
        width_ratios=[1.0, 1.0],
        left=0.27, right=0.985,
        top=0.985, bottom=0.035,
        hspace=0.03, wspace=0.06,
    )
    legend_ax = fig.add_subplot(gs[0, :])
    legend_ax.axis("off")
    ax_yes = fig.add_subplot(gs[1, 0])
    ax_no = fig.add_subplot(gs[1, 1])

    _draw_panel(ax_yes, yes_mat, abbrev_levels, level_colors, title=yes_label,
                show_yticklabels=True)
    _draw_panel(ax_no, no_mat, abbrev_levels, level_colors, title=no_label,
                show_yticklabels=False)
    # Force matching x-axis range so left/right are visually comparable.
    finite_vals = pd.concat([yes_mat.stack(), no_mat.stack()]).abs()
    if not finite_vals.empty:
        x_lim = max(0.5, float(finite_vals.max()) * 1.10)
        ax_yes.set_xlim(-x_lim, x_lim)
        ax_no.set_xlim(-x_lim, x_lim)
    ax_yes.set_xlabel("FAIR-PLR coefficient (log-odds)", fontsize=12, fontweight="bold")
    ax_no.set_xlabel("FAIR-PLR coefficient (log-odds)", fontsize=12, fontweight="bold")

    # Legend at the top
    handles, labels = ax_yes.get_legend_handles_labels()
    if handles:
        legend_ax.legend(
            handles, labels,
            loc="center", ncol=min(len(labels), 6), frameon=True,
            fontsize=14, handlelength=1.8, handleheight=1.2,
            labelspacing=0.4, borderpad=0.6, columnspacing=2.0,
        )

    out_path = os.path.join(_SUPP_LOW, out_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[ok] {out_path}")
    plt.close(fig)


def render_all() -> None:
    yes_no_label = ("With SPD", "Without SPD")
    treatment_label = ("Received Treatment", "No Treatment")
    for out_filename, fair_csv, levels, colors, abbrev in _AXES:
        if "rurality_received_treatment" in out_filename:
            yes_lbl, no_lbl = treatment_label
        else:
            yes_lbl, no_lbl = yes_no_label
        _build_axis_figure(out_filename, fair_csv, levels, colors, abbrev, yes_lbl, no_lbl)


if __name__ == "__main__":
    render_all()
