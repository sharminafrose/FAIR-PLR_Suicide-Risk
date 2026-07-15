"""Render single-axis FAIR-PLR coefficient (log-odds) sub-panels used by
the multi-panel Supplementary Figure 19. One PNG per pre-exposure axis;
the supplementary.tex \\begin{figure}[p] block stitches them together.

Each sub-panel is a horizontal grouped-bar chart of the coefficient
values across the subgroup levels of that axis (e.g. for Sex: bars for
Male and Female on each feature row). Colors match the level palette
used by figureS_supp_cross_sectional.py / figureS_supp_pre_exposure.py
so the same color = same level across the whole supplementary.

Output: npj_supplementary_NSDUH/Figures/low_res/{coeff_gender,
coeff_race, coeff_age, coeff_health_insurance, coeff_bmi,
coeff_rurality}.png

Drug Use Disorder is not a pre-exposure subgroup axis, so there is no
coeff_drug.png sub-panel in the multi-panel supplementary layout.
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
_SUPP_FIGS_LOW = os.path.join(_REPO, "npj_supplementary_NSDUH", "Figures", "low_res")


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


# (output filename, FAIR coefficient CSV, levels in order, level colors, abbrev)
_AXES: List[Tuple[str, str, List[str], List[str], Dict[str, str]]] = [
    (
        "coeff_gender.png",
        "coeff_FAIR_Gender_l1_0.5.csv",
        ["Male", "Female"],
        ["#33a02c", "#ff7f00"],
        {},
    ),
    (
        "coeff_race.png",
        "coeff_FAIR_Race_l1_0.5.csv",
        [
            "NonHisp White", "NonHisp Black/Afr Am", "NonHisp Native Am/AK Native",
            "NonHisp Native HI/Other Pac Isl", "NonHisp Asian",
            "NonHisp more than", "Hispanic",
        ],
        ["#1f78b4", "#33a02c", "#df7173", "#ff7f00", "#975fd4", "#9a962e", "#f768a1"],
        {"NonHisp more than": "NonHisp more than one race"},
    ),
    (
        "coeff_age.png",
        "coeff_FAIR_Age_l1_0.5.csv",
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
        "coeff_health_insurance.png",
        "coeff_FAIR_Health_Insurance_l1_0.5.csv",
        ["Medicaid/CHIP", "Medicare", "Other", "Private plan", "Uninsured"],
        ["#33a02c", "#ff7f00", "#1f78b4", "#975fd4", "#df7173"],
        {},
    ),
    (
        "coeff_bmi.png",
        "coeff_FAIR_BMI_l1_0.5.csv",
        ["Underweight", "Healthy", "Overweight", "Obesity",
         "Severe Obesity", "Unknown"],
        ["#33a02c", "#ff7f00", "#1f78b4", "#975fd4", "#df7173", "#9a962e"],
        {},
    ),
    (
        "coeff_rurality.png",
        "coeff_FAIR_Urban_Residence_l1_0.5.csv",
        ["Large Metropolitan", "Nonmetropolitan", "Small Metropolitan"],
        ["#33a02c", "#1f78b4", "#975fd4"],
        {},
    ),
]


def _abbrev(level: str, mp: Dict[str, str]) -> str:
    return mp.get(level, level)


_TOP_K_FEATURES = 7  # show only the most influential predictors per axis


def _build_panel(
    out_filename: str,
    fair_csv: str,
    base_levels: List[str],
    level_colors: List[str],
    abbrev: Dict[str, str],
) -> None:
    df = pd.read_csv(os.path.join(_COEF_DIR, fair_csv), index_col=0)
    df.columns = [c.replace("\n", "") for c in df.columns]

    # Filter feature order to those actually in the CSV; drop intercept
    # (always 0 in these fits) and Year (uninformative for explainability).
    feature_idx_full = [
        f for f in _FEATURE_ORDER
        if f in df.columns and f not in ("intercept", "Year")
    ]

    # Pick the TOP-K features by max absolute coefficient across the
    # relevant subgroup levels (the "most influential predictors").
    sub_df = df.loc[[lvl for lvl in base_levels if lvl in df.index], feature_idx_full]
    influence = sub_df.abs().max(axis=0)
    top_features = influence.sort_values(ascending=False).head(_TOP_K_FEATURES).index.tolist()
    # Reorder the kept features by canonical _FEATURE_ORDER
    feature_idx = [f for f in feature_idx_full if f in top_features]
    label_index = feature_idx[:]  # display labels (no intercept rename needed)

    cols: Dict[str, np.ndarray] = {}
    for lvl in base_levels:
        if lvl in df.index:
            cols[_abbrev(lvl, abbrev)] = df.loc[lvl][feature_idx].values
    mat = pd.DataFrame(cols, index=label_index)
    abbrev_levels = [_abbrev(l, abbrev) for l in base_levels if l in df.index]

    n_features = len(label_index)
    n_levels = len(abbrev_levels)
    bar_h = 0.85 / n_levels
    y_centers = np.arange(n_features)

    # Compact layout: dedicated legend band on top + spacer + plot area,
    # so the legend never bleeds into the bars. Each panel is included
    # at width=0.47\linewidth in the multi-panel Supp Fig 19 stitch.
    legend_rows = -(-n_levels // min(n_levels, 4))  # ceil(n / ncol)
    legend_band = 0.18 * legend_rows + 0.20         # inches
    spacer_band = 0.30                               # inches between legend and plot
    plot_h = max(2.4, 0.45 * n_features)

    fig_h = legend_band + spacer_band + plot_h + 0.7  # 0.7 in for x-label & padding
    fig_w = 7.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    legend_frac = legend_band / fig_h
    spacer_frac = spacer_band / fig_h
    plot_frac = plot_h / fig_h
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[legend_frac, spacer_frac, plot_frac + (1 - legend_frac - spacer_frac - plot_frac)],
        left=0.42, right=0.985,
        top=0.985, bottom=0.16,
        hspace=0.0,
    )
    legend_ax = fig.add_subplot(gs[0, 0])
    spacer_ax = fig.add_subplot(gs[1, 0])
    ax = fig.add_subplot(gs[2, 0])
    legend_ax.axis("off")
    spacer_ax.axis("off")

    for i, lvl in enumerate(abbrev_levels):
        offset = (i - (n_levels - 1) / 2.0) * bar_h
        ax.barh(
            y_centers + offset,
            mat[lvl].values,
            height=bar_h,
            color=level_colors[i],
            edgecolor="#222222", linewidth=0.3,
            label=lvl,
        )
    ax.axvline(0.0, color="#666666", linewidth=0.7)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(label_index, fontsize=7)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("FAIR-PLR coefficient (log-odds)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.6)

    if n_levels:
        ncol = min(n_levels, 4)
        handles, labels = ax.get_legend_handles_labels()
        legend_ax.legend(
            handles, labels,
            loc="center", ncol=ncol, frameon=True, fontsize=9,
            handlelength=1.6, handleheight=1.0,
            labelspacing=0.3, borderpad=0.4, columnspacing=1.0,
        )

    out_path = os.path.join(_SUPP_FIGS_LOW, out_filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[ok] {out_path}")
    plt.close(fig)


def render_all() -> None:
    for out_filename, fair_csv, levels, colors, abbrev in _AXES:
        _build_panel(out_filename, fair_csv, levels, colors, abbrev)


if __name__ == "__main__":
    render_all()
