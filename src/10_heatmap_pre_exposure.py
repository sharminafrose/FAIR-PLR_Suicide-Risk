"""Regenerate the pre-exposure subgroup coefficient-variability heatmap
(Supplementary Figure 2).

For each of the six pre-exposure FAIR-PLR axes (Sex, Race, Age, BMI,
Health Insurance, Rurality), compute the per-feature standard deviation
of the FAIR-PLR coefficient across subgroup levels of that axis, and
render a single heatmap with features on the y-axis and axes on the
x-axis. Uses the exact feature ordering, display labels, and plot
conventions as the main-manuscript heatmap (9_heatmap_crosssectional.py).

Output: npj_supplementary_NSDUH/Figures/low_res/heatmap_coeff.png
"""
from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_RESULTS_COEF = os.path.join(_REPO_ROOT, "results", "coefficients")
_OUT_PATH = os.path.abspath(os.path.join(
    _REPO_ROOT, "..", "npj_supplementary_NSDUH", "Figures", "low_res",
    "heatmap_coeff.png"))


# Axis spec: (column header in the heatmap, FAIR coefficient CSV stem).
# Order mirrors the original Supplementary Figure 2 column order.
_AXES = [
    ("Sex",                      "coeff_FAIR_Gender_l1_0.5.csv"),
    ("Race",                     "coeff_FAIR_Race_l1_0.5.csv"),
    ("Age",                      "coeff_FAIR_Age_l1_0.5.csv"),
    ("BMI",                      "coeff_FAIR_BMI_l1_0.5.csv"),
    ("Health\nInsurance",        "coeff_FAIR_Health_Insurance_l1_0.5.csv"),
    ("Rurality",                 "coeff_FAIR_Urban_Residence_l1_0.5.csv"),
]


# Row ordering — matches the original Supplementary Figure 2 sequence
# from the previous submission (Sex first, Race grouped by NHPI ordering,
# Age/BMI/Insurance/Rurality next, Drug Use Disorder as its own row right
# after the Urban Residence block, Marital Status / Education / Poverty /
# Employment / Family Income, then the alphabetical behavioral/clinical
# block, with Year and Intercept last).
_FEATURE_ORDER = [
    # Sex block (top)
    "Sex: Male",
    "Sex: Female",
    # Race block
    "Race: NonHisp White",
    "Race: NonHisp Asian",
    "Race: NonHisp Black/Afr Am",
    "Race: NonHisp Native Am/AK Native",
    "Race: NonHisp Native HI/Other Pac Isl",
    "Race: NonHisp more than one race",
    "Race: Hispanic",
    # Age block
    "Age: 18-25 Years Old",
    "Age: 26-34 Years Old",
    "Age: 35-49 Years Old",
    "Age: 50-64 Years Old",
    "Age: 65 or Older",
    # BMI block
    "BMI: Healthy",
    "BMI: Obesity",
    "BMI: Overweight",
    "BMI: Severe Obesity",
    "BMI: Underweight",
    "BMI: Unknown",
    # Health Insurance block
    "Health Insurance: Medicaid/CHIP",
    "Health Insurance: Medicare",
    "Health Insurance: Other",
    "Health Insurance: Private plan",
    "Health Insurance: Uninsured",
    # Urban Residence block
    "Urban Residence: Large Metropolitan",
    "Urban Residence: Nonmetropolitan",
    "Urban Residence: Small Metropolitan",
    # Drug Use Disorder (single-row block, kept here per original layout)
    "Drug Use Disorder (any past year)",
    # Marital Status block
    "Marital Status: Married",
    "Marital Status: Widowed",
    "Marital Status: Divorced or Separated",
    "Marital Status: Never Been Married",
    # Education block (alphabetical)
    "Education: College graduate",
    "Education: High school grad",
    "Education: Less high school",
    "Education: Some coll/Assoc Dg",
    # Poverty block
    "Poverty: Living in Poverty",
    "Poverty: Income Up to 2X Fed Pov Thresh",
    "Poverty: Income More Than 2X Fed Pov Thresh",
    # Employment Status block (alphabetical)
    "Employment Status: Employed full time",
    "Employment Status: Employed part time",
    "Employment Status: Other",
    "Employment Status: Unemployed",
    # Family Income block
    "Family Income: Less than $20,000",
    "Family Income: $20,000 - $49,999",
    "Family Income: $50,000 - $74,999",
    "Family Income: $75,000 or More",
    # Behavioral / clinical block (alphabetical, except Year and Intercept last)
    "Alcohol Use Disorder (any past year)",
    "Binge Alcohol Use (any past month)",
    "Cocaine Use Disorder (any past year)",
    "Difficulty in Work Response (any past year)",
    "Drug or Alcohol Use Disorder (any past year)",
    "Felt Tired/Low Energy (nearly every day)",
    "Felt Worthless (nearly every day)",
    "Hallucinogen Use Disorder (any past year)",
    "Heavy Alcohol Use (any past month)",
    "Heroin Use Disorder (any past year)",
    "Inhalant Use Disorder (any past year)",
    "Major Depressive Episode (any past year)",
    "Marijuana Use Disorder (any past year)",
    "Oxycontin Use (any past year)",
    "Received Substance Use or Mental Health Treatment (any past year)",
    "Sedative Use w/o RX (any past year)",
    "Serious Psychological Distress (any past month)",
    "Serious Psychological Distress (any past year)",
    "Stimulant Use w/o RX (any past year)",
    "Year",
    "Intercept",
]


_CSV_TO_DISPLAY = {
    "intercept": "Intercept",
    # Age
    "Age_18-25 Years Old": "Age: 18-25 Years Old",
    "Age_26-34 Years Old": "Age: 26-34 Years Old",
    "Age_35-49 Years Old": "Age: 35-49 Years Old",
    "Age_50-64 Years Old": "Age: 50-64 Years Old",
    "Age_65 or Older":     "Age: 65 or Older",
    # Sex
    "Gender_Male":         "Sex: Male",
    "Gender_Female":       "Sex: Female",
    # Race
    "Race_NonHisp White":              "Race: NonHisp White",
    "Race_NonHisp Black/Afr Am":       "Race: NonHisp Black/Afr Am",
    "Race_NonHisp Asian":              "Race: NonHisp Asian",
    "Race_NonHisp Native Am/AK Native": "Race: NonHisp Native Am/AK Native",
    "Race_NonHisp Native HI/Other Pac Isl": "Race: NonHisp Native HI/Other Pac Isl",
    "Race_NonHisp more than":          "Race: NonHisp more than one race",
    "Race_Hispanic":                   "Race: Hispanic",
    # Marital Status
    "Marital Status_Married":              "Marital Status: Married",
    "Marital Status_Widowed":              "Marital Status: Widowed",
    "Marital Status_Divorced or Separated": "Marital Status: Divorced or Separated",
    "Marital Status_Never Been Married":   "Marital Status: Never Been Married",
    # Education
    "Education_Less high school":    "Education: Less high school",
    "Education_High school grad":    "Education: High school grad",
    "Education_Some coll/Assoc Dg":  "Education: Some coll/Assoc Dg",
    "Education_College graduate":    "Education: College graduate",
    # Poverty
    "Poverty_Living in Poverty":                  "Poverty: Living in Poverty",
    "Poverty_Income Up to 2X Fed Pov Thresh":     "Poverty: Income Up to 2X Fed Pov Thresh",
    "Poverty_Income More Than 2X Fed Pov Thresh": "Poverty: Income More Than 2X Fed Pov Thresh",
    # Employment
    "Employment Status_Unemployed":          "Employment Status: Unemployed",
    "Employment Status_Employed part time":  "Employment Status: Employed part time",
    "Employment Status_Employed full time":  "Employment Status: Employed full time",
    "Employment Status_Other":               "Employment Status: Other",
    # Family Income
    "Family Income_Less than $20,000":         "Family Income: Less than $20,000",
    "Family Income_$20,000 - $49,999":         "Family Income: $20,000 - $49,999",
    "Family Income_$50,000 - $74,999":         "Family Income: $50,000 - $74,999",
    "Family Income_$75,000 or More":           "Family Income: $75,000 or More",
    # Health Insurance
    "Health Insurance_Medicaid/CHIP":  "Health Insurance: Medicaid/CHIP",
    "Health Insurance_Medicare":       "Health Insurance: Medicare",
    "Health Insurance_Other":          "Health Insurance: Other",
    "Health Insurance_Private plan":   "Health Insurance: Private plan",
    "Health Insurance_Uninsured":      "Health Insurance: Uninsured",
    # Urban Residence
    "Urban Residence_Large Metropolitan": "Urban Residence: Large Metropolitan",
    "Urban Residence_Nonmetropolitan":    "Urban Residence: Nonmetropolitan",
    "Urban Residence_Small Metropolitan": "Urban Residence: Small Metropolitan",
    # BMI
    "BMI_Underweight":    "BMI: Underweight",
    "BMI_Healthy":        "BMI: Healthy",
    "BMI_Overweight":     "BMI: Overweight",
    "BMI_Obesity":        "BMI: Obesity",
    "BMI_Severe Obesity": "BMI: Severe Obesity",
    "BMI_Unknown":        "BMI: Unknown",
}


def _load_std(fair_csv_name: str) -> pd.Series:
    path = os.path.join(_RESULTS_COEF, fair_csv_name)
    df = pd.read_csv(path, index_col=0)
    renamed = {c: c.replace("\n", "") for c in df.columns}
    df = df.rename(columns=renamed)
    stds = df.std(axis=0, ddof=0)
    remapped = {}
    for col, val in stds.items():
        key = _CSV_TO_DISPLAY.get(col, col)
        remapped[key] = float(val)
    return pd.Series(remapped)


def build_heatmap_df() -> pd.DataFrame:
    frames = {header: _load_std(csv_stem) for header, csv_stem in _AXES}
    df = pd.DataFrame(frames)
    df = df.reindex(_FEATURE_ORDER)
    return df


def main() -> None:
    df = build_heatmap_df()
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)

    fig_h = max(26.0, 0.38 * len(df))
    fig, ax = plt.subplots(figsize=(22, fig_h))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "St. Dev", "shrink": 1.0, "aspect": 40, "pad": 0.02},
        annot_kws={"size": 22},
        ax=ax,
    )
    ax.set_title("Standard Deviation of Coefficients", fontsize=30, fontweight="bold", pad=24)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=18)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label("St. Dev", fontsize=24)

    plt.tight_layout()
    plt.savefig(_OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"[ok] wrote {_OUT_PATH}  shape={df.shape}  finite cells={int(df.notna().sum().sum())}")


if __name__ == "__main__":
    main()
