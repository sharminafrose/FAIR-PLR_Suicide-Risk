"""Convert wide-format FAIR / Separate metric CSVs into the long-format
`subgroup_results_<stem>.csv` layout that `src.plotting` consumes.

Long-format schema (one row per subgroup x model x metric):

    subgroup, model, metric, point, ci_low, ci_high

For each axis we emit one long-format CSV per figure spec (see _SPECS below).
Rows for FAIR-PLR come from `FAIR_<axis>_l1_0.5.csv`; rows for Separate-PLR
come from the per-level `Separate_<axis>_<level>_l1_0.5.csv` files and are
only emitted for axes where every subgroup level has a Separate CSV
(otherwise the comparison panel would be incomplete and misleading).

CIs are filled in from `bootstrap_delong_<axis>.csv` where available
(pre-exposure axes only). For cross-sectional axes CIs are left blank so
the plot skips the error-bar caps.
"""

from __future__ import annotations

import os
import glob
import re
import pandas as pd


RESULTS_METRICS = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "metrics")
)

# (csv_stem, fair_file_prefix, separate_file_prefix, axis_group_value,
#  bootstrap_axis_name_or_None)
_SPECS = [
    ("age",               "FAIR_Age",                    "Separate_Age_",                    "Age",         "Age"),
    ("sex",               "FAIR_Gender",                 "Separate_Gender_",                 "Gender",      "Gender"),
    ("race",              "FAIR_Race",                   "Separate_Race_",                   "Race",        "Race"),
    ("bmi",               "FAIR_BMI",                    "Separate_BMI_",                    "BMI",         "BMI"),
    ("insurance",         "FAIR_Health_Insurance",       "Separate_Health_Insurance_",       "Health Insurance", "Health Insurance"),
    ("rurality",          "FAIR_Urban_Residence",        "Separate_Urban_Residence_",        "Urban Residence", "Urban Residence"),
    ("sex_spd",           "FAIR_Sex_SPD",                "Separate_Sex_SPD_",                "Sex:SPD",     None),
    ("age_spd",           "FAIR_Age_SPD",                "Separate_Age_SPD_",                "Age:SPD",     None),
    ("marital_spd",       "FAIR_Marital_Status_SPD",     "Separate_Marital_Status_SPD_",     "Marital Status:SPD", None),
    ("bmi_spd",           "FAIR_BMI_SPD",                "Separate_BMI_SPD_",                "BMI:SPD",     None),
    ("rurality_treatment", "FAIR_Urban_Residence_Treatment", "Separate_Urban_Residence_Treatment_", "Urban Residence:Treatment", None),
]


def _fair_row_to_long(df: pd.DataFrame, model: str, axis_group_value: str) -> pd.DataFrame:
    """Pull per-subgroup rows from the wide FAIR/Agnostic CSV into long format.

    The wide CSV has three rows per subgroup level: Overall (AUC), Top 1% (TPR/Precision top-1),
    Top 5% (TPR/Precision top-5). We pull AUC from the Overall row and
    TPR/Precision from the Top-k rows.
    """
    sub = df[df["Group"] == axis_group_value].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for level, grp in sub.groupby("Group Value"):
        overall = grp[grp["Subset"] == "Overall"]
        top1 = grp[grp["Subset"] == "Top 1% Risk"]
        top5 = grp[grp["Subset"] == "Top 5% Risk"]
        if not overall.empty:
            rows.append((level, model, "auc", float(overall["AUC-ROC"].iloc[0])))
        if not top1.empty:
            rows.append((level, model, "tpr_top1", float(top1["TPR (%)"].iloc[0]) / 100.0))
            rows.append((level, model, "precision_top1", float(top1["Precision (%)"].iloc[0]) / 100.0))
        if not top5.empty:
            rows.append((level, model, "tpr_top5", float(top5["TPR (%)"].iloc[0]) / 100.0))
            rows.append((level, model, "precision_top5", float(top5["Precision (%)"].iloc[0]) / 100.0))
    return pd.DataFrame(rows, columns=["subgroup", "model", "metric", "point"])


def _separate_files_to_long(separate_files: list[str], axis_group_value: str) -> pd.DataFrame:
    """One row per (level, metric) from the stack of Separate-PLR per-level CSVs."""
    frames = []
    for f in separate_files:
        df = pd.read_csv(f)
        frames.append(_fair_row_to_long(df, "Separate-PLR", axis_group_value))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _attach_ci(long_df: pd.DataFrame, bootstrap_axis: str | None) -> pd.DataFrame:
    """Join CI bounds from `bootstrap_delong_<axis>.csv` onto the long frame.

    Only FAIR-PLR rows will match; Separate-PLR rows are left blank for CIs.
    Non-matching rows keep blank ci_low/ci_high, which plotting.py treats as
    no-error-bar.
    """
    long_df = long_df.copy()
    long_df["ci_low"] = ""
    long_df["ci_high"] = ""
    if bootstrap_axis is None:
        return long_df
    boot_path = os.path.join(RESULTS_METRICS, f"bootstrap_delong_{bootstrap_axis}.csv")
    if not os.path.exists(boot_path):
        return long_df
    boot = pd.read_csv(boot_path)
    boot = boot[boot["model"] == "FAIR-PLR"][["subgroup", "metric", "ci_low", "ci_high"]]
    boot = boot.rename(columns={"ci_low": "ci_low_b", "ci_high": "ci_high_b"})
    merged = long_df.merge(boot, on=["subgroup", "metric"], how="left")
    merged["ci_low"] = merged["ci_low_b"].where(merged["model"] == "FAIR-PLR", "")
    merged["ci_high"] = merged["ci_high_b"].where(merged["model"] == "FAIR-PLR", "")
    return merged[["subgroup", "model", "metric", "point", "ci_low", "ci_high"]]


def build_all(metrics_dir: str = RESULTS_METRICS) -> None:
    for csv_stem, fair_prefix, separate_prefix, axis_group_value, bootstrap_axis in _SPECS:
        fair_path = os.path.join(metrics_dir, f"{fair_prefix}_l1_0.5.csv")
        if not os.path.exists(fair_path):
            print(f"[skip] {csv_stem}: no FAIR file at {fair_path}")
            continue
        fair = pd.read_csv(fair_path)
        fair_long = _fair_row_to_long(fair, "FAIR-PLR", axis_group_value)
        if fair_long.empty:
            print(f"[skip] {csv_stem}: FAIR file has no '{axis_group_value}' rows")
            continue

        separate_files = sorted(glob.glob(os.path.join(metrics_dir, f"{separate_prefix}*_l1_0.5.csv")))
        separate_long = _separate_files_to_long(separate_files, axis_group_value)

        if separate_long.empty:
            long_df = fair_long.copy()
            long_df["ci_low"] = ""
            long_df["ci_high"] = ""
            out = os.path.join(metrics_dir, f"subgroup_results_{csv_stem}.csv")
            long_df.to_csv(out, index=False)
            print(f"[ok  ] {out}  (FAIR-PLR only — Separate-PLR files missing)")
        else:
            combined = pd.concat([fair_long, separate_long], ignore_index=True)
            combined = _attach_ci(combined, bootstrap_axis)
            out = os.path.join(metrics_dir, f"subgroup_results_{csv_stem}.csv")
            combined.to_csv(out, index=False)
            print(f"[ok  ] {out}  (FAIR-PLR + Separate-PLR, {len(separate_files)} levels)")


if __name__ == "__main__":
    build_all()
