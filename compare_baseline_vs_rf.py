from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


KEY_COLS = ["system", "workload"]
METRICS = ["MAPE", "MAE", "RMSE"]


def safe_wilcoxon(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None]:
    """Return Wilcoxon statistic and p-value, or (None, None) if invalid."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()

    n = min(len(x), len(y))
    if n == 0:
        return None, None

    x = x.iloc[:n]
    y = y.iloc[:n]

    if np.allclose(x.values, y.values, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return None, None


def add_improvement_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in METRICS:
        b = f"baseline_{metric}_mean"
        r = f"rf_{metric}_mean"
        imp = f"{metric}_improvement_pct"
        out[imp] = np.where(out[b] != 0, (out[b] - out[r]) / out[b] * 100.0, np.nan)
        out[f"rf_better_{metric}"] = out[r] < out[b]
    return out


def build_comparison_summary(baseline_summary: pd.DataFrame, rf_summary: pd.DataFrame) -> pd.DataFrame:
    base_keep = KEY_COLS + [
        c for c in baseline_summary.columns if c.startswith(("MAPE_", "MAE_", "RMSE_"))
    ]
    rf_keep = KEY_COLS + [
        c for c in rf_summary.columns if c.startswith(("MAPE_", "MAE_", "RMSE_"))
    ]

    base = baseline_summary[base_keep].copy()
    rf = rf_summary[rf_keep].copy()

    base = base.rename(columns={c: f"baseline_{c}" for c in base.columns if c not in KEY_COLS})
    rf = rf.rename(columns={c: f"rf_{c}" for c in rf.columns if c not in KEY_COLS})

    merged = pd.merge(base, rf, on=KEY_COLS, how="inner")
    merged = add_improvement_columns(merged)
    return merged.sort_values(KEY_COLS).reset_index(drop=True)


def build_wilcoxon_results(baseline_runs: pd.DataFrame, rf_runs: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        baseline_runs[KEY_COLS + ["repeat"] + METRICS],
        rf_runs[KEY_COLS + ["repeat"] + METRICS],
        on=KEY_COLS + ["repeat"],
        suffixes=("_baseline", "_rf"),
        how="inner",
    )

    rows: List[Dict] = []
    for (system, workload), group in merged.groupby(KEY_COLS):
        row: Dict[str, object] = {"system": system, "workload": workload, "n_paired_runs": len(group)}
        for metric in METRICS:
            b = group[f"{metric}_baseline"]
            r = group[f"{metric}_rf"]
            stat, p = safe_wilcoxon(b, r)
            row[f"{metric}_wilcoxon_stat"] = stat
            row[f"{metric}_p_value"] = p
            row[f"{metric}_significant_0_05"] = (p is not None and p < 0.05)
            row[f"{metric}_baseline_mean"] = float(b.mean())
            row[f"{metric}_rf_mean"] = float(r.mean())
            row[f"{metric}_improvement_pct"] = float(((b.mean() - r.mean()) / b.mean() * 100.0) if b.mean() != 0 else np.nan)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(KEY_COLS).reset_index(drop=True)


def build_overall_report(comparison_summary: pd.DataFrame, wilcoxon_df: pd.DataFrame) -> pd.DataFrame:
    row: Dict[str, object] = {
        "n_systems": int(comparison_summary["system"].nunique()),
        "n_datasets": int(len(comparison_summary)),
    }

    for metric in METRICS:
        row[f"baseline_{metric}_mean_of_dataset_means"] = float(comparison_summary[f"baseline_{metric}_mean"].mean())
        row[f"rf_{metric}_mean_of_dataset_means"] = float(comparison_summary[f"rf_{metric}_mean"].mean())
        row[f"rf_better_count_{metric}"] = int(comparison_summary[f"rf_better_{metric}"].sum())
        row[f"rf_better_ratio_{metric}"] = float(comparison_summary[f"rf_better_{metric}"].mean())
        row[f"mean_improvement_pct_{metric}"] = float(comparison_summary[f"{metric}_improvement_pct"].mean())
        row[f"median_improvement_pct_{metric}"] = float(comparison_summary[f"{metric}_improvement_pct"].median())
        row[f"significant_count_{metric}"] = int(wilcoxon_df[f"{metric}_significant_0_05"].sum())
        row[f"significant_ratio_{metric}"] = float(wilcoxon_df[f"{metric}_significant_0_05"].mean())

    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Linear Regression baseline vs Random Forest results.")
    parser.add_argument("--baseline-summary", type=str, required=True)
    parser.add_argument("--baseline-runs", type=str, required=True)
    parser.add_argument("--rf-summary", type=str, required=True)
    parser.add_argument("--rf-runs", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results_comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = pd.read_csv(args.baseline_summary)
    baseline_runs = pd.read_csv(args.baseline_runs)
    rf_summary = pd.read_csv(args.rf_summary)
    rf_runs = pd.read_csv(args.rf_runs)

    comparison_summary = build_comparison_summary(baseline_summary, rf_summary)
    wilcoxon_df = build_wilcoxon_results(baseline_runs, rf_runs)
    overall_df = build_overall_report(comparison_summary, wilcoxon_df)

    comparison_summary.to_csv(output_dir / "baseline_vs_rf_summary_comparison.csv", index=False)
    wilcoxon_df.to_csv(output_dir / "baseline_vs_rf_wilcoxon.csv", index=False)
    overall_df.to_csv(output_dir / "baseline_vs_rf_overall_report.csv", index=False)

    print("Done.")
    print(f"Saved: {output_dir / 'baseline_vs_rf_summary_comparison.csv'}")
    print(f"Saved: {output_dir / 'baseline_vs_rf_wilcoxon.csv'}")
    print(f"Saved: {output_dir / 'baseline_vs_rf_overall_report.csv'}")


if __name__ == "__main__":
    main()
