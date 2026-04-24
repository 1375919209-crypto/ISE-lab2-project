from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_SYSTEMS = ["batlik", "dconvert", "h2", "jump3r", "kanzi", "lrzip", "x264", "xz", "z3"]


def build_model(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing + linear regression pipeline."""
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    return model


def safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Compute MAPE (%) while avoiding divide-by-zero explosions when true values are 0.
    sklearn's MAPE can become extremely large for near-zero targets, so this helper
    skips exact-zero targets. If all y_true values are 0, returns NaN.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = y_true_arr != 0
    if not np.any(mask):
        return np.nan
    return mean_absolute_percentage_error(y_true_arr[mask], y_pred_arr[mask]) * 100.0


def evaluate_dataset(csv_path: Path, repeats: int, test_size: float, random_seed: int) -> pd.DataFrame:
    """Run repeated train/test splits on one CSV dataset and return per-repeat metrics."""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Dataset {csv_path} must have at least 2 columns.")

    # First n-1 columns are features, last column is target performance.
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    results: List[Dict] = []
    for repeat in range(repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed + repeat,
        )

        model = build_model(X_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mape = safe_mape(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        results.append(
            {
                "system": csv_path.parent.name,
                "workload": csv_path.stem,
                "file_name": csv_path.name,
                "file_path": str(csv_path),
                "repeat": repeat + 1,
                "n_samples": len(df),
                "n_features": X.shape[1],
                "target_name": df.columns[-1],
                "MAPE": mape,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    return pd.DataFrame(results)


def collect_csv_files(data_root: Path, systems: List[str] | None = None) -> List[Path]:
    systems = systems or DEFAULT_SYSTEMS
    files: List[Path] = []
    for system in systems:
        system_dir = data_root / system
        if not system_dir.exists():
            print(f"[WARN] System folder not found, skipped: {system_dir}")
            continue
        files.extend(sorted(system_dir.glob("*.csv")))
    return files


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        results_df.groupby(
            ["system", "workload", "file_name", "file_path", "n_samples", "n_features", "target_name"],
            as_index=False
        )
        .agg(
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
            MAPE_min=("MAPE", "min"),
            MAPE_max=("MAPE", "max"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MAE_min=("MAE", "min"),
            MAE_max=("MAE", "max"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            RMSE_min=("RMSE", "min"),
            RMSE_max=("RMSE", "max"),
        )
    )
    summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary_df.columns
    ]
    return summary_df.sort_values(["system", "workload"]).reset_index(drop=True)


def build_overall(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_systems": [results_df["system"].nunique()],
            "n_datasets": [summary_df.shape[0]],
            "n_total_runs": [results_df.shape[0]],
            "MAPE_mean_across_all_runs": [results_df["MAPE"].mean()],
            "MAPE_std_across_all_runs": [results_df["MAPE"].std()],
            "MAE_mean_across_all_runs": [results_df["MAE"].mean()],
            "MAE_std_across_all_runs": [results_df["MAE"].std()],
            "RMSE_mean_across_all_runs": [results_df["RMSE"].mean()],
            "RMSE_std_across_all_runs": [results_df["RMSE"].std()],
            "MAPE_mean_of_dataset_means": [summary_df["MAPE_mean"].mean()],
            "MAE_mean_of_dataset_means": [summary_df["MAE_mean"].mean()],
            "RMSE_mean_of_dataset_means": [summary_df["RMSE_mean"].mean()],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab2 baseline: Linear Regression for configuration performance learning")
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets",
        help="Path to the lab2 datasets root folder",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Optional list of systems to run, e.g. --systems h2 x264 z3",
    )
    parser.add_argument("--repeats", type=int, default=30, help="Number of repeated random splits")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_baseline",
        help="Directory to save per-repeat and summary CSV files",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = collect_csv_files(data_root, args.systems)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_root}")

    all_results: List[pd.DataFrame] = []
    print(f"Found {len(csv_files)} dataset files.")

    for csv_path in csv_files:
        print(f"Running: {csv_path}")
        result_df = evaluate_dataset(
            csv_path=csv_path,
            repeats=args.repeats,
            test_size=args.test_size,
            random_seed=args.seed,
        )
        all_results.append(result_df)

    results_df = pd.concat(all_results, ignore_index=True)
    results_path = output_dir / "baseline_linear_regression_all_runs.csv"
    results_df.to_csv(results_path, index=False)

    summary_df = build_summary(results_df)
    summary_path = output_dir / "baseline_linear_regression_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    overall_df = build_overall(results_df, summary_df)
    overall_path = output_dir / "baseline_linear_regression_overall.csv"
    overall_df.to_csv(overall_path, index=False)

    print("\nDone.")
    print(f"Per-run results saved to: {results_path}")
    print(f"Per-dataset summary saved to: {summary_path}")
    print(f"Overall summary saved to: {overall_path}")


if __name__ == "__main__":
    main()
