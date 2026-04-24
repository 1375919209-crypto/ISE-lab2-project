import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_comparison/baseline_vs_rf_summary_comparison.csv")

data = [
    df["MAPE_improvement_pct"],
    df["MAE_improvement_pct"],
    df["RMSE_improvement_pct"]
]

plt.figure(figsize=(6, 4))

plt.boxplot(
    data,
    labels=["MAPE", "MAE", "RMSE"],
    showmeans=True,
    meanprops=dict(marker='^', markerfacecolor='black', markersize=5),
    medianprops=dict(color='black'),
)

plt.axhline(0, linestyle="--", linewidth=1, color="gray")

plt.ylabel("Improvement (%)")
plt.xlabel("Metric")

plt.tight_layout()
plt.savefig("improvement_distribution.png", dpi=300)
plt.close()