# Configuration Performance Learning (Lab 2)

This project implements and evaluates configuration performance prediction models for configurable software systems.

## Overview

We compare:
- **Linear Regression** (baseline)
- **Random Forest** (proposed model)

The goal is to predict performance metrics (e.g., runtime) of unseen configurations.

## Project Structure


datasets/                 Original Lab 2 datasets
figures/                  Generated figures
results/
  results_baseline/        Linear Regression baseline results
  results_random_forest/   Random Forest results
  results_comparison/      Comparison and statistical test results
src/                       Python source code


## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib

## How to Run

### 1. Run baseline (Linear Regression)
python src/lab2_baseline.py


### 2. Run Random Forest
python src/lab2_random_forest.py


### 3. Compare results
python src/compare_baseline_vs_rf.py


### 4. Generate plot
python src/improvement_distribution_png.py


## Output

The project generates the following outputs:

### Baseline (Linear Regression)
- `baseline_linear_regression_all_runs.csv`  
  Results of all runs across datasets
- `baseline_linear_regression_summary.csv`  
  Dataset-level averaged results
- `baseline_linear_regression_overall.csv`  
  Overall aggregated performance

### Random Forest
- `random_forest_all_runs.csv`  
  Results of all runs across datasets
- `random_forest_summary.csv`  
  Dataset-level averaged results
- `random_forest_overall.csv`  
  Overall aggregated performance

### Comparison and Statistical Analysis
- `baseline_vs_rf_summary_comparison.csv`  
  Dataset-level comparison and improvement percentages
- `baseline_vs_rf_wilcoxon.csv`  
  Wilcoxon signed-rank test results
- `baseline_vs_rf_overall_report.csv`  
  Overall comparison summary

### Visualization
- `figures/improvement_distribution.png`  
  Boxplot showing improvement distribution across datasets

## Reproducibility

See:
- `manual.pdf`
- `replication.pdf`
- `requirements.pdf`

## Yihan Lan

Yihan Lan  
University of Birmingham
