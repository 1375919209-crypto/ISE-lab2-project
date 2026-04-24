# Configuration Performance Learning (Lab 2)

This project implements and evaluates configuration performance prediction models for configurable software systems.

## Overview

We compare:
- **Linear Regression** (baseline)
- **Random Forest** (proposed model)

The goal is to predict performance metrics (e.g., runtime) of unseen configurations.

## Project Structure
datasets/ # original datasets (Lab 2)
src/ # source code
results/ # experiment results (CSV)
figures/ # generated plots


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

- `results/` contains CSV summaries
- `figures/` contains improvement distribution plot

## Reproducibility

See:
- `manual.pdf`
- `replication.pdf`
- `requirements.pdf`

## Yihan Lan

Yihan Lan  
University of Birmingham
