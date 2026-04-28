# M148 Capstone Project

This repository contains the journey data engineering, modeling, CTMC experiments, notebooks, figures, and submission helpers for the capstone project.

## Repository Layout

- `data/`: local data only; ignored by Git except directory placeholders.
- `src/data_engineering/`: data loading, cleaning, and feature engineering scripts.
- `src/models/`: CTMC, tabular baselines, Optuna tuning, and submission scripts.
- `src/visualizations/`: plotting helpers.
- `notebooks/prototyping/`: exploratory notebooks and older working notebooks.
- `notebooks/benchmarks/`: model comparison and submission notebooks.
- `figures/`: versioned report figures.
- `results/`: local benchmark outputs, submissions, and model artifacts; ignored by Git except placeholders.
- `docs/`: project writeups and method notes.

## Environment Setup

Do not commit virtual environments. Create one locally after cloning:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Large data files are not stored in Git. Put local files under `data/`, using this convention when possible:

- `data/raw/`: original source files.
- `data/cleaned/`: cleaned intermediate files.
- `data/feature_engineered/`: flattened and modeled feature tables.

The current scripts still read the established project filenames from `data/`, such as `journeys_flattened.parquet`, `journey_training_optionA.parquet`, and `open_journeys1.csv`.

## Common Commands

Run CTMC submissions:

```powershell
python src/models/ctmc_submission.py
```

Run tabular baseline submissions:

```powershell
python src/models/tabular_submission.py
```

Run Optuna tuning:

```powershell
python src/models/optuna_tuning.py
```

Open the main benchmark notebook:

```powershell
jupyter notebook notebooks/benchmarks/ctmc_exploration.ipynb
```
