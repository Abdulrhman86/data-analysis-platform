# Data Analysis Platform

A no-code **data mining & analysis** web application built with Streamlit. Upload a
dataset and move through a guided workflow — data-quality assessment, preprocessing,
visualization, dashboards, and machine learning — without writing any code.

> Undergraduate graduation project (DSAI).

## Features

A six-step guided workflow:

1. **Upload Data** — CSV / Excel import with automatic date detection and a data preview.
2. **Data Quality** — automated assessment with severity-ranked recommendations
   (missing values, outliers, skew, high cardinality, duplicates, …).
3. **Preprocessing** — missing-value handling, scaling/normalization, binning, encoding,
   datetime features, feature engineering & selection, plus a recordable/replayable pipeline.
4. **Visualization** — single-variable, relationship, distribution, time-series,
   summary-statistics, and advanced charts (parallel coordinates, radar, 3D scatter, sunburst).
5. **Dashboard** — assemble saved charts into interactive dashboards.
6. **Machine Learning** — train, evaluate, and export classification & regression models
   (Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes,
   Linear / Ridge / Lasso, …).

## Tech stack

Python 3.11 · Streamlit · pandas · NumPy · scikit-learn · Plotly · Matplotlib · Seaborn · SciPy.

## Project structure

```
data_mining/
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .gitignore
└── app/                      # application package
    ├── home.py               # app entry point (landing page)
    ├── config.py             # theme, colors, app settings
    ├── pages/                # the six workflow pages
    ├── utils/                # preprocessing, visualization, charts, and ML modules
    ├── static/               # images
    └── tests/                # pytest suite
```

## Setup & run

Requires **Python 3.11**.

```bash
# 1. From this folder (the project root), create & activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (from the project root)
streamlit run app/home.py
```

The app then opens in your browser at <http://localhost:8501>.

## Testing

```bash
pip install -r requirements-dev.txt
pytest app/tests
```

## Status

Active development. Recent work: a leakage-free scikit-learn training pipeline,
security hardening, disk-persisted dashboards, a shared chart core, additional
models with hyperparameter tuning, a predict-on-new-data tab, and a growing
pytest suite.
