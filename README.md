# Concrete Strength Predictor — single-screen edition

A deliberately small Streamlit app: type a mix, press one button, read the
strength. Everything fits on one screen with no scrolling.

This is a second, simpler front end for the same six trained models served by
the full application in `../app`. It shares nothing with it at runtime — the
folder is self-contained and can be deployed on its own.

## The two tabs

**Predictor** — quick presets, the mix ingredients as typed numbers with
stepper buttons, a dial showing the result against the EN 206 strength
classes, the predicted value and grade, and six derived mix indicators
(water/cement, total binder, binder ratio, total aggregate, paste volume,
strength class). A status bar reports the water/cement ratio and warns when an
input falls outside the range the models were trained on.

Nothing is predicted until **Predict Strength** is pressed. Editing a value or
switching model never runs the model; the last result stays on screen and the
status bar says when it no longer matches the boxes.

Every model and scaler is loaded once when the server starts, behind a
"Loading models" spinner, so a press costs only the prediction itself. The wait
is `scikit-learn`'s import (about 4.5 s), not the model files (0.3 s for all
six).

**Analytics** — how the app is actually being used: total predictions, models
used, mean predicted strength, and a per-model comparison of run count and
mean prediction. Counters are written to `data/stats.json` and survive a
restart.

## Models

Six models, chosen with the two selectors in the header. Metrics are measured
on the same held-out test split used throughout the research.

| Feature set | Model | Test R² | RMSE | MAE |
|---|---|---|---|---|
| 9 inputs (with curing temperature) | Gradient Boosting — Champion | 0.9159 | 4.414 | 3.109 |
| 9 inputs | LightGBM | 0.9131 | 4.485 | 3.136 |
| 9 inputs | XGBoost | 0.9107 | 4.548 | 3.118 |
| 8 inputs (no curing temperature) | XGBoost — best overall | 0.9412 | 3.892 | 2.615 |
| 8 inputs | LightGBM | 0.9336 | 4.136 | 2.917 |
| 8 inputs | Gradient Boosting | 0.9326 | 4.167 | 2.883 |

Curing temperature is a synthetic variable, which is why the 8-input family
scores higher — the penalty applied when it was generated adds variance the
mix-design features cannot explain.

## Running it

Double-click **`run.bat`**. It builds the virtual environment if it is
missing, installs the requirements, and opens the app at
<http://localhost:8502>.

Manually:

```bash
cd app2
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m streamlit run streamlit_app.py --server.port 8502
```

Self-test:

```bash
.venv/Scripts/python.exe scripts/selftest.py
```

## Layout

```
app2/
├── run.bat                  one-click launcher (Windows)
├── streamlit_app.py         the whole interface
├── requirements.txt         pinned dependencies
├── .streamlit/config.toml   dark theme
├── src/
│   ├── config.py            features, model registry, presets, strength classes
│   ├── predictor.py         model loading and prediction
│   └── stats.py             persistent usage counters
├── artifacts/app_meta.json  baked model metrics and feature ranges
├── models/  scalers/        the six models and their scalers
└── data/                    stats.json, written at runtime
```

`artifacts/app_meta.json` holds the per-model test metrics and the per-feature
training ranges, computed once by a build script. That is why this app needs no
spreadsheet reader and starts in well under a second.

`scikit-learn` is pinned to **1.6.1** because the `.joblib` files were pickled
with it.
