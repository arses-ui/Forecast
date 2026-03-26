# Forecast

Small NEPSE stock-direction experiment built around a 7-day LSTM classifier. The model predicts whether the next trading day's close will be higher (`UP`) or not (`DOWN`) for a given symbol.

## Repo contents

- `dataloader.ipynb`: original notebook used to scrape one year of NEPSE market data.
- `forecast_model.ipynb`: original Colab notebook used for training.
- `dataset_utils.py`: local CSV loader that repairs the malformed committed dataset by recomputing engineered columns per symbol.
- `train_model.py`: local training entrypoint that replaces the hardcoded Colab paths.
- `dataloader.py`: inference helper that loads the saved scaler and model weights, fetches recent NEPSE data, and returns `UP` or `DOWN`.

## Setup

Use Python `3.11` or `3.12`. The current machine here is running Python `3.14`, and TensorFlow is not available for that runtime, so training cannot be executed there.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train_model.py --dataset nepse_full_1year.csv --epochs 30
```

This writes:

- `forecast_model.weights.h5`
- `scaler.joblib`

You can optionally choose a manual split date:

```bash
python train_model.py --split-date 2025-06-20
```

## Run inference

```bash
python dataloader.py
```

`dataloader.py` defaults to symbol `NABIL` in the local test block, but you can import `inference(symbol)` from another script.

## Why the current LSTM struggles

The repo has a few concrete problems that make the results weak:

- The committed CSV contains malformed rows, so a plain `pandas.read_csv()` can fail.
- The original training notebook forward-fills and back-fills `ma_5`, `volatility_10`, and `next_close` across the entire dataframe instead of within each symbol. On this dataset that contaminates `1,685` moving-average rows, `4,041` volatility rows, and fills all `443` terminal `next_close` values.
- The model scales raw prices globally across all symbols. In this dataset, `close` ranges from `7.9` to `52,700`, so the model learns nominal price level and symbol identity more easily than directional movement.
- The task itself is noisy. A simple majority-class baseline already gets about `54.8%` accuracy on the notebook split, and a quick logistic-regression baseline on the same 7-day windows collapses to the same score by predicting all `DOWN`.
- The model only sees 7 days of history and uses raw OHLCV values. For next-day direction, that is usually too little signal for a cross-company model unless you move to returns, per-symbol normalization, and stricter validation.

## Recommended next steps

- Convert `open/high/low/close` to returns or percent changes instead of raw prices.
- Normalize per symbol instead of with one global scaler.
- Add a validation split and early stopping so the model does not simply optimize training loss for 100 epochs.
- Compare against simpler baselines first: zero rule, logistic regression, and tree-based models on returns.
- Train separate models by sector or by liquidity bucket instead of one model over every listed instrument.
