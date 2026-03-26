from __future__ import annotations

import datetime
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nepse_scraper import Nepse_scraper

from dataset_utils import FEATURE_COLUMNS


MODEL_DIR = Path(__file__).resolve().parent


def build_sequences(group, window_size=7, feature_cols=FEATURE_COLUMNS):
    X = []
    data = group[feature_cols].values
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
    return np.array(X)


def data_formatting(symbol, window_size=7, feature_cols=FEATURE_COLUMNS):
    scraper = Nepse_scraper()
    start_date = datetime.date.today()

    end_date = start_date - datetime.timedelta(days=15)
    all_data = []

    current = start_date
    while current >= end_date:
        try:
            daily_response = scraper.get_today_price(current.strftime("%Y-%m-%d"))
            companies = daily_response.get("content", [])
            for c in companies:
                if c.get("symbol") == symbol:
                    all_data.append({
                        "date": c.get("businessDate"),
                        "symbol": c.get("symbol"),
                        "open": c.get("openPrice"),
                        "high": c.get("highPrice"),
                        "low": c.get("lowPrice"),
                        "close": c.get("closePrice"),
                        "volume": c.get("totalTradedQuantity"),
                    })
        except Exception as e:
            print(f"Skipped {current}: {e}")
        current -= datetime.timedelta(days=1)
        time.sleep(0.1)

    df = pd.DataFrame(all_data)
    if df.empty:
        raise ValueError(f"No recent data was returned for symbol {symbol}.")

    df = df.sort_values(by=["date"])

    df["ma_5"] = df["close"].rolling(5).mean()
    df["volatility_10"] = df["close"].pct_change().rolling(10).std()
    df[feature_cols] = df[feature_cols].bfill().ffill()

    if len(df) < window_size:
        raise ValueError(f"Not enough rows to build a {window_size}-day sequence for {symbol}.")

    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    df[feature_cols] = scaler.transform(df[feature_cols])

    window = df[feature_cols].tail(window_size).to_numpy(dtype=np.float32)
    return window.reshape(1, window_size, len(feature_cols))


def load_model(window_size=7, feature_count=7):
    try:
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorFlow is not installed. Use Python 3.11/3.12 and install requirements.txt before inference."
        ) from error

    model = Sequential(
        [
            LSTM(units=50, return_sequences=False, input_shape=(window_size, feature_count)),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1, activation="sigmoid"),
        ]
    )
    model.load_weights(MODEL_DIR / "forecast_model.weights.h5")
    return model


def inference(symbol):
    model = load_model(window_size=7, feature_count=len(FEATURE_COLUMNS))
    X = data_formatting(symbol)
    prediction = model.predict(X, verbose=0)
    number = int(prediction[0][0] >= 0.5)
    if number == 1:
        return "UP"
    else:
        return "DOWN"


if __name__ == "__main__":
    print(inference("NABIL"))
