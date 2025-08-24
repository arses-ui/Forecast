import pandas as pd
import numpy as np 
import datetime
import time
from nepse_scraper import Nepse_scraper
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_sequences(group, window_size=7, feature_cols=["open", "high", "low", "close", "volume", "ma_5", "volatility_10"]
):
    X = []
    data = group[feature_cols].values
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
    return np.array(X)


def data_formatting(symbol, window_size = 7, feature_cols=["open", "high", "low", "close", "volume", "ma_5", "volatility_10"]): 

    # 1. Fetch data from NEPSE
    scraper = Nepse_scraper()
    start_date = datetime.date.today()

    all_data = []
    start_date = datetime.date.today()
    end_date = start_date - datetime.timedelta(days=15)  # last 10 days for demo

    all_data = []

    # loop day by day
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
        current -= datetime.timedelta(days=1)  # go backwards
        time.sleep(0.1)
                                        
    # Sort by symbol and date to keep order
    df = pd.DataFrame(all_data)
    df = df.sort_values(by=["date"])

    df["ma_5"] = df["close"].rolling(5).mean()
    df["volatility_10"] = df["close"].pct_change().rolling(10).std()

    # Shift close price by -1 (next day’s close)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    scaler = joblib.load("scaler.joblib")

    df[feature_cols] = scaler.transform(df[feature_cols])

    X = build_sequences(df, window_size=window_size, feature_cols=feature_cols)
    return X[-1].reshape(1, window_size, len(feature_cols))  # latest 7-day window only


def inference( symbol): 
    model =  Sequential([
    LSTM(units=50, return_sequences=False, input_shape=(7, 7)),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1, activation= "sigmoid")
])
    model.load_weights('forecast_model.weights.h5')
    X = data_formatting(symbol)
    prediction = model.predict(X)
    number = (prediction[0]>0.5).astype(int)    
    if number == 1:
        return "UP"
    else:    
        return "DOWN"


#just for testing purpose
if __name__ == '__main__':
    print(inference("NABIL"))