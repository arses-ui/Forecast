from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

from dataset_utils import FEATURE_COLUMNS, build_sequence_arrays, choose_split_date, load_market_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NEPSE direction classifier locally.")
    parser.add_argument("--dataset", default="nepse_full_1year.csv", help="Path to the CSV dataset.")
    parser.add_argument("--window-size", type=int, default=7, help="Number of trading days in each sequence.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--split-date", help="Optional YYYY-MM-DD cutoff. Newer rows become the test set.")
    parser.add_argument("--weights-out", default="forecast_model.weights.h5", help="Output path for model weights.")
    parser.add_argument("--scaler-out", default="scaler.joblib", help="Output path for the fitted scaler.")
    return parser.parse_args()


def import_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, Dropout, LSTM
        from tensorflow.keras.models import Sequential
    except ModuleNotFoundError as error:
        raise SystemExit(
            "TensorFlow is not installed. Use Python 3.11 or 3.12, then run `pip install -r requirements.txt`."
        ) from error

    return tf, Dense, Dropout, LSTM, Sequential


def make_model(window_size: int, feature_count: int):
    tf, Dense, Dropout, LSTM, Sequential = import_tensorflow()
    model = Sequential(
        [
            LSTM(units=50, return_sequences=False, input_shape=(window_size, feature_count)),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall"],
    )
    return model


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    dataframe = load_market_dataframe(dataset_path)

    split_timestamp = pd.Timestamp(args.split_date) if args.split_date else choose_split_date(dataframe)

    train_df = dataframe[dataframe["date"] <= split_timestamp].copy()
    test_df = dataframe[dataframe["date"] > split_timestamp].copy()

    scaler = MinMaxScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    x_train, y_train = build_sequence_arrays(train_df, args.window_size, FEATURE_COLUMNS)
    x_test, y_test = build_sequence_arrays(test_df, args.window_size, FEATURE_COLUMNS)

    model = make_model(args.window_size, len(FEATURE_COLUMNS))
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    baseline = np.zeros_like(y_test)

    print(f"Split date: {split_timestamp.date()}")
    print(f"Train sequences: {len(x_train)}")
    print(f"Test sequences: {len(x_test)}")
    print(
        "Model metrics:",
        {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "positive_rate": round(float(predictions.mean()), 4),
        },
    )
    print(
        "Zero baseline:",
        {
            "accuracy": round(float(accuracy_score(y_test, baseline)), 4),
            "positive_rate": round(float(baseline.mean()), 4),
        },
    )

    model.save_weights(args.weights_out)
    joblib.dump(scaler, args.scaler_out)


if __name__ == "__main__":
    main()
