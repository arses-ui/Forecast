from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "date",
    "symbol",
    "securityId",
    "securityName",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "prevClose",
    "52wHigh",
    "52wLow",
    "trades",
    "avgPrice",
    "marketCap",
]
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "ma_5", "volatility_10"]
NUMERIC_COLUMNS = [column for column in RAW_COLUMNS if column not in {"date", "symbol", "securityName"}]


def load_market_dataframe(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    rows: list[list[str]] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for line_number, row in enumerate(reader, start=2):
            if len(row) < len(RAW_COLUMNS):
                raise ValueError(f"{path} has an incomplete row at line {line_number}.")
            rows.append(row[: len(RAW_COLUMNS)])

    dataframe = pd.DataFrame(rows, columns=RAW_COLUMNS)
    dataframe["date"] = pd.to_datetime(dataframe["date"], errors="raise")

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe = dataframe.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = dataframe.groupby("symbol", group_keys=False)

    dataframe["next_close"] = grouped["close"].shift(-1)
    dataframe["target"] = np.where(
        dataframe["next_close"].isna(),
        np.nan,
        (dataframe["next_close"] > dataframe["close"]).astype(np.float32),
    )
    dataframe["ma_5"] = grouped["close"].transform(lambda series: series.rolling(5, min_periods=1).mean())
    dataframe["volatility_10"] = grouped["close"].transform(
        lambda series: series.pct_change().rolling(10, min_periods=2).std()
    )
    dataframe["volatility_10"] = dataframe.groupby("symbol")["volatility_10"].transform(
        lambda series: series.bfill().ffill()
    )
    dataframe["volatility_10"] = dataframe["volatility_10"].fillna(0.0)
    dataframe[FEATURE_COLUMNS] = dataframe[FEATURE_COLUMNS].astype(float)

    return dataframe


def choose_split_date(dataframe: pd.DataFrame, test_fraction: float = 0.2) -> pd.Timestamp:
    unique_dates = np.sort(dataframe["date"].dropna().unique())
    split_index = max(1, int(len(unique_dates) * (1 - test_fraction))) - 1
    return pd.Timestamp(unique_dates[split_index])


def build_sequence_arrays(
    dataframe: pd.DataFrame,
    window_size: int,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    columns = feature_columns or FEATURE_COLUMNS
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for _, group in dataframe.groupby("symbol"):
        if len(group) < window_size + 1:
            continue

        features = group[columns].to_numpy(dtype=np.float32)
        targets = group["target"].to_numpy(dtype=np.float32)
        for index in range(len(group) - window_size):
            label = targets[index + window_size]
            if np.isnan(label):
                continue
            sequences.append(features[index : index + window_size])
            labels.append(int(label))

    if not sequences:
        raise ValueError("No trainable sequences were created. Check the dataset size and split date.")

    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int32)
