import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(0, len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def preprocess_sequence(df: pd.DataFrame, sequence_length: int):
    # Drop rows with missing values
    df = df.replace("?", np.nan)
    df = df.dropna()
    # Normalize the data
    scaler = MinMaxScaler()
    df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
    # Using 'Global_active_power' as the target variable
    data = df[["Global_active_power"]].values
    X, y = create_sequences(data, sequence_length)

    return X, y
