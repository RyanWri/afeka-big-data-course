import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def preprocess(df: pd.DataFrame, sequence_length: int):
    # Normalize the data
    scaler = MinMaxScaler()
    df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
    # Using 'Global_active_power' as the target variable
    data = df[["Global_active_power"]].values
    X, y = create_sequences(data, sequence_length)

    return X, y


def train_test_split_sequence(X, y, train_size=0.8):
    # Split the data into train and test sets
    split = int(train_size * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test
