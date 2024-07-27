import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.task_2.data_modification.augmentation import augment_sequence_data, add_noise


def split_data_x_and_y(df) -> tuple:
    # Prepare the data with lagged features
    df["lag_1"] = df["Global_active_power"].shift(1)
    df["lag_2"] = df["Global_active_power"].shift(2)

    # Define the target variable and features
    features = ["lag_1", "lag_2"]
    X = df[features].values
    y = df["Global_active_power"].values

    return X, y


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        a = data[i : (i + seq_length), 0]
        X.append(a)
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


def preprocess_sequence(df: pd.DataFrame, sequence_length: int):
    # Scale the data
    scaler = MinMaxScaler()

    # Drop rows with missing values
    df = df.replace("?", np.nan)
    df = df.dropna()

    # Normalize the data
    df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])

    # Using 'Global_active_power' as the target variable
    data = df[["Global_active_power"]].values
    X, y = create_sequences(data, sequence_length)

    return X, y


def augment_sequence(df: pd.DataFrame, sequence_length: int):
    # Scale the data
    scaler = MinMaxScaler()

    # Drop rows with missing values
    df = df.replace("?", np.nan)
    df = df.dropna()

    # Normalize the data
    df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])

    # Using 'Global_active_power' as the target variable
    data = df[["Global_active_power"]].values
    X, y = create_sequences(data, sequence_length)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Augment the data with noise
    X_augmented = add_noise(X)
    y_augmented = y  # Labels remain the same

    return X_augmented, y_augmented
