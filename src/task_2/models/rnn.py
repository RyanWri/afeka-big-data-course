import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import layers, Sequential, callbacks
from src.task_2.evaluation.model_evaluation import (
    run_model_evaluation,
    plot_training_history,
)


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(0, len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def rnn_preprocess(df: pd.DataFrame, sequence_length: int):
    # Drop rows with missing values
    df = df.replace("?", np.nan)
    df = df.dropna()
    # Normalize the data
    scaler = MinMaxScaler()
    df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
    # Using 'Global_active_power' as the target variable
    data = df[["Global_active_power"]].values
    X, y = create_sequences(data, sequence_length)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def build_rnn_model(seq_length):
    # Build the RNN model
    rnn_model = Sequential()
    rnn_model.add(
        layers.SimpleRNN(100, return_sequences=False, input_shape=(seq_length, 1))
    )
    rnn_model.add(layers.Dense(1))

    # Compile the model
    rnn_model.compile(optimizer="adam", loss="mse")

    # Print the model summary
    rnn_model.summary()

    return rnn_model


def run_rnn_model_e2e(full_df):
    sequence_length = 60
    X_train, X_test, y_train, y_test = rnn_preprocess(full_df, sequence_length)
    rnn_model = build_rnn_model(sequence_length)
    early_stopping = callbacks.EarlyStopping(
        monitor="loss", patience=2, restore_best_weights=True
    )
    history = rnn_model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        callbacks=[early_stopping],
    )

    plot_training_history(history)

    # Evaluate the model
    loss = rnn_model.evaluate(X_test, y_test)
    print(f"Test loss: {loss}")
    # Make predictions
    predictions = rnn_model.predict(X_test)

    rnn_results = run_model_evaluation(y_test, predictions)
    return rnn_results
