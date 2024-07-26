import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def split_data_append_lagged_features(full_df, scaler) -> tuple:
    # Prepare the data with lagged features
    full_df["lag_1"] = full_df["Global_active_power"].shift(1)
    full_df["lag_2"] = full_df["Global_active_power"].shift(2)
    full_df["lag_3"] = full_df["Global_active_power"].shift(3)

    # Drop any rows with NaN values created by the shift operation
    full_df.dropna(inplace=True)

    # Define the target variable and features
    features = ["lag_1", "lag_2", "lag_3"]
    X = full_df[features].values
    y = full_df["Global_active_power"].values

    # Scale the data
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # Reshape the input data to 3D for LSTM [samples, time steps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    return X_train, X_test, y_train, y_test


def build_lstm_model(X_train):
    # Design the LSTM model
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(
            50, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Define the early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3, restore_best_weights=True
    )

    return model, early_stopping


def predict_lstm_model(model, X_test, y_test, scaler):
    # Make predictions
    y_pred_test = model.predict(X_test)

    # Inverse transform the predictions and actual values
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    y_test_inv = scaler.inverse_transform(y_test)

    return y_pred_test_inv, y_test_inv


def plot_lstm_results(y_test, predictions):
    # Plot predictions vs actual values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual", color="blue")
    plt.plot(predictions, label="Predicted", color="red")
    plt.xlabel("Time")
    plt.ylabel("Global Active Power (kilowatts)")
    plt.title("Actual vs Predicted Global Active Power (LSTM)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def plot_training_history(history):
    # Plot the training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()


def run_lstm_model_e2e(full_df):
    # Scale the data
    scaler = MinMaxScaler()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data_append_lagged_features(
        full_df, scaler=scaler
    )

    # build LSTM model
    model, early_stopping = build_lstm_model(X_train)
    # Train the LSTM model with early stopping
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=128,
        verbose=2,
        validation_data=(X_test, y_test),
        shuffle=False,
        callbacks=[early_stopping],
    )

    plot_training_history(history)

    # Make predictions
    y_pred_test_inv, y_test_inv = predict_lstm_model(model, X_test, y_test, scaler)
    plot_lstm_results(y_test_inv, y_pred_test_inv)

    print("LSTM COMPLETED")
