from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.task_2.evaluation.model_evaluation import (
    plot_lstm_results,
    plot_training_history,
)
from src.task_2.preprocessing.sequence import preprocess_sequence
from keras import Input, Model, layers, callbacks


def build_lstm_model_with_attention(X_train):
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Define model input
    inputs = Input(shape=input_shape)

    # First LSTM layer
    lstm_out = layers.LSTM(20, activation="relu", return_sequences=True)(inputs)
    lstm_out = layers.BatchNormalization()(lstm_out)
    lstm_out = layers.Dropout(0.2)(lstm_out)

    # Attention layer
    query_value_attention_seq = layers.Attention()([lstm_out, lstm_out])

    # Second LSTM layer
    lstm_out = layers.LSTM(20, activation="relu")(query_value_attention_seq)
    lstm_out = layers.BatchNormalization()(lstm_out)
    lstm_out = layers.Dropout(0.2)(lstm_out)

    # Output layer
    outputs = layers.Dense(1)(lstm_out)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer="adam", loss="mse")

    # Define the early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor="loss", patience=2, restore_best_weights=True
    )

    return model, early_stopping


def run_lstm_model_with_attention_e2e(full_df):
    # Preprocess the data
    X, y = preprocess_sequence(full_df, sequence_length=60)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # build LSTM model
    model, early_stopping = build_lstm_model_with_attention(X_train)
    # Train the LSTM model with early stopping
    history = model.fit(
        X_train,
        y_train,
        epochs=12,
        batch_size=64,
        verbose=2,
        shuffle=False,
        callbacks=[early_stopping],
    )

    plot_training_history(history)

    # Make predictions
    y_pred_test = model.predict(X_test)

    # Inverse transform the predictions and actual values
    scaler = MinMaxScaler()
    scaler.fit_transform(full_df[full_df.columns[1:]])
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    y_test_inv = scaler.inverse_transform(y_test)
    plot_lstm_results(y_test_inv, y_pred_test_inv)

    print("LSTM with Attention COMPLETED")
