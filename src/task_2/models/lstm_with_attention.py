from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from preprocessing.lstm import split_data_append_lagged_features


# Attention mechanism
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = tf.keras.layers.Permute((2, 1))(inputs)
    a = tf.keras.layers.Dense(1, activation="softmax")(a)
    a_probs = tf.keras.layers.Permute((2, 1))(a)
    output_attention_mul = tf.keras.layers.Multiply()([inputs, a_probs])
    return output_attention_mul


def build_lstm_with_attention_model(X_train):
    # Design the LSTM model with attention
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_out = tf.keras.layers.LSTM(50, return_sequences=True, activation="relu")(
        inputs
    )
    attention_out = attention_3d_block(lstm_out)
    attention_out = tf.keras.layers.Flatten()(attention_out)
    output = tf.keras.layers.Dense(1)(attention_out)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer="adam", loss="mse")

    # Define the early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3, restore_best_weights=True
    )

    return model, early_stopping


def run_lstm_with_attention_e2e(full_df):
    # Scale the data
    scaler = MinMaxScaler()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data_append_lagged_features(
        full_df, scaler=scaler
    )

    # build LSTM model
    model, early_stopping = build_lstm_with_attention_model(X_train)
    # Train the LSTM model with early stopping
    history = model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=128,
        verbose=2,
        validation_data=(X_test, y_test),
        shuffle=False,
        callbacks=[early_stopping],
    )

    # Train the LSTM model without validation
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=128,
        verbose=2,
        shuffle=False,
    )
