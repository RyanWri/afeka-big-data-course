import keras
from keras import layers


def build_rnn_model(seq_length, X_train):
    # Build the RNN model
    rnn_model = keras.Sequential()
    rnn_model.add(
        layers.SimpleRNN(50, return_sequences=False, input_shape=(seq_length, 1))
    )
    rnn_model.add(layers.Dense(1))

    # Compile the model
    rnn_model.compile(optimizer="adam", loss="mse")

    # Print the model summary
    rnn_model.summary()

    return rnn_model
