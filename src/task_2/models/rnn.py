import tensorflow as tf
from keras import Sequential, SimpleRNN, Dense


def build_rnn_model(seq_length, X_train):
    # Build the RNN model
    rnn_model = Sequential()
    rnn_model.add(
        SimpleRNN(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2]))
    )
    rnn_model.add(SimpleRNN(50, return_sequences=False))
    rnn_model.add(Dense(25))
    rnn_model.add(Dense(1))

    # Compile the model
    rnn_model.compile(optimizer="adam", loss="mean_squared_error")

    # Print the model summary
    rnn_model.summary()

    return rnn_model
