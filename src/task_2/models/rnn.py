import json
from keras import layers, Sequential, callbacks
from src.task_2.preprocessing.rnn import preprocess as rnn_preprocess
from src.task_2.preprocessing.rnn import train_test_split_sequence as rnn_split
from src.task_2.evaluation.model_evaluation import run_model_evaluation


def build_rnn_model(seq_length, X_train):
    # Build the RNN model
    rnn_model = Sequential()
    rnn_model.add(
        layers.SimpleRNN(4, return_sequences=False, input_shape=(seq_length, 1))
    )
    rnn_model.add(layers.Dense(1))

    # Compile the model
    rnn_model.compile(optimizer="adam", loss="mse")

    # Print the model summary
    rnn_model.summary()

    return rnn_model


def run_rnn_model_e2e(full_df):
    sequence_length = 120
    X, y = rnn_preprocess(full_df, sequence_length)
    X_train, X_test, y_train, y_test = rnn_split(X, y)
    rnn_model = build_rnn_model(sequence_length, X_train)
    early_stopping = callbacks.EarlyStopping(
        monitor="loss", patience=2, mode="min", restore_best_weights=True
    )
    history = rnn_model.fit(
        X_train,
        y_train,
        epochs=2,
        batch_size=256,
        callbacks=[early_stopping],
    )

    # Evaluate the model
    loss = rnn_model.evaluate(X_test, y_test)
    print(f"Test loss: {loss}")
    # Make predictions
    predictions = rnn_model.predict(X_test)

    rnn_results = run_model_evaluation(y_test, predictions)
    print(json.dumps(rnn_results, indent=1))
