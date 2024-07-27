# refer to this notebook on kaggle
# https://www.kaggle.com/code/kakiwang/time-series-data-analysis-using-lstm-tutorial
# https://www.kaggle.com/code/yassinesfaihi/lstm-time-series-household-power-consumption


import pandas as pd
from models.rnn import build_rnn_model
from src.task_2.preprocessing.sequence import preprocess as rnn_preprocess
from src.task_2.preprocessing.sequence import train_test_split_sequence as rnn_split
from evaluation.model_evaluation import run_model_evaluation
import keras
import numpy as np

df = pd.read_csv(
    "data\\household_power_consumption_1867734.csv",
    parse_dates={"dt": ["Date", "Time"]},
    infer_datetime_format=True,
    low_memory=False,
)

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)


sequence_length = 60
X, y = rnn_preprocess(df, sequence_length)
X_train, X_test, y_train, y_test = rnn_split(X, y)
rnn_model = build_rnn_model(sequence_length, X_train)
# Train the model with early stoppage

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
history = rnn_model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
)

# Evaluate the model
loss = rnn_model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Make predictions
predictions = rnn_model.predict(X_test)

# Plotting results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test, label="True")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()

run_model_evaluation(y_test, predictions)
