# refer to this notebook on kaggle
# https://www.kaggle.com/code/kakiwang/time-series-data-analysis-using-lstm-tutorial
# https://www.kaggle.com/code/yassinesfaihi/lstm-time-series-household-power-consumption


import pandas as pd
from models.rnn import build_rnn_model
from preprocessing.rnn import preprocess as rnn_preprocess
from preprocessing.rnn import train_test_split_sequence as rnn_split

df = pd.read_csv(
    "data\\household_power_consumption_1867734.csv",
    parse_dates={"dt": ["Date", "Time"]},
    infer_datetime_format=True,
    low_memory=False,
    na_values=["nan", "?"],
    index_col="dt",
)

sequence_length = 60
X, y = rnn_preprocess(df, sequence_length)
X_train, X_test, y_train, y_test = rnn_split(X, y)
rnn_model = build_rnn_model(sequence_length, X_train)
# Train the model
history = rnn_model.fit(
    X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test)
)
