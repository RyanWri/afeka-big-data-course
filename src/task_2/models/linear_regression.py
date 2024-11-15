from sklearn.linear_model import LinearRegression
from src.task_2.evaluation.model_evaluation import (
    run_model_evaluation,
    plot_eval_results,
)
from sklearn.model_selection import train_test_split


def split_data_append_lagged_features(full_df):
    # Create lagged features
    full_df["lag_1"] = full_df["Global_active_power"].shift(1)
    full_df["lag_2"] = full_df["Global_active_power"].shift(2)
    full_df["lag_3"] = full_df["Global_active_power"].shift(3)

    # Drop any rows with NaN values created by the shift operation
    full_df.dropna(inplace=True)

    # Define the target variable and features
    X = full_df[["lag_1", "lag_2", "lag_3"]]
    y = full_df["Global_active_power"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    return X_train, X_test, y_train, y_test


def run_linear_regression(X_train, y_train, X_test, y_test):
    # Initialize the models
    linear_reg = LinearRegression()

    # Train Linear Regression
    linear_reg.fit(X_train, y_train)

    return linear_reg


def run_linear_regression_e2e(full_df) -> dict:
    X_train, X_test, y_train, y_test = split_data_append_lagged_features(full_df)
    linear_reg = run_linear_regression(X_train, y_train, X_test, y_test)

    # Make predictions on the testing data
    y_pred_lr = linear_reg.predict(X_test)

    # Evaluate the model
    lr_results = run_model_evaluation(y_test, y_pred_lr)

    return lr_results
