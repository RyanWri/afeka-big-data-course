from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def run_model_evaluation(y_test, predictions) -> dict:
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, predictions)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, predictions)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Calculate R-squared (R²) value
    r2 = r2_score(y_test, predictions)

    # Create a dictionary to store the evaluation metrics
    evaluation_metrics = {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "R-squared (R²) value": r2,
    }

    return evaluation_metrics


def plot_eval_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="True")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.show()


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
