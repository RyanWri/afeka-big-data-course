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

    plot_eval_results(y_test, predictions)
    return evaluation_metrics


def plot_eval_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="True")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.show()
