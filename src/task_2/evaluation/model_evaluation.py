from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


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
