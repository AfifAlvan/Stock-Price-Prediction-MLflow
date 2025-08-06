# run_naive.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing.pipeline import load_and_prepare_data

def baseline_naive_forecast(test_df, lag_column="lag_close_1", target_column="target"):
    y_true = test_df[target_column].values
    y_pred = test_df[lag_column].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_true": y_true,
        "y_pred": y_pred
    }

if __name__ == "__main__":
    print("📦 Loading and preparing data for baseline (Naive)...")

    # Load data without validation
    X_train, y_train, X_test, y_test = load_and_prepare_data("BBCA.JK", use_validation=False)

    # Combine X_test and y_test
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    print("📈 Running Naive Forecast (Baseline)...")
    result = baseline_naive_forecast(test_df)

    print(f"\n✅ Done. [Baseline Naive] Metrics:")
    print(f"RMSE: {result['rmse']:.2f}")
    print(f"MAE : {result['mae']:.2f}")
    print(f"R²  : {result['r2']:.4f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name="Baseline_Naive_Forecast"):
        mlflow.log_param("model_type", "naive_lag_1")
        mlflow.log_metric("rmse", result["rmse"])
        mlflow.log_metric("mae", result["mae"])
        mlflow.log_metric("r2", result["r2"])

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(result["y_true"], label="Actual", linewidth=2)
    plt.plot(result["y_pred"], label="Naive Forecast", linewidth=2)
    plt.title("Naive Baseline Forecast vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
