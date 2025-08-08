import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing.pipeline import load_and_prepare_data
from utils.plotting import plot_actual_vs_predicted


def baseline_naive(test_df, lag_column="lag_close_1", target_column="target"):
    y_true = test_df[target_column]
    y_pred = test_df[lag_column]

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
    print("📦 Loading data...")
    X_train, y_train, X_test, y_test = load_and_prepare_data("BBCA.JK", use_validation=False)

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    print("🔁 Running Naive Baseline...")
    result = baseline_naive(test_df)

    print(f"✅ Done. RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f}, R2={result['r2']:.4f}")


    with mlflow.start_run(run_name="Naive_Baseline"):
        mlflow.log_param("model_type", "naive_lag_1")
        mlflow.log_metric("rmse", result["rmse"])
        mlflow.log_metric("mae", result["mae"])
        mlflow.log_metric("r2", result["r2"])

        # Plot
        plot_path = plot_actual_vs_predicted(result["y_true"], result["y_pred"], "naive_baseline")
        mlflow.log_artifact(plot_path, artifact_path="plots")
