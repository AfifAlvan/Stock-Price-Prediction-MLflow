# models/run_xgboost.py

import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature

from preprocessing.pipeline import load_and_prepare_data


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, {"rmse": rmse, "mae": mae, "r2": r2}


def log_to_mlflow(model, X_test, y_test, y_pred, metrics):
    with mlflow.start_run(run_name="XGBoost"):
        # Log model params
        for key, value in model.get_params().items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log model
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="xgboost_model",
            input_example=X_test.iloc[:1],
            signature=signature
        )

        # Plot and log plot
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label="Actual", linewidth=2)
        plt.plot(y_pred, label="Predicted", linewidth=2)
        plt.title("XGBoost Forecast vs Actual")
        plt.xlabel("Time Step")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = "xgboost_forecast_vs_actual.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        plt.close()


if __name__ == "__main__":
    print("📦 Loading and preparing data for XGBoost...")
    X_train, y_train, X_test, y_test = load_and_prepare_data(ticker="BBCA.JK", use_validation=False)

    print("🚀 Training XGBoost...")
    model = train_xgboost(X_train, y_train)

    print("📊 Evaluating model...")
    y_pred, metrics = evaluate_model(model, X_test, y_test)

    print(f"\n✅ Done. [XGBoost Test] Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    log_to_mlflow(model, X_test, y_test, y_pred, metrics)
