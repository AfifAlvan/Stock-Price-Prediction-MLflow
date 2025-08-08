# models/run_xgboost_tuning.py

import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature

from preprocessing.pipeline import load_and_prepare_data


def tune_xgboost(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    }

    grid = GridSearchCV(
        estimator=XGBRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, {"rmse": rmse, "mae": mae, "r2": r2}


def log_to_mlflow(model, X_test, y_test, y_pred, metrics, best_params):
    with mlflow.start_run(run_name="Tuned_XGBoost"):
        mlflow.log_params(best_params)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="xgboost_tuned_model",
            input_example=X_test.iloc[:1],
            signature=signature
        )

        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label="Actual", linewidth=2)
        plt.plot(y_pred, label="Predicted", linewidth=2)
        plt.title("Tuned XGBoost Forecast vs Actual")
        plt.xlabel("Time Step")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = "xgboost_tuned_forecast_vs_actual.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        plt.close()


if __name__ == "__main__":
    print("📦 Loading and preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(ticker="BBCA.JK", use_validation=True)

    print("🔍 Starting hyperparameter tuning for XGBoost...")
    best_model, best_params = tune_xgboost(X_train, y_train)

    print("🧪 Evaluating on test set...")
    y_pred, metrics = evaluate_model(best_model, X_test, y_test)

    print("✅ Tuning complete. Best model:")
    print(best_model)
    print(f"\n📊 Test Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    log_to_mlflow(best_model, X_test, y_test, y_pred, metrics, best_params)
