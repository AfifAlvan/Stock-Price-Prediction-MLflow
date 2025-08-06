# run_xgboost.py

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

    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    print(f"\n✅ Done.[XGBoost Train] Metrics:")
    print(f"RMSE: {rmse_train:.2f}")
    print(f"MAE : {mae_train:.2f}")
    print(f"R²  : {r2_train:.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    with mlflow.start_run(run_name="XGBoost"):
        # Log model params
        params = model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgboost_model",
            input_example=input_example,
            signature=signature
        )

        return {
            "model": model,
            "y_pred": y_pred,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }


if __name__ == "__main__":
    print("📦 Loading and preparing data for XGBoost...")

    # Load preprocessed data (tanpa val set karena tidak digunakan dalam training XGBoost ini)
    X_train, y_train, X_test, y_test = load_and_prepare_data(ticker="BBCA.JK", use_validation=False)

    print("🚀 Training XGBoost...")
    model = train_xgboost(X_train, y_train)

    print("📊 Evaluating model...")
    results = evaluate_model(model, X_test, y_test)

    print(f"\n✅ Done. [XGBoost Test] Metrics:")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"MAE : {results['mae']:.2f}")
    print(f"R²  : {results['r2']:.4f}")

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(results["y_pred"], label="Predicted", linewidth=2)
    plt.title("📈 XGBoost Forecast vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
