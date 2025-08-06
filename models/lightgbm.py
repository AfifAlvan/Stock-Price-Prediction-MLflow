# run_lightgbm.py

import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

from preprocessing.pipeline import load_and_prepare_data

def train_lightgbm(X_train, y_train, X_val, y_val):
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )

    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    print(f"\n✅ Done.[Lightgbm Train] Metrics:")
    print(f"RMSE: {rmse_train:.2f}")
    print(f"MAE : {mae_train:.2f}")
    print(f"R²  : {r2_train:.4f}")
    return model

def evaluate_model(model, X_test, y_test):
    with mlflow.start_run(run_name="LightGBM"):
        # Log parameters
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
        mlflow.sklearn.log_model(model, "lightgbm_model")

        return {
            "model": model,
            "y_pred": y_pred,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

if __name__ == "__main__":
    print("📦 Loading and preparing data for LightGBM...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data("BBCA.JK", use_validation=True)

    print("📈 Training LightGBM...")
    model = train_lightgbm(X_train, y_train, X_val, y_val)

    print("📊 Evaluating model...")
    result = evaluate_model(model, X_test, y_test)

    print(f"\n✅ Done. [LightGBM Test] Metrics:")
    print(f"RMSE: {result['rmse']:.2f}")
    print(f"MAE : {result['mae']:.2f}")
    print(f"R²  : {result['r2']:.4f}")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(result["y_pred"], label="Predicted", linewidth=2)
    plt.title("LightGBM Forecast vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
