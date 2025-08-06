# run_linear_regression.py

import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature

from preprocessing.pipeline import load_and_prepare_data


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Evaluate on training data (optional, for checking overfitting)
    y_train_pred = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    print(f"\n✅ Done.[Linear Regression Train] Metrics:")
    print(f"RMSE: {rmse_train:.2f}")
    print(f"MAE : {mae_train:.2f}")
    print(f"R²  : {r2_train:.4f}")
    return model


def evaluate_model(model, X_test, y_test):
    with mlflow.start_run(run_name="LinearRegression"):
        # Log model parameters
        params = model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model with input example and signature
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="linear_regression_model",
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
    print("📦 Loading and preparing data for Linear Regression...")

    # Load preprocessed data
    X_train, y_train, X_test, y_test = load_and_prepare_data(ticker="BBCA.JK", use_validation=False)

    # Train model
    print("📈 Training Linear Regression...")
    model = train_linear_regression(X_train, y_train)

    # Evaluate
    print("📊 Evaluating model...")
    result = evaluate_model(model, X_test, y_test)

    print(f"\n✅ Done.[Linear Regression Test] Metrics:")
    print(f"RMSE: {result['rmse']:.2f}")
    print(f"MAE : {result['mae']:.2f}")
    print(f"R²  : {result['r2']:.4f}")
    # Optional: plot prediction vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(result["y_pred"], label="Predicted", linewidth=2)
    plt.title("Linear Regression Forecast vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
