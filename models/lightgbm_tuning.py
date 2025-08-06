# run_lightgbm_tuning.py

import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing.pipeline import load_and_prepare_data


def tune_lightgbm(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 300],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "num_leaves": [15, 31]
    }

    model = LGBMRegressor(random_state=42)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_


def evaluate_model(model, X_test, y_test, best_params):
    with mlflow.start_run(run_name="Tuning_LightGBM"):
        mlflow.log_params(best_params)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, artifact_path="lightgbm_tuned_model")

        return {
            "model": model,
            "y_pred": y_pred,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }


if __name__ == "__main__":
    print("📦 Loading and preparing data for LightGBM tuning...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data("BBCA.JK", use_validation=True)

    print("🔍 Starting hyperparameter tuning for LightGBM...")
    model, best_params = tune_lightgbm(X_train, y_train)

    print("✅ Best Parameters Found:", best_params)

    print("📊 Evaluating tuned model on test set...")
    result = evaluate_model(model, X_test, y_test, best_params)

    print(f"\n✅ Done. [LightGBM Tuning] Metrics:")
    print(f"RMSE: {result['rmse']:.2f}")
    print(f"MAE : {result['mae']:.2f}")
    print(f"R²  : {result['r2']:.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(result["y_pred"], label="Predicted", linewidth=2)
    plt.title("Tuned LightGBM Forecast vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
