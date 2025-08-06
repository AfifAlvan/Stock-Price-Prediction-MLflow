import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing.pipeline import load_and_prepare_data


def tune_xgboost(X_train, y_train, X_val, y_val):
    param_grid = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    }

    grid = GridSearchCV(
        estimator=XGBRegressor(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)

    # MLflow logging
    with mlflow.start_run(run_name="Tuned_XGBoost"):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_r2", r2)

        mlflow.sklearn.log_model(best_model, artifact_path="xgboost_tuned_model")

    return best_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    print("📦 Loading and preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(ticker="BBCA.JK", use_validation=True)

    print("🔍 Starting hyperparameter tuning for XGBoost...")
    best_model = tune_xgboost(X_train, y_train, X_val, y_val)

    print("🧪 Evaluating best model on test set...")
    y_pred, metrics = evaluate_model(best_model, X_test, y_test)

    print("✅ Tuning complete. Best model:")
    print(best_model)
    print(f"\n📊 Test Metrics:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE : {metrics['mae']:.2f}")
    print(f"R²  : {metrics['r2']:.4f}")
    

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2)
    plt.title("📈 Tuned XGBoost Forecast vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
