import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature

from preprocessing.pipeline import load_and_prepare_data
from utils.plotting import plot_actual_vs_predicted


def tune_linear_regression(X_train, y_train):
    param_grid = {
        "alpha": [0.1, 1.0],
        "fit_intercept": [True],
        "solver": ["auto", "svd"]
    }

    grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def evaluate_model(model, X_test, y_test, best_params):
    with mlflow.start_run(run_name="Ridge_Tuned"):
        mlflow.log_params(best_params)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "ridge_model", signature=signature)

        # Log plot
        plot_path = plot_actual_vs_predicted(y_test, y_pred, "ridge_tuned")
        mlflow.log_artifact(plot_path, artifact_path="plots")

        return {
            "model": model,
            "y_pred": y_pred,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }


if __name__ == "__main__":
    print("📦 Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data("BBCA.JK")

    print("🔍 Tuning Ridge Regression...")
    model, best_params = tune_linear_regression(X_train, y_train)
    print("✅ Best Params:", best_params)

    print("📊 Evaluating...")
    result = evaluate_model(model, X_test, y_test, best_params)

    print(f"✅ Done. RMSE={result['rmse']:.2f}, R2={result['r2']:.4f}")
