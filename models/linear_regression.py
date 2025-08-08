import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
from utils.plotting import plot_actual_vs_predicted  # ✅ import plot helper

def train_linear_regression(X_train, y_train, X_test, y_test):
    model_name = "LinearRegression"
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Logging model type
    mlflow.log_param("model_type", model_name)
    for key, value in model.get_params().items():
        mlflow.log_param(key, value)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log model
    signature = infer_signature(X_test, y_pred)
    input_example = X_test.iloc[:1] if hasattr(X_test, "iloc") else X_test[:1]
    mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)

    # Log plot (🔥 penting)
    plot_path = plot_actual_vs_predicted(y_test, y_pred, model_name)
    mlflow.log_artifact(plot_path, artifact_path="plots")

    return {
        "model": model,
        "y_pred": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

# Optional test runner
if __name__ == "__main__":
    from preprocessing.pipeline import load_and_prepare_data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data("BBCA.JK")
    
    with mlflow.start_run(run_name="LinearRegression-Standalone"):
        result = train_linear_regression(X_train, y_train, X_test, y_test)
        print(f"✅ Done. RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f}, R2={result['r2']:.4f}")
