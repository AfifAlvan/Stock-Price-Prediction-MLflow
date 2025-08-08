import mlflow
from preprocessing.pipeline import load_and_prepare_data
from forecasting.future_forecasting import forecast_future_prices
from models.model_selector import get_best_model
import pandas as pd

if __name__ == "__main__":
    mlflow.start_run(run_name="BBCA_Forecast")  # 1 run untuk semua proses

    print("📦 Loading data...")
    df = pd.read_csv("data/bbca.csv")

    print("🧪 Preparing train/test sets...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data("BBCA.JK")

    print("🤖 Selecting best model...")
    # model_name, model, rmse, mae, r2 = get_best_model(X_train, y_train, X_test, y_test)
    model_name, model, rmse, mae, r2 = get_best_model(X_train, y_train, X_test, y_test, X_val, y_val)
    print("🔮 Forecasting 30 days ahead...")
    forecast_df = forecast_future_prices(model, df, steps=30)

    forecast_df.to_csv("outputs/results/forecast_30_days.csv", index=False)
    mlflow.log_artifact("outputs/results/forecast_30_days.csv")

    print("✅ Forecast saved to results/forecast_30_days.csv")
    mlflow.end_run()
