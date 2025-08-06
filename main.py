from forecasting.future_forecasting import forecast_future_prices
from models.linear_regression_tuning import tune_ridge_regression
from preprocessing.pipeline import load_and_prepare_data
import pandas as pd

if __name__ == "__main__":
    print("📦 Loading data...")
    df = pd.read_csv("data/bbca.csv")

    print("🔧 Training best model (Linear Regression with tuning)...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data("BBCA.JK")
    model, best_params = tune_ridge_regression(X_train, y_train)

    print("🔮 Forecasting 30 days ahead...")
    forecast_df = forecast_future_prices(model, df, steps=30)

    forecast_df.to_csv("results/forecast_30_days.csv", index=False)
    print("✅ Forecast saved to results/forecast_30_days.csv")