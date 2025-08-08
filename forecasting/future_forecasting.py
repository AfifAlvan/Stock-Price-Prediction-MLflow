import pandas as pd
from preprocessing.pipeline import generate_features

import pandas as pd

def forecast_future_prices(model, last_data, steps=30):
    forecast = []

    # Pastikan Date bertipe datetime sekali di awal
    last_data["Date"] = pd.to_datetime(last_data["Date"])
    current_data = last_data.copy()

    for _ in range(steps):
        # Pastikan Date tetap datetime
        current_data["Date"] = pd.to_datetime(current_data["Date"])

        # Skip cleaning saat forecast
        features = generate_features(current_data, clean=False, verbose=False)

        latest_features = features.iloc[[-1]]
        drop_cols = [col for col in ["Date", "target"] if col in latest_features.columns]
        latest_features = latest_features.drop(columns=drop_cols)

        next_price = model.predict(latest_features)[0]

        last_date = current_data["Date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)

        # Buat baris baru (isi Volume > 0 supaya aman)
        new_row = {"Date": next_date, "Close": next_price, "Volume": 1}
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)

        forecast.append((next_date, next_price))

    return pd.DataFrame(forecast, columns=["Date", "Predicted_Close"])
