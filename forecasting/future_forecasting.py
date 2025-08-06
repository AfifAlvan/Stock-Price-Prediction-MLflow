import pandas as pd
from preprocessing.pipeline import generate_features

def forecast_future_prices(model, last_data, steps=30):
    forecast = []
    current_data = last_data.copy()

    for _ in range(steps):
        # Buat fitur dari data saat ini
        features = generate_features(current_data)
        latest_features = features.iloc[[-1]]  # Ambil baris terakhir

        # Drop kolom yang tidak dipakai saat training
        if "Date" in latest_features.columns:
            latest_features = latest_features.drop(columns=["Date"])
        if "target" in latest_features.columns:
            latest_features = latest_features.drop(columns=["target"])

        # Prediksi harga berikutnya
        next_price = model.predict(latest_features)[0]

        # Buat tanggal berikutnya
        last_date = pd.to_datetime(current_data["Date"].iloc[-1])
        next_date = last_date + pd.Timedelta(days=1)

        # Tambahkan prediksi ke data agar bisa dipakai di langkah selanjutnya
        new_row = {"Date": next_date, "Close": next_price}
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)

        # Simpan hasil forecast
        forecast.append((next_date, next_price))

    return pd.DataFrame(forecast, columns=["Date", "Predicted_Close"])
