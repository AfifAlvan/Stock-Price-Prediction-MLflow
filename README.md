python main.py


Script ini akan:

Mengambil data historis BBCA dari Yahoo Finance

Membersihkan data

Membuat fitur lag & rolling

Membagi data (train/val/test)

Melatih model Linear Regression (tuning)

Memprediksi harga 30 hari ke depan

2. Output
📁 data/cleaned.csv: data yang sudah dibersihkan

📁 data/features.csv: data dengan fitur baru

🧠 Model disimpan (opsional)

📊 Prediksi disimpan / ditampilkan via terminal



## 📊 Evaluasi Model

| Model                    | RMSE     | MAE      | R²      |
|--------------------------|----------|----------|---------|
| **Naive (Baseline)**     | 194.05   | 148.64   | 0.9098  |
| **Linear Regression**    |          |          |         |
| ├─ Train                 | 101.95   | 74.99    | 0.9946  |
| └─ Test                  | 146.69   | 110.22   | 0.9484  |
| **Linear Regression (Tuning)** | 147.02   | 110.34   | 0.9482  |
| **XGBoost**              |          |          |         |
| ├─ Train                 | 77.99    | 59.04    | 0.9969  |
| └─ Test                  | 320.02   | 239.57   | 0.7546  |
| **XGBoost (Tuning)**     | 892.39   | 721.57   | -0.9082 |
| **LightGBM**             |          |          |         |
| ├─ Train                 | 43.34    | 33.14    | 0.9988  |
| └─ Test                  | 888.13   | 720.53   | -0.8900 |
| **LightGBM (Tuning)**    | 888.73   | 719.48   | -0.8925 |
> 📌 **Catatan**:
> - **Linear Regression** memberikan hasil terbaik di test set (RMSE 146.69, R² 0.9484).
> - Model boosting (XGBoost & LightGBM) mengalami **overfitting signifikan**, terutama setelah tuning.
> - Model **naive** digunakan sebagai baseline pembanding.

🔮 Forecasting 30 Hari Kedepan
Fungsi forecast_future_prices(model, df, steps=30) digunakan untuk melakukan prediksi harga berdasarkan hasil pelatihan model.



📦 Dependencies Utama
scikit-learn
xgboost
lightgbm
pandas
numpy
matplotlib
yfinance
mplfinance