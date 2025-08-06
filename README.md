
# 📈 BBCA Stock Price Forecasting

`python main.py`

Proyek ini melakukan prediksi harga saham **BBCA** dengan pipeline lengkap: pengambilan data, pembersihan, pembuatan fitur, pelatihan model, evaluasi performa, dan forecasting 30 hari ke depan.

---

## 🚀 Fitur Utama

Script akan secara otomatis:

1. Mengambil data historis BBCA dari Yahoo Finance
2. Membersihkan data (handling null, duplikat, format tanggal)
3. Membuat fitur:
   - Lag (Close[-1])
   - Rolling mean (Close)
4. Membagi data menjadi:
   - Train
   - Validation
   - Test
5. Melatih model:
   - Linear Regression (dengan & tanpa tuning)
   - XGBoost
   - LightGBM
6. Melakukan prediksi harga saham ke depan (30 hari)

---

## 📂 Struktur Output

| File / Folder         | Deskripsi                                   |
|-----------------------|---------------------------------------------|
| `data/cleaned.csv`    | Data yang sudah dibersihkan                 |
| `data/features.csv`   | Data dengan fitur lag & rolling             |
| `actual.csv`          | Hasil pengambilan harga aktual terbaru      |
| `model/`              | Folder untuk menyimpan model       |
| `mlruns/`             | Log experiment dari MLflow  |

---

## 📊 Evaluasi Model

| Model                      | RMSE     | MAE      | R²      |
|----------------------------|----------|----------|---------|
| **Naive (Baseline)**       | 194.05   | 148.64   | 0.9098  |
| **Linear Regression** ✅     |          |          |         |
|  Train                   | 101.95   | 74.99    | 0.9946  |
|  Test                    | 146.69   | 110.22   | 0.9484  |
| **Linear Regression (Tuned)** | 147.02   | 110.34   | 0.9482  |
| **XGBoost**                |          |          |         |
|  Train                   | 77.99    | 59.04    | 0.9969  |
|  Test                    | 320.02   | 239.57   | 0.7546  |
| **XGBoost (Tuned)**        | 892.39   | 721.57   | -0.9082 |
| **LightGBM**               |          |          |         |
|  Train                   | 43.34    | 33.14    | 0.9988  |
|  Test                    | 888.13   | 720.53   | -0.8900 |
| **LightGBM (Tuned)**       | 888.73   | 719.48   | -0.8925 |


<img width="1000" height="500" alt="linear vs actual test" src="https://github.com/user-attachments/assets/1dacab33-b8ae-481f-9b4d-9357d4b5201c" />



> 📌 **Catatan:**
>
> ✅ Linear Regression menghasilkan performa terbaik dan paling stabil antara data train dan test.
>
> ⚠️ XGBoost dan LightGBM menunjukkan indikasi overfitting setelah tuning, terlihat dari selisih performa yang signifikan antara data train dan test.
>
>🔹 Model Naive (Baseline) digunakan sebagai acuan awal untuk membandingkan kinerja model-model lainnya.
---

## 🔮 Forecasting 30 Hari ke Depan

Fungsi `forecast_future_prices(model, df, steps=30)` digunakan untuk melakukan prediksi harga berdasarkan model yang telah dilatih.

---

## 📦 Dependencies Utama

```
scikit-learn
xgboost
lightgbm
pandas
numpy
matplotlib
yfinance
mplfinance
mlflow
```
## 🚀 Cara Menjalankan

### 1. Clone Repo
```
git clone https://github.com/AfifAlvan/Stock-Price-Prediction-MLflow.git
cd Stock-Price-Prediction-MLflow
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Menjalankan Model
```
# Naive Forecast (baseline)
python -m models.naive

# Linear Regression
python -m models.linear_regression

# Linear Regression (Tuning)
python -m models.linear_regression_tuning

# XGBoost
python -m models.xgboost

# XGBoost (Tuning)
python -m models.xgboost_tuning

# LightGBM
python -m models.lightgbm

# LightGBM (Tuning)
python -m models.lightgbm_tuning
```
