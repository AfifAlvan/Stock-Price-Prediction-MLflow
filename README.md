# Stock Price Forcasting

## Deskripsi
Project ini bertujuan untuk mengambil data harga saham dari Indonesia (contoh: BBCA.JK) menggunakan ```yfinance```, melakukan pembersihan data, ekstraksi fitur time series, pelatihan model machine learning untuk memprediksi harga closing price pada hari berikutnya **(next trading day)**, serta melakukan pemantauan eksperimen dan manajemen model dengan MLflow.

---

## Tahapan dan Cara Menjalankan

1. **Install dependencies**

   Pastikan kamu sudah punya Python (disarankan versi 3.7 ke atas), lalu jalankan perintah berikut untuk install semua library yang diperlukan:

   ```bash
   pip install -r requirements.txt

2. **Download dan simpan data**

    ```bash
    python -m data_loader.data_loader
    ```
   Data akan tersimpan di folder ```data/``` dengan nama file ```bbca.csv``` (atau ticker lain sesuai pengaturan).
    
3. **Menjalankan Pipeline**
   
   Jalankan pipeline lengkap berikut:
   
   ```bash
    python main.py

Pipeline ini akan melakukan:

* Pembersihan data (cleaning)
* Ekstraksi fitur lag dan rolling window (feature engineering)
* Pembagian data train, validation, dan test (time-series split)
* Pelatihan beberapa model regresi (linear regression, XGBoost, LightGBM)
* Evaluasi model menggunakan metrik RMSE, MAE, dan R²
* Pemilihan model terbaik secara otomatis berdasarkan metrik utama (RMSE)
* Prediksi harga penutupan untuk hari berikutnya (next day)
* Logging eksperimen dan model ke MLflow
* Menyimpan model terbaik dan hasil prediksi

## Detail Preprocessing dan Feature Engineering

### 🧹 Data Cleaning

Langkah-langkah yang dilakukan:
* Konversi kolom tanggal menjadi tipe datetime
* Urutkan data berdasarkan tanggal
* Hapus duplikat
* Tangani missing values (cek dan isi/hapus)
* Cek nilai negatif atau nol pada harga dan volume
* Laporkan dan tangani anomali harga jika ada
* Pastikan kolom numerik memiliki tipe numerik
* Cek tanggal bisnis yang hilang
* Hapus baris dengan volume perdagangan <= 0

### ⚙️ Feature Engineering

Membuat fitur untuk forecasting harga closing hari berikutnya:

* Lag features: harga close pada hari sebelumnya (lag 1, 2, 3)
* Rolling window features: rata-rata dan statistik lain dalam window 5 dan 10 hari
* Target variabel: ```Close(t+1)```, yaitu harga close pada hari berikutnya

### 📈 Data Splitting
Data dibagi berdasarkan waktu (time series split) menjadi:

* Training set (~70-80%)
* Validation set (~10-15%, opsional)
* Test set (~10-20%)

Fungsi splitter memastikan data valid tanpa data leakage.

---
## **Output yang di hasilkan**
     
  📁Folder ```data/```
  * ```cleaned.csv```
  Data yang sudah dibersihkan dan siap dipakai untuk modeling.
  
  * ```feature.csv```
  Data hasil ekstraksi fitur lag (lagged features) untuk machine learning.
  
  📁 Folder ```mlruns/```
  
  * Folder hasil dari AutoML (MLflow) yang berisi tracking eksperimen model.
  
  📁 Folder ```outputs/plots/```
  
  * Semua grafik hasil evaluasi model, seperti perbandingan nilai aktual vs prediksi, disimpan dalam format gambar.

  📁Folder ```outputs/results/```
  
  * 🧠 ```best_model.pkl```
  Model machine learning terbaik yang sudah dilatih dan siap digunakan.
  
  * 📊```forecast_30_days.csv```
  File CSV berisi prediksi harga saham untuk 30 hari ke depan mulai dari hari terakhir data.

---


## 📊 Evaluasi Model

| Model                | RMSE    | MAE     | R²       |
|----------------------|---------|---------|----------|
| Naive (Baseline)     | 194.05  | 148.64  | 0.9098   |
| Linear Regression    | 147.25  | 110.51  | 0.9480   |
| XGBoost              | 866.55  | 699.77  | -0.7993  |
| XGBoost (Tuning)     | 900.44  | 729.50  | -0.9427  |
| LightGBM             | 883.11  | 715.66  | -0.8687  |
| LightGBM (Tuning)    | 897.99  | 728.87  | -0.9322  |

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/af992333-1d1a-4570-a8f1-a300895ad61b" />



> 📌 **Catatan**:
> - **Linear Regression** memberikan hasil terbaik di test set (RMSE 147.25, MAE 110.51, R² 0.9480).
> - Model boosting (XGBoost & LightGBM) mengalami **overfitting signifikan**, terutama setelah tuning.
> - Model **naive** digunakan sebagai baseline pembanding.


## Integrasi MLOps dengan MLflow

* Semua eksperimen model, metrik (RMSE, MAE, R²), parameter, dan artifact (plot evaluasi, model pkl) dicatat otomatis ke MLflow.
* Model terbaik dipilih dan diregistrasi otomatis berdasarkan metrik RMSE terkecil.
* Logging ini menjamin eksperimen dapat direproduksi dan mudah dievaluasi melalui UI MLflow.

> Untuk hasil MLflow lebih lengkapnya bisa dilihat di **Documentation MLFlow Stock Forecasting.pdf**

## 📦 Dependencies Utama
```
pandas
numpy
yfinance
scikit-learn
matplotlib
xgboost
lightgbm
yfinance
mplfinance
```

### For more information
* Linkedin : https://www.linkedin.com/in/afif-alvan/
* Youtube  : https://www.youtube.com/@afifalvan
* Medium   : https://medium.com/@afifalvan

