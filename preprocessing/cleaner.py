import pandas as pd

def cleaning(df):
    """
    Membersihkan data saham:
    - Konversi tipe tanggal
    - Urutkan berdasarkan tanggal
    - Hapus duplikat
    - Cek dan tangani missing values
    - Cek nilai negatif atau nol
    - Cek dan laporkan anomali harga
    - Pastikan semua kolom numerik bertipe numerik
    - Cek tanggal bisnis yang hilang
    - Hapus baris dengan Volume <= 0
    
    Return:
        df_bersih: DataFrame yang telah dibersihkan
    """
    df = df.copy()

    # Konversi tanggal dan urutkan
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Hapus duplikat
    df = df.drop_duplicates()

    # Ubah kolom numerik ke float
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Tangani missing values (NaN)
    if df.isnull().sum().sum() > 0:
        print("Missing values found, applying forward fill and backward fill...")
        df = df.fillna(method='ffill').fillna(method='bfill')

    # Cek nilai negatif atau nol
    if (df[cols_to_numeric] <= 0).any().any():
        print("Warning: Terdapat nilai <= 0 di kolom harga/volume.")
        print((df[cols_to_numeric] <= 0).sum())

    # Cek anomali harga (Close tidak boleh < Low atau > High, Low tidak boleh > High)
    epsilon = 1e-6
    anomalies = df[
        (df['Close'] - df['High'] > epsilon) |
        (df['Low'] - df['Close'] > epsilon) |
        (df['Low'] - df['High'] > epsilon)
    ]
    if not anomalies.empty:
        print(f"Ada {len(anomalies)} baris dengan anomali harga:")
        print(anomalies)

    # Cek apakah ada tanggal bisnis yang hilang (tidak dihapus, hanya info)
    business_days = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='B')
    missing_dates = business_days.difference(df['Date'])
    if len(missing_dates) > 0:
        print(f"Terdapat {len(missing_dates)} tanggal bisnis yang hilang (mungkin hari libur):")
        print(missing_dates.strftime('%Y-%m-%d').tolist())

    # Tampilkan baris dengan Volume <= 0
    invalid_volume_rows = df[df['Volume'] <= 0]
    if not invalid_volume_rows.empty:
        print("Baris dengan Volume <= 0:")
        print(invalid_volume_rows)

    print(f"Jumlah baris dengan Volume <= 0 yang akan dihapus: {(df['Volume'] <= 0).sum()}")

    # Hapus baris dengan Volume <= 0
    df = df[df['Volume'] > 0].reset_index(drop=True)

    return df
