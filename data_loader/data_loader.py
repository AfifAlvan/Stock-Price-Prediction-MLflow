import yfinance as yf
import pandas as pd
import os

def get_data(ticker, start="2020-01-01", end="2025-08-01"):
    # Unduh data dari Yahoo Finance
    data = yf.download(ticker, start=start, end=end)
    
    # Tangani MultiIndex jika ada
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index
    data.reset_index(inplace=True)

    # Pilih kolom yang dibutuhkan
    columns_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data = data[[col for col in columns_order if col in data.columns]]

    # Pastikan folder 'data' ada
    os.makedirs("data", exist_ok=True)

    # Simpan data ke file CSV di folder 'data' dengan nama 'bbca.csv'
    file_path = os.path.join("data", f"{ticker.lower().split('.')[0]}.csv")
    data.to_csv(file_path, index=False)

    print(f"Data untuk {ticker} telah disimpan di {file_path}")

    return data

if __name__ == "__main__":
    get_data("BBCA.JK")