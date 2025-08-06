
def create_features(df, lags=[1, 2, 3], rolling_windows=[5, 10]):
    """
    Membuat fitur lag dan rolling untuk modeling time series regression.
    Target: Close(t+1)
    """
    df = df.copy()

    # Pastikan data terurut
    df = df.sort_values("Date").reset_index(drop=True)

    # LAG features
    for lag in lags:
        df[f"lag_close_{lag}"] = df["Close"].shift(lag)

    # Rolling mean & std
    for window in rolling_windows:
        df[f"roll_mean_{window}"] = df["Close"].shift(1).rolling(window=window).mean()
        df[f"roll_std_{window}"] = df["Close"].shift(1).rolling(window=window).std()

    # Target: Close price esok hari
    df["target"] = df["Close"].shift(-1)

    # Drop baris dengan NaN (dari shifting/rolling)
    df = df.dropna().reset_index(drop=True)
    
    return df
