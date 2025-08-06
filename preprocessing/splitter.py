# preprocessing/splitter.py

def time_series_train_test_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def split_X_y(df, target_col="target"):
    X = df.drop(columns=[target_col, "Date"])
    y = df[target_col]
    return X, y
