# preprocessing/pipeline.py

from data_loader.data_loader import get_data
from preprocessing.cleaner import cleaning
from preprocessing.feature_engineer import create_features
from preprocessing.splitter import time_series_train_test_split, split_X_y
import os

def load_and_prepare_data(
    ticker="BBCA.JK",
    test_size=0.2,
    val_size=0.1,
    lags=[1, 2, 3],
    rolling_windows=[5, 10],
    save_dir="data",
    use_validation=True
):
    print("📥 Loading raw data...")
    raw_data = get_data(ticker)
    cleaned_data = cleaning(raw_data)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cleaned_data.to_csv(f"{save_dir}/cleaned.csv", index=False)

    print("🛠️  Creating features...")
    feature_df = create_features(cleaned_data, lags=lags, rolling_windows=rolling_windows)

    if save_dir:
        feature_df.to_csv(f"{save_dir}/features.csv", index=False)

    # Split test set
    trainval_df, test_df = time_series_train_test_split(feature_df, test_size=test_size)
    X_test, y_test = split_X_y(test_df)

    if use_validation:
        # Further split train into train/val
        train_df, val_df = time_series_train_test_split(trainval_df, test_size=val_size)
        X_train, y_train = split_X_y(train_df)
        X_val, y_val = split_X_y(val_df)

        print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    else:
        # No validation
        X_train, y_train = split_X_y(trainval_df)

        print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test
    
# def generate_features(df, lags=[1, 2, 3], rolling_windows=[5, 10]):
#     """
#     Bersihkan dan buat fitur dari dataframe mentah (misal: data terbaru untuk prediksi ke depan).
#     """
#     from preprocessing.cleaner import cleaning
#     from preprocessing.feature_engineer import create_features

#     cleaned = cleaning(df)
#     features = create_features(cleaned, lags=lags, rolling_windows=rolling_windows)
#     return features

def generate_features(df, lags=[1, 2, 3], rolling_windows=[5, 10], clean=True, verbose=True):
    from preprocessing.cleaner import cleaning
    from preprocessing.feature_engineer import create_features

    if clean:
        cleaned = cleaning(df, verbose=verbose)
    else:
        cleaned = df.copy()

    features = create_features(cleaned, lags=lags, rolling_windows=rolling_windows)
    return features
