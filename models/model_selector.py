import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from models.naive import baseline_naive
from models.linear_regression_tuning import tune_linear_regression
from models.linear_regression import train_linear_regression
from models.lightgbm_tuning import tune_lightgbm
from models.lightgbm import train_lightgbm
from models.xgboost_tuning import tune_xgboost
from models.xgboost import train_xgboost, evaluate_model
from preprocessing.pipeline import load_and_prepare_data

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.plotting import plot_actual_vs_predicted  # ✅ Import dari utils

def get_best_model(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    # X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(
    #     ticker="BBCA.JK",
    #     use_validation=True
    # )
    candidates = []

    # ===== Baseline Naive ====
    print("\n🔍 Running Baseline Naive ...")

    # Gabungkan X_test dan y_test jadi DataFrame
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    # Pastikan ada kolom lag_close_1
    # Kalau belum ada, buat dari y_train/y_test
    # test_df["lag_close_1"] = pd.concat([y_train.tail(1), y_test[:-1]]).values

    result = baseline_naive(test_df)

    mlflow.log_params({"Model": "Baseline_Naive"})
    mlflow.log_metrics({
        "Naive_RMSE": result["rmse"],
        "Naive_MAE": result["mae"],
        "Naive_R2": result["r2"]
    })

    plot_actual_vs_predicted(
        pd.Series(result["y_true"]),
        pd.Series(result["y_pred"]),
        "Baseline Naive"
    )
    candidates.append(("Baseline Naive", None, result["rmse"], result["mae"], result["r2"]))
    print(f"   → RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}, R2: {result['r2']:.4f}")

    # # ===== Linear Regression ====
    print("🔍 Linear Regression...")

    # Train linear regression biasa dan langsung dapat hasil lengkap
    result_lr = train_linear_regression(X_train, y_train, X_test, y_test)
    model_lr = result_lr["model"]
    rmse_lr = result_lr["rmse"]
    mae_lr = result_lr["mae"]
    r2_lr = result_lr["r2"]
    y_pred_lr = result_lr["y_pred"]

    # Log params dari model langsung (jika mau log param model sklearn)
    mlflow.log_params({f"LR_{k}": v for k, v in model_lr.get_params().items()})

    # Log metrik
    mlflow.log_metrics({
        "LR_RMSE": rmse_lr,
        "LR_MAE": mae_lr,
        "LR_R2": r2_lr
    })

    # Plot hasil
    plot_actual_vs_predicted(y_test, y_pred_lr, "LinearRegression")
    # Simpan kandidat
    candidates.append(("LinearRegression", model_lr, rmse_lr, mae_lr, r2_lr))

    # ===== Linear Regression Tuning=====
    print("\n🔍 Training Linear Regression Tuning...")
    model_lr_tuned, params_lr_tuned = tune_linear_regression(X_train, y_train)

    y_pred_lr_tuned = model_lr_tuned.predict(X_test)
    rmse_lr_tuned = np.sqrt(mean_squared_error(y_test, y_pred_lr_tuned))
    mae_lr_tuned = mean_absolute_error(y_test, y_pred_lr_tuned)
    r2_lr_tuned = r2_score(y_test, y_pred_lr_tuned)

    # Log ke MLflow
    mlflow.log_params({f"LR_Tuned_{k}": v for k, v in params_lr_tuned.items()})
    mlflow.log_metrics({
        "LR_Tuned_RMSE": rmse_lr_tuned,
        "LR_Tuned_MAE": mae_lr_tuned,
        "LR_Tuned_R2": r2_lr_tuned
    })

    # Plot hasil
    plot_actual_vs_predicted(y_test, y_pred_lr_tuned, "Linear Regression Tuning")

    # Simpan kandidat
    candidates.append(("Linear Regression Tuning ", model_lr_tuned, rmse_lr_tuned, mae_lr_tuned, r2_lr_tuned))


    # ===== XGBoost =====
    print("\n🔍 Training XGBoost...")
    model_xgb = train_xgboost(X_train, y_train)
    y_pred_xgb, metrics_xgb = evaluate_model(model_xgb, X_test, y_test)

    rmse_xgb = metrics_xgb["rmse"]
    mae_xgb = metrics_xgb["mae"]
    r2_xgb = metrics_xgb["r2"]

    mlflow.log_params({f"XGB_{k}": v for k, v in model_xgb.get_params().items()})
    mlflow.log_metrics({
        "XGB_RMSE": rmse_xgb,
        "XGB_MAE": mae_xgb,
        "XGB_R2": r2_xgb
    })

    plot_actual_vs_predicted(y_test, y_pred_xgb, "XGBoost")
    candidates.append(("XGBoost", model_xgb, rmse_xgb, mae_xgb, r2_xgb))


    # ===== XGBoost Tuning=====
    print("\n🔍 Training XGBoost (Tuning)...")
    model_xgb_tuned, params_xgb_tuned = tune_xgboost(X_train, y_train)
    y_pred_xgb_tuned = model_xgb_tuned.predict(X_test)
    rmse_xgb_tuned = np.sqrt(mean_squared_error(y_test, y_pred_xgb_tuned))
    mae_xgb_tuned = mean_absolute_error(y_test, y_pred_xgb_tuned)
    r2_xgb_tuned = r2_score(y_test, y_pred_xgb_tuned)

    # Log ke MLflow
    mlflow.log_params({f"XGB_Tuned_{k}": v for k, v in params_xgb_tuned.items()})
    mlflow.log_metrics({
        "XGB_Tuned_RMSE": rmse_xgb_tuned,
        "XGB_Tuned_MAE": mae_xgb_tuned,
        "XGB_Tuned_R2": r2_xgb_tuned
    })

    # Plot hasil
    plot_actual_vs_predicted(y_test, y_pred_xgb_tuned, "XGBoost (Tuning)")

    # Simpan kandidat
    candidates.append(("XGBoost (Tuning)", model_xgb_tuned, rmse_xgb_tuned, mae_xgb_tuned, r2_xgb_tuned))

    # ===== LightGBM =====
    print("\n🔍 Training LightGBM...")
    model_lgbm = train_lightgbm(X_train, y_train, X_val, y_val)
    y_pred_lgbm = model_lgbm.predict(X_test)

    rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
    mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
    r2_lgbm = r2_score(y_test, y_pred_lgbm)

    mlflow.log_params({f"LGBM_{k}": v for k, v in model_lgbm.get_params().items()})
    mlflow.log_metrics({
        "LGBM_RMSE": rmse_lgbm,
        "LGBM_MAE": mae_lgbm,
        "LGBM_R2": r2_lgbm
    })

    plot_actual_vs_predicted(y_test, y_pred_lgbm, "LightGBM")
    candidates.append(("LightGBM", model_lgbm, rmse_lgbm, mae_lgbm, r2_lgbm))


    # ===== LightGBM Tuning =====
    print("\n🔍 Training LightGBM (Tuning)...")
    model_lgbm_tuned, params_lgbm_tuned = tune_lightgbm(X_train, y_train)

    y_pred_lgbm_tuned = model_lgbm_tuned.predict(X_test)
    rmse_lgbm_tuned = np.sqrt(mean_squared_error(y_test, y_pred_lgbm_tuned))
    mae_lgbm_tuned = mean_absolute_error(y_test, y_pred_lgbm_tuned)
    r2_lgbm_tuned = r2_score(y_test, y_pred_lgbm_tuned)

    mlflow.log_params({f"LGBM_Tuned_{k}": v for k, v in params_lgbm_tuned.items()})
    mlflow.log_metrics({
        "LGBM_Tuned_RMSE": rmse_lgbm_tuned,
        "LGBM_Tuned_MAE": mae_lgbm_tuned,
        "LGBM_Tuned_R2": r2_lgbm_tuned
    })

    plot_actual_vs_predicted(y_test, y_pred_lgbm_tuned, "LightGBM (Tuning)")
    candidates.append(("LightGBM (Tuning)", model_lgbm_tuned, rmse_lgbm_tuned, mae_lgbm_tuned, r2_lgbm_tuned))


    # ===== Pilih best model =====
    best = min(candidates, key=lambda x: x[2])
    print(f"\n🏆 Best model: {best[0]}")
    print(f"   → RMSE: {best[2]:.4f}")
    print(f"   → MAE : {best[3]:.4f}")
    print(f"   → R²  : {best[4]:.4f}")

    # Log best model ke MLflow
    mlflow.log_param("Best_Model", best[0])
    mlflow.log_metrics({
        "Best_RMSE": best[2],
        "Best_MAE": best[3],
        "Best_R2": best[4]
    })
    mlflow.sklearn.log_model(best[1], "Best_Model_File")

    # Simpan best model ke lokal
    os.makedirs("outputs/results", exist_ok=True)
    local_model_path = os.path.join("outputs/results", "best_model.pkl")
    joblib.dump(best[1], local_model_path)

    # Log file lokal ke MLflow
    mlflow.log_artifact(local_model_path, artifact_path="models")

    return best[0], best[1], best[2], best[3], best[4]
