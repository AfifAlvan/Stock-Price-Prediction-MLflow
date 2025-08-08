import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import mlflow

def plot_actual_vs_predicted(y_true, y_pred, model_name, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true.values, label="Actual", linewidth=2)
    plt.plot(y_pred, label="Predicted", linewidth=2)
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(save_dir, f"{model_name}_actual_vs_pred.png")
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path, artifact_path="plots")

    return path